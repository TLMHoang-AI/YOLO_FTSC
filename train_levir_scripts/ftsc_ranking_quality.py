"""Streaming, dependency-light FTSC within-GT ranking diagnostics.

The accumulator consumes detached positive-level snapshots emitted by
``AnchorFreeFTSCCalibrator``.  It never calls the model and therefore cannot
change TAL assignment, loss values, gradients, or inference behavior.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def rankdata_average(values: Iterable[float]) -> np.ndarray:
    """Average ranks with deterministic tie handling, without SciPy."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_tie_aware(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return Spearman rho, or ``None`` for singleton/constant/invalid groups."""
    x, y = np.asarray(list(left), dtype=np.float64), np.asarray(list(right), dtype=np.float64)
    if x.size < 2 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    xr, yr = rankdata_average(x), rankdata_average(y)
    if np.isclose(xr.std(), 0.0) or np.isclose(yr.std(), 0.0):
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def effective_number(weights: Iterable[float]) -> float:
    values = np.asarray(list(weights), dtype=np.float64)
    if values.size == 0:
        return float("nan")
    denominator = float(np.square(values).sum())
    return float(np.square(values.sum()) / denominator) if denominator > 0 else float("nan")


@dataclass
class _State:
    groups: int = 0
    singleton: int = 0
    support_counts: list[int] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    undefined: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def consume_metric(self, name: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            self.undefined[name] += 1
        else:
            self.metrics[name].append(float(value))


class FTSCRankingAccumulator:
    """Aggregate compact per-GT and per-level diagnostics from audit snapshots."""

    CORRELATIONS = {
        "position_vs_iou": ("position_log_evidence_centered", "audit_decoded_iou"),
        "dfl_vs_iou": ("dfl_log_evidence_centered", "audit_decoded_iou"),
        "w_cls_vs_iou": ("w_cls", "audit_decoded_iou"),
        "w_box_vs_iou": ("w_box", "audit_decoded_iou"),
        "w_dfl_vs_iou": ("w_dfl", "audit_decoded_iou"),
        "tal_vs_iou": ("tal_target_score", "audit_decoded_iou"),
        "w_cls_vs_tal": ("w_cls", "tal_target_score"),
        "position_vs_tal": ("position_log_evidence_centered", "tal_target_score"),
        "dfl_vs_tal": ("dfl_log_evidence_centered", "tal_target_score"),
    }

    def __init__(self, *, sample_limit: int = 0, sample_seed: int = 42) -> None:
        self._states: dict[str, _State] = {"all": _State()}
        self._batch_id = 0
        self._sample_limit = max(int(sample_limit), 0)
        self._sample_rng = np.random.default_rng(int(sample_seed))
        self._sample: list[dict[str, float | int | str]] = []
        self._sample_seen = 0

    @staticmethod
    def _array(snapshot: dict[str, object], name: str, size: int, default: float = float("nan")) -> np.ndarray:
        value = snapshot.get(name)
        if value is None:
            return np.full(size, default, dtype=np.float64)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1)

    def _state(self, level: str) -> _State:
        if level not in self._states:
            self._states[level] = _State()
        return self._states[level]

    def _maybe_sample(self, row: dict[str, float | int | str]) -> None:
        if not self._sample_limit:
            return
        self._sample_seen += 1
        if len(self._sample) < self._sample_limit:
            self._sample.append(row)
            return
        slot = int(self._sample_rng.integers(0, self._sample_seen))
        if slot < self._sample_limit:
            self._sample[slot] = row

    def _consume_group(self, values: dict[str, np.ndarray], indices: np.ndarray, level: str) -> None:
        n = int(indices.size)
        state = self._state(level)
        state.groups += 1
        state.support_counts.append(n)
        if n == 1:
            state.singleton += 1
        group = {name: array[indices] for name, array in values.items()}
        for name, (left, right) in self.CORRELATIONS.items():
            state.consume_metric(name, spearman_tie_aware(group[left], group[right]))

        for task in ("cls", "box", "dfl"):
            weights = group[f"w_{task}"]
            finite = weights[np.isfinite(weights)]
            if finite.size:
                state.metrics[f"weight_{task}_mean"].append(float(finite.mean()))
                state.metrics[f"weight_{task}_std"].append(float(finite.std()))
                state.metrics[f"weight_{task}_min"].append(float(finite.min()))
                state.metrics[f"weight_{task}_max"].append(float(finite.max()))
                state.metrics[f"weight_{task}_range"].append(float(finite.max() - finite.min()))
                mean = float(finite.mean())
                state.metrics[f"weight_{task}_cv"].append(float(finite.std() / abs(mean)) if abs(mean) > 1e-12 else float("nan"))
                state.metrics[f"weight_{task}_near1_lt001"].append(float(np.mean(np.abs(finite - 1.0) < 0.01)))
                state.metrics[f"weight_{task}_near1_lt005"].append(float(np.mean(np.abs(finite - 1.0) < 0.05)))
                state.metrics[f"weight_{task}_clipped_fraction"].append(float(np.mean(
                    (group[f"w_{task}_clipped_low"] > 0.5) | (group[f"w_{task}_clipped_high"] > 0.5)
                )))
                n_eff = effective_number(finite)
                state.metrics[f"n_eff_{task}"].append(n_eff)
                state.metrics[f"n_eff_ratio_{task}"].append(n_eff / n)

        iou = group["audit_decoded_iou"]
        if np.isfinite(iou).any():
            state.metrics["audit_decoded_iou_mean"].append(float(np.nanmean(iou)))
        entropy = group.get("dfl_entropy_mean", np.array([], dtype=np.float64))
        if entropy.size and np.isfinite(entropy).any():
            state.metrics["dfl_entropy_mean"].append(float(np.nanmean(entropy)))

        if n >= 2 and np.isfinite(iou).all():
            best_iou = int(np.argmax(iou))
            for name, evidence in {
                "position": "position_log_evidence_centered",
                "dfl": "dfl_log_evidence_centered",
                "ftsc_cls": "w_cls",
                "ftsc_box": "w_box",
                "ftsc_dfl": "w_dfl",
                "tal": "tal_target_score",
            }.items():
                evidence_values = group[evidence]
                metric_name = f"top1_{name}_iou_agreement"
                if np.isfinite(evidence_values).all():
                    self._state(level).metrics[metric_name].append(
                        float(int(np.argmax(evidence_values)) == best_iou)
                    )
                else:
                    self._state(level).undefined[metric_name] += 1
            for name, evidence in {
                "position": "position_log_evidence_centered",
                "dfl": "dfl_log_evidence_centered",
                "ftsc_cls": "w_cls",
                "ftsc_box": "w_box",
                "ftsc_dfl": "w_dfl",
                "tal": "tal_target_score",
            }.items():
                evidence_values = group[evidence]
                metric_name = f"pairwise_{name}_iou_order_accuracy"
                if not np.isfinite(evidence_values).all():
                    self._state(level).undefined[metric_name] += 1
                    continue
                order = evidence_values[:, None] - evidence_values[None, :]
                target = iou[:, None] - iou[None, :]
                mask = np.triu(np.ones((n, n), dtype=bool), 1) & (order != 0)
                if mask.any():
                    state.metrics[metric_name].append(
                        float(np.mean(np.sign(order[mask]) == np.sign(target[mask])))
                    )

        if self._sample_limit and level == "all":
            for j in range(n):
                self._maybe_sample({
                    "batch_id": self._batch_id,
                    "image_index": int(values["image_index"][indices[j]]),
                    "gt_id": int(values["gt_id"][indices[j]]),
                    "positive_index": int(values["positive_index"][indices[j]]),
                    "stride": float(values["stride"][indices[j]]),
                    "gt_positive_count": int(values["gt_positive_count"][indices[j]]),
                    **{key: float(array[indices[j]]) for key, array in values.items()
                       if key not in {"image_index", "gt_id", "positive_index", "stride", "gt_positive_count"}},
                })

    def update(self, snapshot: dict[str, object]) -> None:
        """Consume one detached positive snapshot emitted by the criterion."""
        if not snapshot:
            return
        size = len(next(iter(snapshot.values())))
        if size == 0:
            return
        values = {name: self._array(snapshot, name, size) for name in snapshot}
        required = {"image_index", "gt_id", "stride", "audit_decoded_iou"}
        missing = required - set(values)
        if missing:
            raise ValueError(f"FTSC audit snapshot missing required columns: {sorted(missing)}")
        keys = np.stack((values["image_index"].astype(np.int64), values["gt_id"].astype(np.int64)), axis=1)
        for key in np.unique(keys, axis=0):
            indices = np.flatnonzero((keys == key).all(axis=1))
            strides = values["stride"][indices]
            unique_strides = set(np.round(strides).astype(int).tolist())
            level = "P2" if unique_strides == {4} else "P3" if unique_strides == {8} else "cross_level"
            self._consume_group(values, indices, level)
            self._consume_group(values, indices, "all") if level != "all" else None
        self._batch_id += 1

    @staticmethod
    def _quantiles(values: list[float]) -> tuple[float | None, float | None, float | None]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return None, None, None
        return tuple(float(v) for v in np.quantile(finite, [0.25, 0.5, 0.75]))

    def _rows(self, state: _State, metadata: dict[str, object]) -> list[dict[str, object]]:
        rows = []
        for metric in sorted(state.metrics):
            values = state.metrics[metric]
            finite = [float(value) for value in values if np.isfinite(value)]
            q25, median, q75 = self._quantiles(finite)
            rows.append({
                **metadata,
                "metric": metric,
                "valid_gt_count": len(finite),
                "undefined_gt_count": int(state.undefined.get(metric, 0)),
                "mean": float(np.mean(finite)) if finite else None,
                "median": median,
                "std": float(np.std(finite)) if finite else None,
                "q25": q25,
                "q75": q75,
            })
        rows.append({
            **metadata, "metric": "gt_support_count", "valid_gt_count": state.groups,
            "undefined_gt_count": 0, "mean": float(np.mean(state.support_counts)) if state.support_counts else None,
            "median": float(np.median(state.support_counts)) if state.support_counts else None,
            "std": float(np.std(state.support_counts)) if state.support_counts else None,
            "q25": float(np.quantile(state.support_counts, 0.25)) if state.support_counts else None,
            "q75": float(np.quantile(state.support_counts, 0.75)) if state.support_counts else None,
            "singleton_fraction": state.singleton / state.groups if state.groups else None,
            "support_2_4_fraction": sum(2 <= n <= 4 for n in state.support_counts) / state.groups if state.groups else None,
            "support_5plus_fraction": sum(n >= 5 for n in state.support_counts) / state.groups if state.groups else None,
            "positive_count": int(sum(state.support_counts)),
        })
        return rows

    def summary_rows(self, **metadata: object) -> list[dict[str, object]]:
        """Return compact all-GT metrics; undefined correlations are explicit."""
        return self._rows(self._states["all"], dict(metadata, level="all"))

    def level_rows(self, **metadata: object) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        total_positives = sum(self._states["all"].support_counts)
        for level, state in self._states.items():
            if level != "all":
                level_rows = self._rows(state, dict(metadata, level=level))
                for row in level_rows:
                    row["positive_count"] = int(sum(state.support_counts))
                    row["positive_share"] = (
                        float(sum(state.support_counts) / total_positives) if total_positives else None
                    )
                rows.extend(level_rows)
        return rows

    @property
    def sample(self) -> list[dict[str, float | int | str]]:
        return list(self._sample)

    def results_payload(self, **metadata: object) -> dict[str, object]:
        return {
            "provenance": dict(metadata),
            "summary": self.summary_rows(**metadata),
            "by_level": self.level_rows(**metadata),
            "positive_sample": self.sample,
            "groups": {level: state.groups for level, state in self._states.items()},
        }
