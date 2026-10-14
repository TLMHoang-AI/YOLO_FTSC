"""Pure, dataset-agnostic decision logic for FTSC six-core completion.

Never trains, evaluates, downloads or uploads. Scientific verifiers live in
dataset adapters. A missing archive marker is NOT a missing checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CORE = ("AP50", "mAP50-95", "AP50-Small", "parameters", "GFLOPs", "inference_ms")


def validate_six_core(metrics: Mapping[str, Any], keys: Mapping[str, str]) -> tuple[str, ...]:
    """Return missing/invalid semantic fields, using dataset-native producer keys."""
    invalid: list[str] = []
    for name in CORE:
        key = keys[name]
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            invalid.append(name)
        elif name in ("parameters", "GFLOPs", "inference_ms") and value <= 0:
            invalid.append(name)
        elif name not in ("parameters", "GFLOPs", "inference_ms") and not 0 <= value <= 1:
            invalid.append(name)
    return tuple(invalid)


@dataclass(frozen=True)
class CompletionDecision:
    dataset: str
    case: str
    seed: int
    action: str
    reason: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return vars(self).copy()


def decide(*, dataset: str, case: str, seed: int,
           inventory_successful: bool, candidates: Sequence[Mapping[str, Any]],
           existing_metrics: Mapping[str, Any] | None,
           metric_keys: Mapping[str, str], not_applicable: bool = False) -> CompletionDecision:
    """Accept only preverified candidates (each with 'verified' and 'sha256').

    No candidate following a successful exhaustive index: TRAIN_REQUIRED.
    Any unverified candidate, hash conflict, or HF failure: BLOCKED.
    """
    base = dict(dataset=dataset, case=case, seed=seed)
    if not_applicable:
        return CompletionDecision(**base, action="NOT_APPLICABLE", reason="Case excluded by dataset protocol")
    if not inventory_successful:
        return CompletionDecision(**base, action="BLOCKED_HF_UNAVAILABLE", reason="HF inventory unavailable")
    if not candidates:
        return CompletionDecision(**base, action="TRAIN_REQUIRED", reason="No HF checkpoint candidate in registered namespaces")
    if any(not c.get("verified") for c in candidates):
        return CompletionDecision(**base, action="BLOCKED_UNVERIFIED",
                                  reason="One or more HF candidates lack verified scientific provenance")
    hashes = {c.get("sha256") for c in candidates}
    if len(candidates) > 1 and (None in hashes or len(hashes) != 1):
        return CompletionDecision(**base, action="BLOCKED_CONFLICT",
                                  reason="Multiple scientifically plausible checkpoints have different SHA-256")
    chosen = sorted(candidates, key=lambda c: c["checkpoint_path"])[0]
    identity = dict(checkpoint_path=chosen["checkpoint_path"],
                    checkpoint_sha256=chosen.get("sha256"))
    if existing_metrics is not None and not validate_six_core(existing_metrics, metric_keys):
        if chosen.get("sha256") and existing_metrics.get("checkpoint_sha256") == chosen["sha256"] and existing_metrics.get("evaluator_verified") is True:
            return CompletionDecision(**base, **identity, action="SKIP_COMPLETE", reason="Verified six-core metrics")
    return CompletionDecision(**base, **identity, action="EVAL_ONLY",
                              reason="Verified checkpoint; metric completion/revalidation required")
