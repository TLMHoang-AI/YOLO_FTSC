"""Canonical FTSC Hub transport for the reorganized dataset repository.

This is a strict adapter around the pre-migration HFResultArchive:
- train uploads go to Dataset/category/experiment/case/seed;
- existing checkpoint directories are immutable;
- source recovery uses audited checkpoint_reference.json and SHA-256;
- re-evaluation uploads go to Dataset/derived_evaluations/<suite>/...,
  never into an existing checkpoint's weights or original metrics.

No network action occurs on import. No token is stored in source.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.hf_result_archive import (
    DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
    HFResultArchive,
    HFResultArchiveError,
    MARKER_FILENAME,
)
from utils.ftsc_hf_layout import (
    REPO_ID,
    augmentation_run,
    augmentation_checkpoint,
    dataset_of_namespace,
    evaluation_root,
    prefix,
    seed_dir,
    segment,
    training_root,
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalFTSCArchive(HFResultArchive):
    """New layout with fail-closed publishing; legacy archive semantics preserved."""

    def _require_canonical_repo(self) -> None:
        if self.repo_id != REPO_ID or self.repo_type != "dataset":
            raise HFResultArchiveError(
                "Canonical FTSC transport is locked to the audited dataset repository"
            )

    @staticmethod
    def remote_path(dataset_slug: str, variant: str, run_name: str) -> str:
        case = variant
        if variant.startswith("historical__") and "journal_phase2" in dataset_slug:
            from utils.ftsc_hf_layout import dataset_of_namespace
            dataset = dataset_of_namespace(dataset_slug)
            root = prefix(f"{dataset}/derived_evaluations/journal_phase2")
        elif "cross_yolo_portability" in dataset_slug:
            parts = variant.split("__", 1)
            if len(parts) != 2:
                raise HFResultArchiveError("Cross-YOLO variant must be family__case")
            family, case = map(segment, parts)
            root = training_root(dataset_slug, family=family)
        elif not (variant.startswith("historical__") and "journal_phase2" in dataset_slug):
            root = training_root(dataset_slug)
        return prefix(f"{root}/{segment(case)}/{segment(run_name)}")

    def publish_run(self, *, dataset_slug: str, variant: str, seed: int | str | None,
                    run_name: str, run_dir: str | Path,
                    required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
                    summary_metadata: Mapping[str, Any] | None = None,
                    summary_schema: str = "ftsc_seed_summary_v1") -> dict[str, Any]:
        """Refuse to clobber any existing path, even if its upload marker is absent."""
        path = self.remote_path(dataset_slug, variant, run_name)
        self._require_canonical_repo()
        self._assert_repository_access()
        files = set(self._retry(lambda: self._get_api().list_repo_files(
            repo_id=self.repo_id, repo_type=self.repo_type)))
        if any(file == path or file.startswith(path + "/") for file in files):
            raise HFResultArchiveError(f"Canonical HF run already exists; refusing overwrite: {path}")
        return super().publish_run(
            dataset_slug=dataset_slug, variant=variant, seed=seed,
            run_name=run_name, run_dir=run_dir,
            required_artifacts=required_artifacts,
            summary_metadata=summary_metadata, summary_schema=summary_schema,
        )

    def restore_completed_runs(self, *, dataset_slug: str, project: str | Path,
                               variants: Sequence[str], seeds: Sequence[int | str | None],
                               run_name_fn, required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
                               contract_file: str = "experiment_manifest.json") -> dict[str, Any]:
        """Restore only *new-style marked* archives; historical primary checkpoints
        use restore_augmentation_source, not this method.
        """
        self._require_canonical_repo()
        if self.cache_dir is None:
            raise HFResultArchiveError("A cache directory is required for archive restore")
        from utils.hf_result_archive import _normalise_required_artifacts
        required = _normalise_required_artifacts(required_artifacts)
        specs = []
        for variant in variants:
            for seed in seeds:
                run_name = segment(run_name_fn(variant, seed))
                remote = self.remote_path(dataset_slug, variant, run_name)
                specs.append((variant, seed, run_name, remote))
        if not specs:
            raise HFResultArchiveError("No cases specified for restore")
        self._assert_repository_access()
        root = Path(self._retry(lambda: self._get_snapshot_download()(
            repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
            allow_patterns=[f"{remote}/**" for *_, remote in specs],
            local_dir=str(self.cache_dir.resolve()),
            **self.snapshot_download_kwargs,
        ))).resolve()
        output = []
        for variant, seed, run_name, remote in specs:
            downloaded = root / remote
            migrated = False
            if not (downloaded / MARKER_FILENAME).is_file():
                if not downloaded.is_dir():
                    continue
                # A migrated run has no new completion marker. Never interpret
                # it as missing training evidence. Verify source SHA and *all*
                # required scientific artifacts before any reuse.
                ref_path = downloaded / "checkpoint_reference.json"
                legacy = downloaded / "legacy_upload_complete.json"
                if not ref_path.is_file() or not legacy.is_file():
                    raise HFResultArchiveError(
                        f"Existing unmarked HF run needs manual provenance audit: {remote}"
                    )
                ref = json.loads(ref_path.read_text(encoding="utf-8"))
                from utils.ftsc_hf_layout import dataset_of_namespace
                expected_dataset = dataset_of_namespace(dataset_slug)
                expected_best = remote + "/weights/best.pt"
                best = downloaded / "weights/best.pt"
                if (ref.get("dataset") != expected_dataset or ref.get("seed") != seed
                        or ref.get("canonical_best_pt") != expected_best
                        or not best.is_file() or not _HASH_RE.fullmatch(
                            str(ref.get("best_sha256", "")))
                        or self.sha256(best) != ref["best_sha256"]):
                    raise HFResultArchiveError(
                        f"Migrated HF checkpoint SHA/identity mismatch: {remote}"
                    )
                missing = [name for name in (*required, contract_file)
                           if not (downloaded / name).is_file()]
                if missing:
                    raise HFResultArchiveError(
                        f"Migrated HF checkpoint exists but current completion "
                        f"contract is incomplete: {remote}; missing {missing}. "
                        "Do not automatically retrain."
                    )
                legacy_payload = json.loads(legacy.read_text(encoding="utf-8"))
                hashes = legacy_payload.get("artifact_sha256")
                if not isinstance(hashes, dict):
                    raise HFResultArchiveError(
                        f"Migrated archive has no per-file provenance checksums: {remote}")
                for name in set((*required, contract_file, "weights/best.pt")):
                    expected_hash = hashes.get(name)
                    if (not isinstance(expected_hash, str)
                            or not _HASH_RE.fullmatch(expected_hash)
                            or self.sha256(downloaded / name) != expected_hash):
                        raise HFResultArchiveError(
                            f"Migrated archive artifact checksum mismatch: {remote}/{name}")
                migrated = True
            else:
                self.validate_archive(run_dir=downloaded, dataset_slug=dataset_slug,
                                      variant=variant, seed=seed, run_name=run_name,
                                      required_artifacts=required)
            target = Path(project).expanduser().resolve() / variant / run_name
            self._assert_local_contract_compatible(local_run=target, remote_run=downloaded,
                                                    contract_file=contract_file)
            # Matching manifest does not authorize replacing different weights.
            # This applies to both newly marked and migrated source archives.
            local_best = target / "weights/best.pt"
            remote_best = downloaded / "weights/best.pt"
            if local_best.is_file() and remote_best.is_file():
                if self.sha256(local_best) != self.sha256(remote_best):
                    raise HFResultArchiveError(
                        f"Local best.pt differs from HF; refusing overwrite: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(downloaded, target, dirs_exist_ok=True)
            if migrated:
                ref = json.loads((target / "checkpoint_reference.json").read_text(encoding="utf-8"))
                if self.sha256(target / "weights/best.pt") != ref["best_sha256"]:
                    raise HFResultArchiveError(f"Local migrated restore SHA mismatch: {target}")
            else:
                self.validate_archive(run_dir=target, dataset_slug=dataset_slug,
                                      variant=variant, seed=seed, run_name=run_name,
                                      required_artifacts=required)
            output.append(dict(case=variant, seed=seed, run_dir=str(target),
                               remote_path=remote,
                               restore_kind="MIGRATED_SHA_VERIFIED" if migrated else "MARKED_ARCHIVE_VERIFIED"))
        return {"status": "ready", "dataset": dataset_slug,
                "verified_local_run_dirs": [x["run_dir"] for x in output],
                "restored": output}

    def _source_reference(self, dataset: str, case: str, seed: int) -> tuple[dict[str, Any], str]:
        """Read metadata at one pinned revision; do not guess from legacy aliases."""
        root = augmentation_run(dataset, case, seed)
        self._require_canonical_repo()
        self._assert_repository_access()
        info = self._retry(lambda: self._get_api().repo_info(
            repo_id=self.repo_id, repo_type="dataset"))
        revision = getattr(info, "sha", None)
        if not isinstance(revision, str) or len(revision) < 30:
            raise HFResultArchiveError("Cannot pin Hub revision for checkpoint verification")
        from huggingface_hub import hf_hub_download
        ref_path = Path(hf_hub_download(
            repo_id=self.repo_id, repo_type="dataset", revision=revision,
            filename=root + "/checkpoint_reference.json", token=self.token))
        data = json.loads(ref_path.read_text(encoding="utf-8"))
        expected = augmentation_checkpoint(dataset, case, seed)
        expected_case = augmentation_run(dataset, case, seed).split("/")[-2]
        if (data.get("dataset") != dataset or data.get("seed") != seed
                or data.get("case") != expected_case
                or data.get("source_repo") != REPO_ID
                or data.get("canonical_best_pt") != expected
                or not _HASH_RE.fullmatch(str(data.get("best_sha256", "")))):
            raise HFResultArchiveError(f"Checkpoint reference identity mismatch: {root}")
        return data, revision

    def restore_augmentation_source(self, *, dataset: str, case: str, seed: int,
                                    destination: str | Path, include_predictions: bool = False) -> dict[str, Any]:
        """Restore best.pt with SHA check; refuse unrelated pre-existing local bytes.

        For TinyPerson, optionally download original-image merged predictions too.
        """
        ref, revision = self._source_reference(dataset, case, seed)
        from huggingface_hub import hf_hub_download
        remote = augmentation_checkpoint(dataset, case, seed)
        source = Path(hf_hub_download(
            repo_id=self.repo_id, repo_type="dataset",
            revision=revision, filename=remote, token=self.token))
        digest = self.sha256(source)
        if digest != ref["best_sha256"]:
            raise HFResultArchiveError("Downloaded best.pt differs from canonical checkpoint reference")
        target = Path(destination).expanduser().resolve()
        if target.is_file():
            if self.sha256(target) != digest:
                raise HFResultArchiveError(f"Local checkpoint collision; refusing overwrite: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as temp:
                staging = Path(temp.name)
            try:
                shutil.copyfile(source, staging)
                if self.sha256(staging) != digest:
                    raise HFResultArchiveError("Checkpoint copy failed SHA-256 verification")
                if target.exists():
                    raise HFResultArchiveError("Target appeared during download; refusing overwrite")
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
        predictions = None
        if include_predictions:
            if dataset != "TinyPerson":
                raise HFResultArchiveError("Merged predictions are only supported for TinyPerson")
            prediction_remote = augmentation_run(dataset, case, seed) + "/evaluation/test_merged_predictions.json"
            prediction_source = Path(hf_hub_download(
                repo_id=self.repo_id, repo_type="dataset", revision=revision,
                filename=prediction_remote, token=self.token))
            prediction_target = target.parent.parent / "evaluation" / "test_merged_predictions.json"
            if prediction_target.is_file():
                if self.sha256(prediction_target) != self.sha256(prediction_source):
                    raise HFResultArchiveError("Local merged predictions collision")
            else:
                prediction_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prediction_source, prediction_target)
            predictions = str(prediction_target)
        return dict(dataset=dataset, case=case, seed=seed, revision=revision,
                    source=remote, checkpoint=str(target), sha256=digest,
                    merged_predictions=predictions)

    def publish_evaluation(self, *, dataset: str, case: str, seed: int,
                           metric_file: str | Path, suite: str = "capstone_aug_sixcore_v1",
                           evaluator_contract: str, source_sha256: str | None = None,
                           dry_run: bool = True) -> dict[str, Any]:
        """Publish evaluation-only data into a *new* derived_evaluations suite.

        A distinct metadata marker is sent last; no source checkpoint or
        original training artifact is uploaded. Default is read-only dry-run.
        """
        source = augmentation_checkpoint(dataset, case, seed)
        self._require_canonical_repo()
        path = Path(metric_file).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise HFResultArchiveError(f"Missing completed evaluation metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise HFResultArchiveError("Evaluation metrics must be a JSON object")
        required = ("test_merged/AP50", "test_merged/mAP50-95", "test_merged/AP50-Small") if dataset == "TinyPerson" else (
            "test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small")
        for metric in (*required, "model/parameters", "model/GFLOPs"):
            value = metrics.get(metric)
            if (type(value) not in (int, float) or not math.isfinite(float(value))
                    or float(value) <= 0):
                raise HFResultArchiveError(f"Missing/invalid six-core metric: {metric}")
        if not evaluator_contract or not evaluator_contract.strip():
            raise HFResultArchiveError("Explicit evaluator contract required for publication")
        if dataset == "TinyPerson" and evaluator_contract != "tinybenchmark_merged_original_iou50_95_v2":
            raise HFResultArchiveError("TinyPerson publication requires original-image merged IoU .50:.95 metrics")
        declared_source = metrics.get("evaluation/source_checkpoint")
        declared_sha = metrics.get("evaluation/source_checkpoint_sha256")
        if declared_source is not None and declared_source != source:
            raise HFResultArchiveError("Metric file checkpoint path disagrees with canonical source")
        if declared_sha is not None and (
            not _HASH_RE.fullmatch(str(declared_sha))
            or (source_sha256 is not None and declared_sha != source_sha256)
        ):
            raise HFResultArchiveError("Metric file checkpoint digest disagrees with supplied source")
        declared_contract = metrics.get("evaluation/evaluator_contract")
        if declared_contract is not None and declared_contract != evaluator_contract:
            raise HFResultArchiveError("Metric file evaluator contract mismatch")
        destination = prefix(f"{evaluation_root(dataset, suite)}/{segment(case)}/{seed_dir(dataset, seed)}")
        digest = self.sha256(path)
        result = dict(status="DRY_RUN" if dry_run else "PUBLISHED", source_checkpoint=source,
                      remote_evaluation_path=destination, metrics_sha256=digest)
        if dry_run:
            return result
        ref, revision = self._source_reference(dataset, case, seed)
        if source_sha256 is not None and source_sha256 != ref["best_sha256"]:
            raise HFResultArchiveError("Evaluation source checkpoint SHA mismatch")
        files = set(self._retry(lambda: self._get_api().list_repo_files(
            repo_id=self.repo_id, repo_type=self.repo_type, revision=revision)))
        if any(p.startswith(destination + "/") for p in files):
            raise HFResultArchiveError(f"Evaluation destination already exists: {destination}")
        manifest = dict(schema="ftsc_evaluation_only_v1", dataset=dataset, case=case,
                        seed=seed, source_checkpoint=source, source_sha256=ref["best_sha256"],
                        source_revision=revision, evaluator_contract=evaluator_contract,
                        evaluation_sha256=digest, evaluation_file=path.name)
        with tempfile.TemporaryDirectory(prefix="ftsc_evaluation_hf_") as temp:
            manifest_path = Path(temp) / "evaluation_provenance.json"
            marker_path = Path(temp) / "evaluation_upload_complete.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            marker = dict(schema="ftsc_evaluation_completion_v1", checkpoint_sha256=ref["best_sha256"],
                          evaluation_sha256=digest, manifest_sha256=self.sha256(manifest_path))
            marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            for local_file, remote_name in (
                (path, "evaluation_metrics_extended.json"),
                (manifest_path, manifest_path.name),
                (marker_path, marker_path.name),
            ):
                self._retry(lambda file=local_file, name=remote_name: self._get_api().upload_file(
                    repo_id=self.repo_id, repo_type="dataset", token=self.token,
                    path_or_fileobj=str(file), path_in_repo=destination + "/" + name,
                    commit_message=f"FTSC evaluation only {dataset} {case} seed {seed}"))
        return result
