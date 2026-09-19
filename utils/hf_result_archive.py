"""Dataset-agnostic persistence for completed FTSC experiment runs.

This module archives *completed* run directories to a Hugging Face dataset
repository and restores only archives whose completion marker validates.  It
is shared infrastructure for TinyPerson, Varroa, LEVIR-Ship, and future
datasets; acquiring or preparing a dataset is deliberately out of scope.

The protocol is transactional in the useful sense: the complete run directory
is uploaded first (excluding ``upload_complete.json``), then the marker is
written and uploaded in a final request.  A remote directory without a valid
marker is never considered resumable.

Typical use::

    archive = HFResultArchive(
        repo_id="TLMHoang/FTSC_results", token=os.environ["HF_TOKEN"]
    )
    archive.restore_completed_runs(
        dataset_slug="varroa", project=project_dir, variants=variants,
        seeds=[42, 43, 44], run_name_fn=lambda _variant, seed: f"seed_{seed}",
    )

No token is embedded here and no repository is created by this utility.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "causal_suite_complete.json",
    "results.csv",
    "args.yaml",
    "config.yaml",
)
"""Current FTSC seed-run completeness contract, excluding generated summary."""

MARKER_FILENAME = "upload_complete.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
ARCHIVE_SCHEMA = "ftsc_seed_archive_v1"
SUMMARY_SCHEMA = "ftsc_seed_summary_v1"


class HFResultArchiveError(RuntimeError):
    """A completed-result archive could not be safely published or restored."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_segment(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFResultArchiveError(f"{field} must be a non-empty string")
    value = value.strip()
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ".." in path.parts
    ):
        raise HFResultArchiveError(
            f"{field} must be one safe remote path segment, got {value!r}"
        )
    return value


def _validate_relative_artifact(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise HFResultArchiveError("required artifact paths must be non-empty strings")
    path = path.strip()
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or path.startswith("/") or "\\" in path:
        raise HFResultArchiveError(f"artifact path must be relative and traversal-free: {path!r}")
    if path in {".", ""}:
        raise HFResultArchiveError(f"artifact path must name a file: {path!r}")
    return parsed.as_posix()


def _normalise_required_artifacts(required_artifacts: Iterable[str]) -> tuple[str, ...]:
    required = tuple(_validate_relative_artifact(item) for item in required_artifacts)
    if not required:
        raise HFResultArchiveError("at least one required artifact is required")
    if len(set(required)) != len(required):
        raise HFResultArchiveError("required artifact paths must not contain duplicates")
    if MARKER_FILENAME in required:
        raise HFResultArchiveError(f"{MARKER_FILENAME} cannot be a required artifact")
    return required


class HFResultArchive:
    """Publish and restore verified, dataset-agnostic completed run archives.

    ``api`` and ``snapshot_download_fn`` are dependency-injection hooks for
    offline tests.  In normal use they are omitted and Hugging Face Hub is
    imported lazily only when a network operation is requested.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        repo_type: str = "dataset",
        cache_dir: str | Path | None = None,
        api: Any | None = None,
        snapshot_download_fn: Callable[..., str | Path] | None = None,
        retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise HFResultArchiveError("repo_id must be a non-empty string")
        if not isinstance(repo_type, str) or not repo_type.strip():
            raise HFResultArchiveError("repo_type must be a non-empty string")
        if retries < 1:
            raise HFResultArchiveError("retries must be at least one")
        if retry_backoff_seconds < 0:
            raise HFResultArchiveError("retry_backoff_seconds must be non-negative")

        resolved_token = token if token is not None else os.environ.get("HF_TOKEN")
        if not isinstance(resolved_token, str) or not resolved_token.strip():
            raise HFResultArchiveError(
                "A Hugging Face token is required; pass token= or set HF_TOKEN"
            )

        self.repo_id = repo_id.strip()
        self.repo_type = repo_type.strip()
        self.token = resolved_token.strip()
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._api = api
        self._snapshot_download_fn = snapshot_download_fn
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep

    @staticmethod
    def sha256(path: str | Path) -> str:
        """Return the SHA-256 digest of one file."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def remote_path(dataset_slug: str, variant: str, run_name: str) -> str:
        """Build the canonical ``dataset/variant/run-name`` remote directory."""
        return "/".join(
            (
                _validate_segment(dataset_slug, "dataset_slug"),
                _validate_segment(variant, "variant"),
                _validate_segment(run_name, "run_name"),
            )
        )

    def _get_api(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise HFResultArchiveError(
                    "huggingface_hub is required to publish or restore result archives"
                ) from exc
            self._api = HfApi(token=self.token)
        return self._api

    def _get_snapshot_download(self) -> Callable[..., str | Path]:
        if self._snapshot_download_fn is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise HFResultArchiveError(
                    "huggingface_hub is required to restore result archives"
                ) from exc
            self._snapshot_download_fn = snapshot_download
        return self._snapshot_download_fn

    def _retry(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.retries):
            try:
                return operation()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                self._sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")

    def _assert_repository_access(self) -> None:
        self._retry(
            lambda: self._get_api().repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token,
            )
        )

    def required_artifact_metadata(
        self,
        run_dir: str | Path,
        required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
    ) -> dict[str, dict[str, int | str]]:
        """Verify required files and return their digest and byte metadata."""
        run_path = Path(run_dir).expanduser().resolve()
        required = _normalise_required_artifacts(required_artifacts)
        if not run_path.is_dir():
            raise HFResultArchiveError(f"run directory does not exist: {run_path}")

        artifacts: dict[str, dict[str, int | str]] = {}
        for relative in required:
            path = run_path / relative
            if not path.is_file():
                raise HFResultArchiveError(
                    f"Cannot archive incomplete run; missing required artifact {path}"
                )
            artifacts[relative] = {"sha256": self.sha256(path), "bytes": path.stat().st_size}
        return artifacts

    def write_run_summary(
        self,
        *,
        run_dir: str | Path,
        dataset_slug: str,
        variant: str,
        seed: int | str | None,
        summary_metadata: Mapping[str, Any] | None = None,
        summary_schema: str = SUMMARY_SCHEMA,
    ) -> Path:
        """Write generic per-run metadata and return ``run_summary.json``."""
        dataset_slug = _validate_segment(dataset_slug, "dataset_slug")
        variant = _validate_segment(variant, "variant")
        if not isinstance(summary_schema, str) or not summary_schema.strip():
            raise HFResultArchiveError("summary_schema must be a non-empty string")
        run_path = Path(run_dir).expanduser().resolve()
        if not run_path.is_dir():
            raise HFResultArchiveError(f"run directory does not exist: {run_path}")

        payload = dict(summary_metadata or {})
        payload.update(
            {
                "schema": summary_schema,
                "dataset": dataset_slug,
                "case": variant,
                "seed": seed,
                "created_at_utc": _utc_now(),
            }
        )
        summary_path = run_path / RUN_SUMMARY_FILENAME
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return summary_path

    def publish_run(
        self,
        *,
        dataset_slug: str,
        variant: str,
        seed: int | str | None,
        run_name: str,
        run_dir: str | Path,
        required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
        summary_metadata: Mapping[str, Any] | None = None,
        summary_schema: str = SUMMARY_SCHEMA,
    ) -> dict[str, Any]:
        """Archive one complete run, uploading its completion marker last."""
        dataset_slug = _validate_segment(dataset_slug, "dataset_slug")
        variant = _validate_segment(variant, "variant")
        run_name = _validate_segment(run_name, "run_name")
        run_path = Path(run_dir).expanduser().resolve()
        if run_path.name != run_name:
            raise HFResultArchiveError(
                f"run_dir name {run_path.name!r} does not match supplied run_name {run_name!r}"
            )

        artifacts = self.required_artifact_metadata(run_path, required_artifacts)
        summary_path = self.write_run_summary(
            run_dir=run_path,
            dataset_slug=dataset_slug,
            variant=variant,
            seed=seed,
            summary_metadata=summary_metadata,
            summary_schema=summary_schema,
        )
        artifacts[RUN_SUMMARY_FILENAME] = {
            "sha256": self.sha256(summary_path),
            "bytes": summary_path.stat().st_size,
        }
        remote_path = self.remote_path(dataset_slug, variant, run_name)
        self._assert_repository_access()

        # The marker is deliberately absent from this folder request.  If it
        # already exists from an interrupted earlier attempt it must not make
        # the remote archive look complete before this upload has succeeded.
        upload_commit = self._retry(
            lambda: self._get_api().upload_folder(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                folder_path=str(run_path),
                path_in_repo=remote_path,
                token=self.token,
                ignore_patterns=[MARKER_FILENAME],
                commit_message=f"archive {dataset_slug} {variant} {run_name}",
            )
        )

        marker_path = run_path / MARKER_FILENAME
        marker = {
            "schema": ARCHIVE_SCHEMA,
            "dataset": dataset_slug,
            "case": variant,
            "seed": seed,
            "run_relative_path": f"{variant}/{run_name}",
            "uploaded_at_utc": _utc_now(),
            "artifact_sha256": {
                relative: metadata["sha256"] for relative, metadata in artifacts.items()
            },
        }
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        marker_commit = self._retry(
            lambda: self._get_api().upload_file(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                path_or_fileobj=str(marker_path),
                path_in_repo=f"{remote_path}/{MARKER_FILENAME}",
                token=self.token,
                commit_message=f"mark complete {dataset_slug} {variant} {run_name}",
            )
        )
        return {
            "status": "uploaded",
            "dataset": dataset_slug,
            "case": variant,
            "seed": seed,
            "run_dir": str(run_path),
            "remote_path": remote_path,
            "artifact_count": len(artifacts),
            "artifact_bytes": sum(int(item["bytes"]) for item in artifacts.values()),
            "upload_commit": getattr(upload_commit, "oid", None),
            "marker_commit": getattr(marker_commit, "oid", None),
        }

    def validate_archive(
        self,
        *,
        run_dir: str | Path,
        dataset_slug: str,
        variant: str,
        run_name: str,
        seed: int | str | None,
        required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
    ) -> dict[str, Any]:
        """Validate a marked local or downloaded archive without trusting it."""
        dataset_slug = _validate_segment(dataset_slug, "dataset_slug")
        variant = _validate_segment(variant, "variant")
        run_name = _validate_segment(run_name, "run_name")
        required = _normalise_required_artifacts(required_artifacts)
        run_path = Path(run_dir).expanduser().resolve()
        marker_path = run_path / MARKER_FILENAME
        if not marker_path.is_file():
            raise HFResultArchiveError(f"Archive has no completion marker: {run_path}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HFResultArchiveError(f"Invalid completion marker: {marker_path}") from exc
        if not isinstance(marker, dict):
            raise HFResultArchiveError(f"Completion marker must be a JSON object: {marker_path}")
        if marker.get("schema") != ARCHIVE_SCHEMA:
            raise HFResultArchiveError(f"Unsupported archive marker in {run_path}")
        if marker.get("dataset") != dataset_slug:
            raise HFResultArchiveError(f"Archive dataset mismatch in {run_path}")
        if marker.get("case") != variant:
            raise HFResultArchiveError(f"Archive case mismatch in {run_path}")
        if seed is not None and marker.get("seed") != seed:
            raise HFResultArchiveError(f"Archive seed mismatch in {run_path}")
        if marker.get("run_relative_path") != f"{variant}/{run_name}":
            raise HFResultArchiveError(f"Archive run-path identity mismatch in {run_path}")

        digests = marker.get("artifact_sha256")
        if not isinstance(digests, dict) or not digests:
            raise HFResultArchiveError(f"Archive has no artifact checksums: {run_path}")
        expected_artifacts = (*required, RUN_SUMMARY_FILENAME)
        for relative in expected_artifacts:
            if relative not in digests:
                raise HFResultArchiveError(
                    f"Archive marker omits required artifact {relative}: {run_path}"
                )
        for relative, expected_hash in digests.items():
            relative = _validate_relative_artifact(relative)
            if not isinstance(expected_hash, str) or not expected_hash:
                raise HFResultArchiveError(f"Archive has invalid checksum for {relative}: {run_path}")
            artifact = run_path / relative
            if not artifact.is_file():
                raise HFResultArchiveError(f"Archive is missing {relative}: {run_path}")
            actual_hash = self.sha256(artifact)
            if actual_hash != expected_hash:
                raise HFResultArchiveError(
                    f"Archive checksum mismatch for {relative}: {actual_hash} != {expected_hash}"
                )
        return marker

    def restore_completed_runs(
        self,
        *,
        dataset_slug: str,
        project: str | Path,
        variants: Sequence[str],
        seeds: Sequence[int | str | None],
        run_name_fn: Callable[[str, int | str | None], str],
        required_artifacts: Iterable[str] = DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
        contract_file: str = "experiment_manifest.json",
    ) -> dict[str, Any]:
        """Restore expected marked archives from one dataset namespace only."""
        dataset_slug = _validate_segment(dataset_slug, "dataset_slug")
        if self.cache_dir is None:
            raise HFResultArchiveError("cache_dir is required to restore completed runs")
        if not callable(run_name_fn):
            raise HFResultArchiveError("run_name_fn must be callable")
        required = _normalise_required_artifacts(required_artifacts)
        contract_file = _validate_relative_artifact(contract_file)
        project_path = Path(project).expanduser().resolve()
        checked_variants = tuple(_validate_segment(item, "variant") for item in variants)
        if not checked_variants:
            raise HFResultArchiveError("at least one variant is required for restore")

        self._assert_repository_access()
        cache_dir = self.cache_dir.resolve()
        snapshot_root = Path(
            self._retry(
                lambda: self._get_snapshot_download()(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    token=self.token,
                    allow_patterns=[f"{dataset_slug}/**"],
                    local_dir=str(cache_dir),
                )
            )
        ).resolve()
        remote_dataset = snapshot_root / dataset_slug
        restored: list[dict[str, Any]] = []

        if remote_dataset.is_dir():
            for variant in checked_variants:
                for seed in seeds:
                    run_name = _validate_segment(run_name_fn(variant, seed), "run_name")
                    remote_run = remote_dataset / variant / run_name
                    if not (remote_run / MARKER_FILENAME).is_file():
                        continue
                    self.validate_archive(
                        run_dir=remote_run,
                        dataset_slug=dataset_slug,
                        variant=variant,
                        run_name=run_name,
                        seed=seed,
                        required_artifacts=required,
                    )
                    local_run = project_path / variant / run_name
                    self._assert_local_contract_compatible(
                        local_run=local_run,
                        remote_run=remote_run,
                        contract_file=contract_file,
                    )
                    local_run.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(remote_run, local_run, dirs_exist_ok=True)
                    self.validate_archive(
                        run_dir=local_run,
                        dataset_slug=dataset_slug,
                        variant=variant,
                        run_name=run_name,
                        seed=seed,
                        required_artifacts=required,
                    )
                    restored.append(
                        {
                            "case": variant,
                            "seed": seed,
                            "run_dir": str(local_run.resolve()),
                            "remote_path": self.remote_path(dataset_slug, variant, run_name),
                        }
                    )
        return {
            "status": "ready",
            "dataset": dataset_slug,
            "verified_local_run_dirs": [entry["run_dir"] for entry in restored],
            "restored": restored,
        }

    def _assert_local_contract_compatible(
        self, *, local_run: Path, remote_run: Path, contract_file: str
    ) -> None:
        if not local_run.exists() or not any(local_run.iterdir()):
            return
        local_contract = local_run / contract_file
        remote_contract = remote_run / contract_file
        if not local_contract.is_file() or not remote_contract.is_file():
            raise HFResultArchiveError(
                "Refusing to overwrite a non-empty local run without matching "
                f"contract files: {local_run}"
            )
        if self.sha256(local_contract) != self.sha256(remote_contract):
            raise HFResultArchiveError(
                "Refusing to overwrite a local run with a different experiment "
                f"contract: {local_run}"
            )

    @staticmethod
    def restored_run_dirs(restore_result: Mapping[str, Any]) -> set[str]:
        """Return verified restored directories for duplicate-upload skipping."""
        values = restore_result.get("verified_local_run_dirs", [])
        if not isinstance(values, list):
            raise HFResultArchiveError("restore result has invalid verified_local_run_dirs")
        return {str(Path(value).resolve()) for value in values}
