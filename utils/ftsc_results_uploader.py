"""The sole FTSC live-result repository contract.

Tokens are supplied by the notebook only.  Archive integrity and the
completion-marker-last transaction are delegated to :mod:`hf_result_archive`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.hf_result_archive import (
    DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
    HFResultArchive,
)

HF_REPO_ID = "TLMHoang/FTSC_results"
HF_REPO_TYPE = "dataset"


def _has_authentication_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("token" in str(key).lower() or _has_authentication_field(child) for key, child in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_has_authentication_field(child) for child in value)
    return False


class FTSCResultsUploader:
    """Repository-locked wrapper around ``HFResultArchive``."""

    def __init__(self, *, token: str, cache_dir: str | Path | None = None, **archive_kwargs: Any) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("A non-empty notebook-supplied Hugging Face token is required.")
        if {"repo_id", "repo_type", "token"}.intersection(archive_kwargs):
            raise TypeError("Repository identity and authentication are fixed by FTSCResultsUploader.")
        self._archive = HFResultArchive(
            HF_REPO_ID, token=token, repo_type=HF_REPO_TYPE, cache_dir=cache_dir, **archive_kwargs
        )

    @property
    def repo_id(self) -> str: return HF_REPO_ID

    def restore_completed_runs(self, *, namespace: str, project: str | Path, variants: Iterable[str], seeds: Iterable[int], run_name_fn=None, required_artifacts: Iterable[str] | None = None) -> dict[str, Any]:
        kwargs = {"required_artifacts": required_artifacts} if required_artifacts is not None else {}
        return self._archive.restore_completed_runs(
            dataset_slug=namespace, project=project, variants=variants, seeds=seeds,
            run_name_fn=run_name_fn or (lambda _case, seed: f"seed_{seed}"),
            **kwargs,
        )

    def publish_run(self, *, namespace: str, case: str, seed: int, run_dir: str | Path, required_artifacts: Iterable[str], summary_metadata: Mapping[str, Any] | None = None, run_name: str | None = None) -> dict[str, Any]:
        run_path = Path(run_dir)
        resolved_run_name = run_name or run_path.name
        if resolved_run_name != run_path.name:
            raise ValueError("run_name must match the supplied run directory basename.")
        metadata = dict(summary_metadata or {})
        if _has_authentication_field(metadata):
            raise ValueError("Summary metadata cannot contain authentication fields.")
        return self._archive.publish_run(
            dataset_slug=namespace, variant=case, seed=seed, run_name=resolved_run_name, run_dir=run_path,
            required_artifacts=required_artifacts, summary_metadata=metadata,
        )

    def required_artifact_metadata(self, run_dir: str | Path, required_artifacts: Iterable[str]) -> dict[str, Any]:
        return self._archive.required_artifact_metadata(run_dir, required_artifacts)
