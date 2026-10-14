"""Read-only multi-namespace HF checkpoint discovery and opt-in verified recovery.

A best.pt is a *candidate*, never proof that the scientific experiment matches.
Unlike HFResultArchive.restore_completed_runs, this utility can inspect runs
without upload_complete.json; it must not relax completed-archive semantics.
The caller verifies YAML, train args, split and lineage *before* restore().
No training, evaluation or upload is performed here.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

_METADATA = ("config.yaml", "args.yaml", "experiment_manifest.json",
             "run_summary.json", "upload_complete.json")
_SEED_RE = re.compile(r"^seed[_-]?0*(\d+)(?:[_-].*)?$", re.IGNORECASE)


class CheckpointRecoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointCandidate:
    repo_id: str
    revision: str
    prefix: str
    case: str
    seed: int
    run_path: str
    checkpoint_path: str
    metadata_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _safe_path(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise CheckpointRecoveryError("Unsafe or missing HF path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(p in ("", ".", "..") for p in parsed.parts):
        raise CheckpointRecoveryError(f"Unsafe HF path: {path!r}")
    return str(parsed)


def _digest(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HFCheckpointRecovery:
    """Index full HF paths; download only metadata until verification succeeds.

    Inject api/download_fn in unit tests. In production supply HF_TOKEN and
    use a revision-pinned scan so discovery and later downloads match.
    """

    def __init__(self, repo_id: str, *, token: str | None = None,
                 api: Any | None = None, download_fn: Callable[..., str] | None = None):
        if not repo_id or "/" not in repo_id:
            raise CheckpointRecoveryError("Provide an explicit HF repo ID")
        self.repo_id = repo_id
        self.token = token
        self._api = api
        self._download_fn = download_fn

    def _get_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def _get_download(self) -> Callable[..., str]:
        if self._download_fn is None:
            from huggingface_hub import hf_hub_download
            self._download_fn = hf_hub_download
        return self._download_fn

    def inventory(self, prefixes: Sequence[str]) -> dict[str, Any]:
        """List once, fail CLOSED on HF failure; never confuse unavailable and empty."""
        allowed = tuple(_safe_path(p).rstrip("/") for p in prefixes)
        if not allowed:
            raise CheckpointRecoveryError("At least one dataset namespace is required")
        api = self._get_api()
        info = api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        revision = getattr(info, "sha", None)
        if not revision:
            raise CheckpointRecoveryError("HF revision unavailable; cannot pin checkpoint identity")
        files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", revision=revision)
        file_set = sorted({_safe_path(p) for p in files
                           if any(p.startswith(prefix + "/") for prefix in allowed)})
        return {"repo_id": self.repo_id, "revision": revision, "prefixes": list(allowed),
                "files": file_set}

    def candidates(self, inventory: Mapping[str, Any], *,
                   case: str, seed: int, aliases: Sequence[str],
                   prefixes: Sequence[str] | None = None,
                   match_case: bool = True) -> list[CheckpointCandidate]:
        """Match exact path components, NOT partial text or similar experiment names."""
        if inventory.get("repo_id") != self.repo_id or not inventory.get("revision"):
            raise CheckpointRecoveryError("Untrusted or incomplete HF inventory")
        names = set(aliases)
        if not names or any("/" in n or not n for n in names):
            raise CheckpointRecoveryError("Aliases must be exact single path components")
        allowed = tuple(_safe_path(p) for p in (prefixes or inventory["prefixes"]))
        file_set = set(inventory["files"])
        results: list[CheckpointCandidate] = []
        for filename in sorted(file_set):
            if not filename.endswith("/weights/best.pt"):
                continue
            if not any(filename.startswith(prefix + "/") for prefix in allowed):
                continue
            run = filename[: -len("/weights/best.pt")]
            parts = PurePosixPath(run).parts
            if match_case and not (names & set(parts)):
                continue
            seed_positions = [i for i, p in enumerate(parts) if
                              (m := _SEED_RE.fullmatch(p)) and int(m.group(1)) == seed]
            if not seed_positions or (match_case and not any(
                idx < pos for idx, part in enumerate(parts) if part in names
                for pos in seed_positions
            )):
                continue
            prefix = max((p for p in allowed if filename.startswith(p + "/")),
                         key=len)
            meta = tuple(f"{run}/{m}" for m in _METADATA if f"{run}/{m}" in file_set)
            results.append(CheckpointCandidate(self.repo_id, inventory["revision"],
                                               prefix, case, seed, run, filename, meta))
        return results

    def download_metadata(self, candidate: CheckpointCandidate,
                          cache_dir: Path) -> dict[str, Path]:
        """Small files only. Missing config/args requires caller to BLOCK, not TRAIN."""
        output: dict[str, Path] = {}
        for remote in candidate.metadata_paths:
            local = Path(self._get_download()(
                repo_id=self.repo_id, repo_type="dataset",
                filename=remote, revision=candidate.revision, token=self.token,
                local_dir=str(cache_dir),
            ))
            if not local.is_file():
                raise CheckpointRecoveryError(f"Missing downloaded metadata: {remote}")
            output[PurePosixPath(remote).name] = local
        return output

    def restore(self, candidate: CheckpointCandidate, *, destination: Path,
                verified: bool, cache_dir: Path,
                expected_sha256: str | None = None) -> dict[str, Any]:
        """Never download weights or write destination without caller verification."""
        if not verified:
            raise CheckpointRecoveryError("Refusing unverified checkpoint recovery")
        source = Path(self._get_download()(
            repo_id=self.repo_id, repo_type="dataset",
            filename=candidate.checkpoint_path, revision=candidate.revision,
            token=self.token, local_dir=str(cache_dir),
        ))
        if not source.is_file():
            raise CheckpointRecoveryError("HF checkpoint download did not create a file")
        digest = _digest(source)
        if expected_sha256 and digest != expected_sha256:
            raise CheckpointRecoveryError("HF checkpoint SHA-256 differs from archive marker")
        destination = Path(destination)
        if destination.is_file():
            if _digest(destination) != digest:
                raise CheckpointRecoveryError("Local checkpoint collision; refusing overwrite")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + ".recovery_tmp")
            if temp.exists():
                raise CheckpointRecoveryError("Stale recovery temp file; refusing overwrite")
            try:
                shutil.copy2(source, temp)
                if _digest(temp) != digest:
                    raise CheckpointRecoveryError("Recovered checkpoint failed checksum")
                temp.replace(destination)
            finally:
                if temp.exists():
                    temp.unlink()
        return {"repo_id": self.repo_id, "revision": candidate.revision,
                "hf_checkpoint": candidate.checkpoint_path,
                "checkpoint_sha256": digest, "local_checkpoint": str(destination)}

    @staticmethod
    def marker_checksum(metadata: Mapping[str, Path]) -> str | None:
        marker_path = metadata.get("upload_complete.json")
        if marker_path is None:
            return None
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        sha = marker.get("artifact_sha256", {}).get("weights/best.pt")
        if sha is not None and (not isinstance(sha, str) or
                                re.fullmatch(r"[0-9a-f]{64}", sha) is None):
            raise CheckpointRecoveryError("Invalid best.pt hash in HF marker")
        return sha
