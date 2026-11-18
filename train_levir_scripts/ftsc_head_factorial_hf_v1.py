"""HF-first archive transport for the LEVIR head-configuration x FTSC factorial.

Strictly locks to the existing FTSC canonical dataset repo and a new journal
namespace. Never infers absence from network errors. Never silently replaces
local checkpoints, partial remote archives or incompatible contracts. Credentials
come exclusively from the live runtime environment (not this source).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.ftsc_results_uploader import FTSCResultsUploader, HF_REPO_ID, HF_REPO_TYPE
from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
from utils.hf_result_archive import MARKER_FILENAME
from train_levir_scripts import train_levir_ftsc_head_factorial_v1 as factorial

NAMESPACE = "levir_ftsc_head_factorial_v1"
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "config.yaml",
    "experiment_manifest.json",
    "factorial_contract.json",
    "factorial_completion.json",
    "evaluation_metrics_extended.json",
)


def remote_prefix(case: str, seed: int) -> str:
    factorial.require(case in factorial.CASES and seed in factorial.SEEDS, "Invalid factorial cell")
    prefix = CanonicalFTSCArchive.remote_path(NAMESPACE, case, f"seed_{seed}")
    factorial.require(prefix.startswith("LEVIR-Ship/journal_experiments/head_factorial_v1/"),
                      "Canonical HF namespace routing changed")
    return prefix


def inventory(token: str, *, api: Any | None = None) -> dict[str, Any]:
    """Read a pinned HF revision; an API error must abort, never imply missing data."""
    factorial.require(isinstance(token, str) and bool(token.strip()), "HF_TOKEN must be set")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    info = api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    revision = info.sha
    factorial.require(isinstance(revision, str) and len(revision) >= 30,
                      "HF revision was not pinned")
    paths = set(api.list_repo_files(repo_id=HF_REPO_ID,
                                    repo_type=HF_REPO_TYPE, revision=revision))
    rows = []
    for case in factorial.CASES:
        for seed in factorial.SEEDS:
            prefix = remote_prefix(case, seed)
            archived = {f[len(prefix)+1:] for f in paths if f.startswith(prefix + "/")}
            if not archived:
                state = "ABSENT"
            else:
                missing = sorted(set(REQUIRED + (MARKER_FILENAME,)) - archived)
                factorial.require(not missing,
                                  f"Partial or incomplete HF archive at {prefix}; "
                                  f"missing: {missing}. Manual review required.")
                state = "MARKED_PRESENT"
            rows.append({"case": case, "seed": seed, "remote_prefix": prefix, "state": state})
    return {"repo": HF_REPO_ID, "revision": revision, "rows": rows}


def make_uploader(token: str, cache_dir: Path) -> FTSCResultsUploader:
    return FTSCResultsUploader(token=token, cache_dir=cache_dir)


def restore_available(uploader: FTSCResultsUploader, inventory_report: dict[str, Any],
                      args: Any, data_yaml: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Restore each complete remote arm and check its local scientific contract."""
    restored = []
    for row in inventory_report["rows"]:
        if row["state"] != "MARKED_PRESENT":
            continue
        case, seed = row["case"], row["seed"]
        result = uploader.restore_completed_runs(
            namespace=NAMESPACE, project=args.project, variants=[case], seeds=[seed],
            required_artifacts=REQUIRED,
        )
        factor_run = args.project / case / f"seed_{seed}"
        expected = factorial.contract(args, case, seed, data_yaml, source)
        factorial.require(factorial.current_state(factor_run, expected) == "COMPLETE",
                          f"HF restore did not yield complete matched run {case}/{seed}")
        factorial.require(result["restored"], f"HF marked run was not restored: {case}/{seed}")
        restored.append({"case": case, "seed": seed, "state": "SHA_VERIFIED_RESTORED"})
    return restored


def publish_or_verify(uploader: FTSCResultsUploader, token: str, args: Any,
                      case: str, seed: int, data_yaml: Path,
                      source: dict[str, Any]) -> dict[str, Any]:
    """Archive immediately after each finished run; reject all remote collisions."""
    run = args.project / case / f"seed_{seed}"
    expected = factorial.contract(args, case, seed, data_yaml, source)
    factorial.require(factorial.current_state(run, expected) == "COMPLETE",
                      f"Refusing to upload an incomplete or mismatched run: {run}")
    for rel in REQUIRED:
        factorial.require((run / rel).is_file(), f"Required HF artifact missing: {run / rel}")
    # Refresh inventory after training: another session may have published first.
    report = inventory(token)
    row = next(x for x in report["rows"] if x["case"] == case and x["seed"] == seed)
    if row["state"] == "MARKED_PRESENT":
        # Canonical restore refuses to replace local files with different hashes.
        restore_available(uploader, {"rows": [row]}, args, data_yaml, source)
        return {"case": case, "seed": seed, "status": "ALREADY_PUBLISHED_VERIFIED"}
    result = uploader.publish_run(
        namespace=NAMESPACE, case=case, seed=seed, run_dir=run,
        required_artifacts=REQUIRED,
        summary_metadata={
            "dataset": "LEVIR-Ship", "suite": NAMESPACE,
            "head_configuration": factorial.FACTORS[case][0],
            "ftsc_enabled": factorial.FACTORS[case][1],
            "best_sha256": factorial.sha(run / "weights/best.pt"),
            "scientific_contract_sha256": factorial.sha(run / "factorial_contract.json"),
        },
    )
    # The canonical uploader writes completion marker last and verifies publication;
    # do not mark the local run uploaded unless the call returned successfully.
    confirmed = inventory(token)
    confirmed_row = next(x for x in confirmed["rows"]
                         if x["case"] == case and x["seed"] == seed)
    factorial.require(confirmed_row["state"] == "MARKED_PRESENT",
                      f"HF completion marker not visible after upload: {case}/{seed}")
    record = {"case": case, "seed": seed, "status": "UPLOADED_AND_REMOTE_MARKER_CONFIRMED",
              "remote": remote_prefix(case, seed),
              "verified_revision": confirmed["revision"]}
    (run / "factorial_hf_publication.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record
