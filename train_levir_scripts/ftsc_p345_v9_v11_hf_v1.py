"""Fail-closed HF-first archive for ONLY LEVIR P345+FTSC YOLOv9t and YOLO11n.

Uses the existing canonical FTSC dataset repository and completion-marker-last
transport, without touching historical or P23 archives. No credentials in code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.ftsc_results_uploader import FTSCResultsUploader, HF_REPO_ID, HF_REPO_TYPE
from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
from utils.hf_result_archive import MARKER_FILENAME
from train_levir_scripts import train_levir_ftsc_p345_v9_v11_v1 as suite

NAMESPACE = suite.NAMESPACE
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "config.yaml",
    "experiment_manifest.json",
    "p345_ftsc_contract.json",
    "p345_ftsc_completion.json",
    "evaluation_metrics_extended.json",
)
REMOTE_ROOT = "LEVIR-Ship/journal_experiments/p345_ftsc_v9_v11_v1/"


def remote_prefix(case: str, seed: int) -> str:
    suite.require(case in suite.CASES and seed in suite.SEEDS,
                  "Only the two authorized P345+FTSC cases are supported")
    path = CanonicalFTSCArchive.remote_path(NAMESPACE, case, f"seed_{seed}")
    suite.require(path.startswith(REMOTE_ROOT), f"HF archive namespace drift: {path}")
    return path


def inventory(token: str, *, api: Any | None = None) -> dict[str, Any]:
    suite.require(isinstance(token, str) and bool(token.strip()), "HF_TOKEN is required")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    info = api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    revision = info.sha
    suite.require(isinstance(revision, str) and len(revision) >= 30,
                  "HF results revision was not verified")
    paths = set(api.list_repo_files(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, revision=revision))
    rows = []
    for case in suite.CASES:
        for seed in suite.SEEDS:
            remote = remote_prefix(case, seed)
            contents = {p[len(remote) + 1:] for p in paths if p.startswith(remote + "/")}
            if not contents:
                status = "ABSENT"
            else:
                missing = sorted(set((*REQUIRED, MARKER_FILENAME)) - contents)
                suite.require(not missing,
                              f"Partial remote run {remote} is not safe to reuse/retrain. Missing {missing}")
                status = "MARKED_PRESENT"
            rows.append({"case": case, "seed": seed, "remote": remote, "status": status})
    return {"repo_id": HF_REPO_ID, "revision": revision, "rows": rows}


def uploader(token: str, cache: Path) -> FTSCResultsUploader:
    return FTSCResultsUploader(token=token, cache_dir=cache)


def restore(upl: FTSCResultsUploader, report: dict[str, Any],
            args: Any, data_yaml: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    done = []
    for row in report["rows"]:
        if row["status"] != "MARKED_PRESENT":
            continue
        case, seed = row["case"], row["seed"]
        result = upl.restore_completed_runs(
            namespace=NAMESPACE, project=args.project, variants=[case], seeds=[seed],
            required_artifacts=REQUIRED,
        )
        run = args.project / case / f"seed_{seed}"
        expected = suite.contract(args, case, seed, data_yaml, source)
        suite.require(result["restored"] and suite.state(run, expected) == "COMPLETE",
                      f"HF restored run not scientifically complete: {case}/seed_{seed}")
        done.append({"case": case, "seed": seed, "status": "SHA_VERIFIED_RESTORED"})
    return done


def publish_or_verify(upl: FTSCResultsUploader, token: str, args: Any,
                      case: str, seed: int, data_yaml: Path,
                      source: dict[str, Any]) -> dict[str, Any]:
    """After each completed seed: publish exactly once or verify matching remote."""
    run = args.project / case / f"seed_{seed}"
    expected = suite.contract(args, case, seed, data_yaml, source)
    suite.require(suite.state(run, expected) == "COMPLETE",
                  f"Refusing upload before complete valid test evaluation: {run}")
    for relative in REQUIRED:
        suite.require((run / relative).is_file(),
                      f"HF required artifact missing: {run / relative}")
    current = inventory(token)  # network failure must NEVER be construed as no checkpoint
    cell = next(r for r in current["rows"] if r["case"] == case and r["seed"] == seed)
    if cell["status"] == "MARKED_PRESENT":
        restore(upl, {"rows": [cell]}, args, data_yaml, source)
        return {"case": case, "seed": seed, "status": "EXISTING_HF_ARCHIVE_SHA_VERIFIED"}
    result = upl.publish_run(
        namespace=NAMESPACE, case=case, seed=seed, run_dir=run,
        required_artifacts=REQUIRED,
        summary_metadata={
            "dataset": "LEVIR-Ship", "suite": NAMESPACE,
            "family": suite.CASES[case]["family"],
            "detection_heads": "P3/P4/P5", "ftsc_enabled": True,
            "best_sha256": suite.sha(run / "weights/best.pt"),
            "contract_sha256": suite.sha(run / "experiment_manifest.json"),
        },
    )
    verified = inventory(token)
    row = next(r for r in verified["rows"] if r["case"] == case and r["seed"] == seed)
    suite.require(row["status"] == "MARKED_PRESENT",
                  f"Completion marker absent after HF upload: {case}/seed_{seed}")
    receipt = {
        "case": case, "seed": seed,
        "status": "UPLOADED_AND_MARKER_VISIBLE",
        "remote_prefix": remote_prefix(case, seed),
        "verified_revision": verified["revision"],
    }
    (run / "p345_ftsc_hf_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
