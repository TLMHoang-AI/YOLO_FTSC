#!/usr/bin/env python3
"""Run a complete, idempotent, HF-backed cross-YOLO suite from one live notebook cell.

This is a live orchestrator, NOT an alternative training/evaluation implementation:
dataset preparation, training, checkpoint state and six-core evaluation remain
owned by their respective frozen dataset runners.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "models_related/ultralytics"
for entry in (ROOT, ULTRA):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from utils import cross_yolo_portability as common

SUITES: dict[str, dict[str, Any]] = {
    "tinyperson": {
        "runner": "train_tinyperson_scripts/train_tinyperson_ftsc_cross_yolo_portability_v1.py",
        "suffix": "_corner_sw640_sh512",
        "workers": 4,
        "yaml_subdir": "tinyperson",
    },
    "varroa": {
        "runner": "train_levir_scripts/train_varroa_ftsc_cross_yolo_portability_v1.py",
        "suffix": "",
        "workers": 8,
        "yaml_subdir": "varroa",
    },
}


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUITES)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--hf-repo-id", default="TLMHoang/FTSC_results")
    parser.add_argument("--families", nargs="+", default=list(common.FAMILIES))
    parser.add_argument("--cases", nargs="+", default=list(common.CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SEEDS))
    return parser.parse_args()


def main() -> None:
    args = cli()
    spec = SUITES[args.dataset]
    families, cases, seeds = tuple(args.families), tuple(args.cases), tuple(args.seeds)
    for name, selection, allowed in (
        ("families", families, common.FAMILIES),
        ("cases", cases, common.CASES),
        ("seeds", seeds, common.SEEDS),
    ):
        common.require(
            bool(selection) and len(set(selection)) == len(selection)
            and set(selection).issubset(allowed),
            f"Invalid overnight {name} selection: {selection}",
        )
    workers = spec["workers"] if args.workers is None else args.workers
    common.require(workers >= 0, "Invalid workers")
    token = os.environ.get("HF_TOKEN")
    common.require(bool(token), "HF_TOKEN required for unattended archival")
    data_root = args.data_root.resolve()
    dataset_root = args.dataset_root.resolve()
    common.require(data_root.is_dir(), f"Missing dataset source: {data_root}")
    runner = ROOT / spec["runner"]
    namespace = f"{args.dataset}_ftsc_cross_yolo_portability_v1"
    run_root = ROOT / "runs" / namespace
    run_root.mkdir(parents=True, exist_ok=True)
    progress_file = run_root / "overnight_progress.json"
    ledger: dict[str, Any] = {
        "schema": "ftsc_cross_yolo_overnight_v1",
        "dataset": args.dataset,
        "expected_runs": len(families) * len(cases) * len(seeds),
        "latest_step": "starting",
        "runs": {},
    }

    def record(step: str, identity: str | None = None, status: str | None = None, reason: str | None = None) -> None:
        ledger["latest_step"] = step
        ledger["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if identity is not None:
            ledger["runs"][identity] = {"status": status, "reason": reason}
        common.write_json(progress_file, ledger)
        print(f"[{namespace}] {step} {identity or ''} {status or ''}", flush=True)

    def run_dir(family: str, case: str, seed: int) -> Path:
        return run_root / family / case / f"seed_{seed}{spec['suffix']}"

    def run_cli(action: str, selected_families: tuple[str, ...], selected_cases: tuple[str, ...], selected_seeds: tuple[int, ...], *, smoke: bool = False, capture: bool = False) -> str:
        command = [
            sys.executable, "-u", str(runner), "--action", action,
            "--data-root", str(data_root), "--dataset-root", str(dataset_root),
            "--device", args.device, "--workers", str(workers),
            "--hf-repo-id", args.hf_repo_id,
            "--families", *selected_families,
            "--cases", *selected_cases,
            "--seeds", *(str(seed) for seed in selected_seeds),
        ]
        if smoke:
            command.append("--live-model-smoke")
        result = subprocess.run(
            command, check=False, cwd=str(ROOT), text=True, capture_output=capture,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if result.returncode != 0:
            if capture:
                print((result.stdout + "\n" + result.stderr)[-14000:], flush=True)
            raise RuntimeError(f"{namespace}: action={action} failed (exit={result.returncode})")
        return result.stdout if capture else ""

    def check_sources() -> None:
        required = [
            runner, ROOT / "utils/cross_yolo_portability.py",
            ROOT / "utils/hf_result_archive.py",
        ]
        dirs = {"yolov9t": "yolov9", "yolov10n": "yolov10", "yolo11n": "yolo11"}
        for family in families:
            for suffix in ("b1_noftsc", "ftsc_h2"):
                required.append(
                    ROOT / "models_related/models_config" / dirs[family]
                    / spec["yaml_subdir"] / f"{family}_p2_{args.dataset}_{suffix}.yaml"
                )
        missing = [str(file) for file in required if not file.is_file()]
        common.require(not missing, f"Missing GitHub-synced runner/YAML; push local code first: {missing}")

    check_sources()
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics.utils.downloads import attempt_download_asset
    from utils.hf_result_archive import HFResultArchive

    archive = HFResultArchive(
        repo_id=args.hf_repo_id, token=token,
        cache_dir=f"/marimo/hf_results_cache/{namespace}",
    )
    api = HfApi(token=token)

    def fetch_pretrained() -> None:
        cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            for family in families:
                pretrained = ROOT / f"{family}.pt"
                if not pretrained.is_file():
                    downloaded = Path(attempt_download_asset(pretrained.name)).resolve()
                    common.require(downloaded.is_file(), f"Missing pretrained: {pretrained}")
                    if downloaded != pretrained:
                        shutil.copy2(downloaded, pretrained)
                common.require(pretrained.stat().st_size >= 100_000, f"Invalid pretrained checkpoint: {pretrained}")
                print(f"Pretrained {family} sha256={common.sha256(pretrained)}", flush=True)
        finally:
            os.chdir(cwd)

    def prepare_dataset() -> None:
        if args.dataset == "tinyperson":
            from train_tinyperson_scripts import train_all_tinyperson as native
            test_dir = native.prepare_test_set(data_root, dataset_root)
            for seed in seeds:
                native.prepare_seed_dataset(data_root, dataset_root, test_dir, seed)
        else:
            from argparse import Namespace
            from train_levir_scripts import train_varroa_dbss_hit as native
            for seed in seeds:
                native.dataset_for_seed(Namespace(data_root=data_root, dataset_root=dataset_root), seed)

    def restore_archived_runs() -> None:
        variants = tuple(f"{family}__{case}" for family in families for case in cases)
        staging = run_root / "_restore_staging"
        restored = archive.restore_completed_runs(
            dataset_slug=namespace,
            project=staging,
            variants=variants,
            seeds=seeds,
            run_name_fn=lambda _variant, seed: f"seed_{seed}{spec['suffix']}",
            required_artifacts=common.FINAL_ARTIFACTS,
        )
        for row in restored["restored"]:
            family, case = row["case"].split("__", 1)
            destination = run_dir(family, case, row["seed"])
            source = Path(row["run_dir"])
            if destination.exists() and any(destination.iterdir()):
                print(f"Preserving existing local state: {destination}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        print(f"HF recovered completed runs: {len(restored['restored'])}", flush=True)

    def parse_preflight(text: str) -> dict[str, Any]:
        # The runner's final JSON may follow native Ultralytics diagnostics.
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value = json.loads(text[index:])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get("suite") == namespace:
                return value
        raise RuntimeError("Unable to parse final live preflight JSON")

    def verify_or_publish(family: str, case: str, seed: int) -> None:
        current = run_dir(family, case, seed)
        common.require(
            all((current / name).is_file() for name in common.FINAL_ARTIFACTS),
            f"Cannot archive incomplete run: {current}",
        )
        variant = f"{family}__{case}"
        prefix = f"{namespace}/{variant}/{current.name}"
        marker_name = prefix + "/upload_complete.json"
        remote_files = api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset")
        if marker_name in remote_files:
            marker_file = hf_hub_download(
                repo_id=args.hf_repo_id, repo_type="dataset",
                filename=marker_name, token=token,
            )
            marker = json.loads(Path(marker_file).read_text(encoding="utf-8"))
            common.require(
                marker.get("dataset") == namespace and marker.get("case") == variant
                and marker.get("seed") == seed
                and marker.get("run_relative_path") == f"{variant}/{current.name}",
                f"HF completed archive identity mismatch: {prefix}",
            )
            digests = marker.get("artifact_sha256")
            common.require(isinstance(digests, dict), f"HF archive marker invalid: {prefix}")
            for name in common.FINAL_ARTIFACTS:
                common.require(
                    digests.get(name) == common.sha256(current / name),
                    f"HF archive SHA mismatch: {prefix}/{name}",
                )
            print(f"HF archive verified (already uploaded): {prefix}", flush=True)
        else:
            common.require(
                not any(name.startswith(prefix + "/") for name in remote_files),
                f"Unmarked HF archive exists, refusing to overwrite: {prefix}",
            )
            run_cli("archive", (family,), (case,), (seed,))

    record("pretrained")
    fetch_pretrained()
    record("dataset_prepare")
    prepare_dataset()
    record("hf_first_restore")
    restore_archived_runs()
    record("live_preflight_with_synthetic_backward")
    report = parse_preflight(run_cli("preflight", families, cases, seeds, smoke=True, capture=True))
    rows = report.get("rows")
    common.require(
        isinstance(rows, list) and len(rows) == len(families) * len(cases) * len(seeds),
        "Live preflight did not cover all selected runs",
    )
    valid_states = {"NEW_RUN_REQUIRED", "RESUME_REQUIRED", "EVAL_BACKFILL_REQUIRED", "COMPLETE"}
    blocked = [(row.get("family"), row.get("case"), row.get("seed"), row.get("reason"))
               for row in rows if row.get("status") not in valid_states]
    common.require(not blocked, f"Blocked during live preflight; no training launched: {blocked}")
    common.require(
        all(isinstance(row.get("runtime"), dict)
            and row["runtime"].get("synthetic_loss") is True
            and row["runtime"].get("synthetic_backward") is True for row in rows),
        "Runtime forward/loss/backward smoke missing or incomplete; no training permitted",
    )
    print("Preflight states:", dict(collections.Counter(row["status"] for row in rows)), flush=True)

    failed: list[str] = []
    try:
        for family in families:
            for seed in seeds:
                for case in cases:
                    identity = f"{family}/{case}/seed_{seed}"
                    record("train_resume_or_eval_backfill", identity, "RUNNING")
                    try:
                        # Dataset-native runner chooses NEW/RESUME/EVAL_BACKFILL/COMPLETE.
                        run_cli("train", (family,), (case,), (seed,))
                        verify_or_publish(family, case, seed)
                        record("validated_and_archived", identity, "COMPLETE")
                    except Exception as error:
                        # The live smoke and protocol checks have already passed;
                        # an isolated training/upload failure must not waste the
                        # remaining overnight GPU window. Incomplete experiments
                        # remain BLOCKED, never promoted to COMPLETE.
                        failed.append(identity)
                        record("run_failed_others_continue", identity, "BLOCKED", f"{type(error).__name__}: {error}")
                        continue
    finally:
        # Persist partial summaries even if one run fails: incomplete seed sets
        # cannot appear as full aggregates in the dataset-native runner.
        try:
            record("aggregate_current_results")
            run_cli("aggregate", families, cases, seeds)
        except Exception as error:
            print(f"Aggregate failed: {type(error).__name__}: {error}", flush=True)
    if failed:
        record("finished_with_blockers")
        raise RuntimeError(
            f"{namespace}: {len(failed)} run(s) require follow-up; other runs were processed: {failed}. "
            f"Details in {progress_file}."
        )
    record("finished")
    print(f"All {len(families) * len(cases) * len(seeds)} {namespace} runs COMPLETE + HF verified.", flush=True)


if __name__ == "__main__":
    main()
