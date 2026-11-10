#!/usr/bin/env python3
"""Safe, resumable FTSC Hugging Face archive reorganization.

This tool reads an audited checkpoint inventory + migration plan produced from
TLMHoang/FTSC_results. It *never* selects an experiment from unrelated metrics.
Exactly identical SHA-256 best.pt payloads share one canonical checkpoint,
while all other run identities/seeds remain distinct.

- inspect: offline integrity / source-target inventory summary.
- preflight: HF read-only validation of source paths and LFS identities.
- stage: copy to the new layout without deleting anything. This is opt-in.
- verify: read-only SHA/source/destination verification.
- retire: deliberately not implemented until all notebook consumers are updated.

HF_TOKEN must be supplied through the trusted runtime environment, never CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
AUDIT_DIR = WORKSPACE / "note" / "hf_reorganization_2026_11_08"
DEFAULT_PLAN = AUDIT_DIR / "migration_plan.json"
REPO_ID = "TLMHoang/FTSC_results"
EXPECTED_SEEDS = {42, 43, 44}


class MigrationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MigrationError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    require(plan.get("schema") == "ftsc_hf_reorganization_plan_v1", "Unknown plan schema")
    require(plan.get("source_repo") == REPO_ID and plan.get("repo_type") == "dataset",
            "Refusing to operate on an unexpected repository")
    entries = plan.get("entries", [])
    require(len(entries) == plan.get("summary", {}).get("best_total"),
            "Checkpoint count differs from the audited migration plan")
    assert_plan_invariants(entries)
    return plan


def assert_plan_invariants(entries: list[dict[str, Any]]) -> None:
    dest: set[str] = set()
    by_sha: dict[str, list[dict[str, Any]]] = {}
    by_path = {row["best_path"]: row for row in entries}
    require(len(by_path) == len(entries), "Duplicate source best.pt paths")
    for row in entries:
        source = row["source_run_dir"]
        target = row["target_run_dir"]
        require(source and target and not source.startswith("/") and not target.startswith("/"),
                "Invalid absolute source/destination path")
        require(".." not in PurePosixPath(source).parts and ".." not in PurePosixPath(target).parts,
                "Path traversal in plan")
        require(source != target and target not in dest, "Migration target collision")
        require("//" not in source and "//" not in target,
                "Malformed case path (empty directory segment)")
        source_parts, target_parts = PurePosixPath(source).parts, PurePosixPath(target).parts
        require(len(source_parts) >= 3 and len(target_parts) >= 4,
                "Malformed checkpoint source/target depth")
        require(source_parts[-2:] == target_parts[-2:],
                "Case and seed folder names MUST be byte-for-byte unchanged: "
                f"{source} -> {target}")
        require(PurePosixPath(target).parts[0] == row["dataset"],
                "Target folder belongs to wrong dataset")
        dest.add(target)
        require(int(row["seed"]) in EXPECTED_SEEDS, "Seed outside 42/43/44")
        require(row["best_path"] == source + "/weights/best.pt",
                "Source directory does not match checkpoint path")
        require(row["dataset"] in {"LEVIR-Ship", "TinyPerson", "Varroa", "VisDrone"},
                "Unexpected dataset")
        if row.get("sha256"):
            require(len(row["sha256"]) == 64, "Malformed SHA")
            by_sha.setdefault(row["sha256"], []).append(row)
        else:
            require(row["action"].startswith("REVIEW_"),
                    "Unverified checkpoint must not be silently deduplicated")
    for sha, group in by_sha.items():
        require(len({row["seed"] for row in group}) == 1,
                f"Same checkpoint SHA across independent seeds: {sha}")
        require(len({row["dataset"] for row in group}) == 1,
                f"Same checkpoint SHA across datasets: {sha}")
        primary = [r for r in group if r["action"] == "STAGE_PRIMARY_CHECKPOINT"]
        require(len(primary) == 1, f"SHA {sha}: expected exactly one canonical checkpoint")
        target = primary[0]["target_run_dir"] + "/weights/best.pt"
        for row in group:
            require(row["canonical_target_checkpoint"] == target,
                    f"SHA {sha}: inconsistent canonical reference")
            require(row["canonical_sha_source"] in by_path,
                    f"SHA {sha}: source checkpoint disappeared from plan")


def summarize(plan: dict[str, Any]) -> dict[str, Any]:
    rows = plan["entries"]
    return {
        "source_repo": REPO_ID,
        "best_checkpoint_paths": len(rows),
        "verified_sha_from_completion_markers": sum(
            bool(r.get("sha256")) and not r.get("sha256_source")
            for r in rows),
        "historical_source_provenance_only": sum(
            r.get("sha256_source") == "historical_source_provenance_json"
            for r in rows),
        "unique_known_best_sha": len({r["sha256"] for r in rows if r.get("sha256")}),
        "confirmed_duplicate_aliases": sum(r["action"] == "METADATA_ONLY_ALIAS" for r in rows),
        "unverified_best": sum(r["action"].startswith("REVIEW_") for r in rows),
        "by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "by_scope": dict(Counter(r["scope"] for r in rows)),
        "legacy_paths_retained_until_consumer_migration": True,
    }


def api_from_env() -> Any:
    token = os.environ.get("HF_TOKEN", "").strip()
    require(token, "HF_TOKEN is required for HF preflight/stage/verify; do not paste it into a prompt")
    from huggingface_hub import HfApi
    return HfApi(token=token)


def get_remote_snapshot(api: Any) -> tuple[str, set[str]]:
    # Failure to read HF must fail closed, never trigger a migration/retraining.
    info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    revision = info.sha
    require(isinstance(revision, str) and len(revision) == 40,
            "Unexpected Hugging Face revision")
    files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", revision=revision))
    require(files, "Empty HF repository listing is not acceptable")
    return revision, files


def remote_sha256(api: Any, paths: list[str], *, revision: str) -> dict[str, str | None]:
    """Return LFS blob SHA-256 if API exposes it, else None; never invent hashes."""
    out: dict[str, str | None] = {}
    for offset in range(0, len(paths), 75):
        chunk = paths[offset:offset + 75]
        infos = api.get_paths_info(repo_id=REPO_ID, repo_type="dataset",
                                   paths=chunk, revision=revision, expand=True)
        for info in infos:
            lfs = getattr(info, "lfs", None)
            payload = lfs if isinstance(lfs, dict) else vars(lfs) if lfs else {}
            out[getattr(info, "path")] = payload.get("sha256")
    return out


# Journal LEVIR B1 FTSC-off is a three-seed cohort. Seed 44 is still
# training/uploading at the time the audit was prepared; never treat 42/43
# alone as the completed experiment.
JOURNAL_B1_PREFIX = "levir_v8_p23_ftsc_off_hf_first_v1/V8_B1_P23_NOFTSC"
JOURNAL_B1_TARGET = "LEVIR-Ship/journal_experiments/ftsc_off_hf_first_v1/V8_B1_P23_NOFTSC"
JOURNAL_B1_SEEDS = (42, 43, 44)


def require_journal_b1_complete(entries: list[dict[str, Any]]) -> None:
    cohort = [row for row in entries
              if row["source_run_dir"].startswith(JOURNAL_B1_PREFIX + "/")]
    seeds = [row["seed"] for row in cohort]
    require(len(cohort) == 3 and set(seeds) == set(JOURNAL_B1_SEEDS),
            "Journal LEVIR FTSC-off backfill requires three distinct seeds "
            "(42, 43, 44). Seed 44 is still pending in the migration plan; "
            "do not stage a partial paper cohort. Re-run notebook Cell 4 "
            "after seed_44 checkpoint and upload_complete.json appear on HF.")
    for row in cohort:
        seed = row["seed"]
        src = f"{JOURNAL_B1_PREFIX}/seed_{seed}"
        dst = f"{JOURNAL_B1_TARGET}/seed_{seed}"
        require(row["source_run_dir"] == src and row["target_run_dir"] == dst,
                "Journal B1 suite/case/seed target was changed or mixed")
        require(row["case"] == "V8_B1_P23_NOFTSC"
                and row["dataset"] == "LEVIR-Ship",
                "Journal B1 dataset/case mismatch")


def prepare_journal_b1_three_seed_cohort(
    api: Any, plan: dict[str, Any], revision: str, files: set[str]
) -> dict[str, Any]:
    """Read-only HF provenance validation; add seed44 IN MEMORY when completed.

    Nothing on HF or local disk is changed here. Notebook Cell 5 hydrates SHA
    and commits the refreshed local plan and registry as one operation.
    """
    from huggingface_hub import hf_hub_download

    def load_json(remote_path: str) -> dict[str, Any]:
        cached = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=remote_path,
            revision=revision, token=api.token,
        )
        content = json.loads(Path(cached).read_text(encoding="utf-8"))
        require(isinstance(content, dict), f"Invalid JSON document: {remote_path}")
        return content

    # Do not silently start migration if only 2/3 journal results are present.
    for seed in JOURNAL_B1_SEEDS:
        prefix = f"{JOURNAL_B1_PREFIX}/seed_{seed}"
        needed = (
            "weights/best.pt", "weights/last.pt",
            "upload_complete.json", "experiment_manifest.json",
            "args.yaml", "config.yaml", "results.csv",
            "run_summary.json", "evaluation_metrics.json",
        )
        missing = [f"{prefix}/{suffix}" for suffix in needed
                   if f"{prefix}/{suffix}" not in files]
        require(not missing,
                f"Journal LEVIR B1 seed_{seed} is incomplete or still uploading; "
                f"missing HF artifacts: {missing}. No migration will run.")

    infos = remote_sha256(
        api,
        [f"{JOURNAL_B1_PREFIX}/seed_{seed}/weights/best.pt"
         for seed in JOURNAL_B1_SEEDS],
        revision=revision,
    )
    manifests: dict[int, dict[str, Any]] = {}
    sha_by_seed: dict[int, str] = {}
    archive_runs = []
    for seed in JOURNAL_B1_SEEDS:
        prefix = f"{JOURNAL_B1_PREFIX}/seed_{seed}"
        manifest = load_json(f"{prefix}/experiment_manifest.json")
        upload = load_json(f"{prefix}/upload_complete.json")
        contract = manifest.get("scientific_contract", {})
        require(isinstance(contract, dict)
                and contract.get("schema") == "levir_v8_p23_ftsc_off_hf_first_v1",
                f"Journal B1 seed_{seed}: unexpected scientific-contract schema")
        require(contract.get("dataset") == "LEVIR-Ship"
                and contract.get("case") == "V8_B1_P23_NOFTSC"
                and contract.get("seed") == seed
                and contract.get("canonical_seeds") == [42, 43, 44]
                and contract.get("ftsc_enabled") is False
                and contract.get("detect_inputs") == [19, 22],
                f"Journal B1 seed_{seed}: wrong FTSC state, graph or seed")
        require(upload.get("case") == "V8_B1_P23_NOFTSC"
                and upload.get("seed") == seed
                and upload.get("schema") == "ftsc_seed_archive_v1",
                f"Journal B1 seed_{seed}: invalid upload completion marker")
        artifact_sha = upload.get("artifact_sha256", {})
        best_sha = artifact_sha.get("weights/best.pt")
        lfs_sha = infos.get(f"{prefix}/weights/best.pt")
        require(isinstance(best_sha, str) and len(best_sha) == 64
                and best_sha == lfs_sha,
                f"Journal B1 seed_{seed}: best.pt LFS SHA does not match "
                "upload completion marker")
        manifests[seed] = contract
        sha_by_seed[seed] = best_sha
        archive_runs.append(prefix)

    baseline = manifests[42]
    for seed in (43, 44):
        contract = manifests[seed]
        keys = (
            "schema", "dataset", "case", "canonical_seeds", "split_seed",
            "dataset_manifest_sha256", "model_config_sha256",
            "pretrained_sha256", "ftsc_enabled", "detect_inputs",
        )
        for key in keys:
            require(contract.get(key) == baseline.get(key),
                    f"Journal B1 seed_{seed}: protocol differs from seed_42: {key}")
        train_42 = dict(baseline.get("training") or {})
        train_other = dict(contract.get("training") or {})
        train_42.pop("seed", None)
        train_other.pop("seed", None)
        require(train_42 and train_42 == train_other,
                f"Journal B1 seed_{seed}: training recipe differs from seed_42")

    rows = plan["entries"]
    existing = {r["source_run_dir"]: r for r in rows}
    for seed in JOURNAL_B1_SEEDS:
        src = archive_runs[seed - 42]
        dst = f"{JOURNAL_B1_TARGET}/seed_{seed}"
        if src in existing:
            row = existing[src]
            require(row["sha256"] == sha_by_seed[seed]
                    and row["target_run_dir"] == dst,
                    f"Journal B1 seed_{seed}: local migration plan "
                    "does not match the verified HF checkpoint")
            continue
        require(seed == 44,
                f"Journal B1 seed_{seed}: unexpected missing planned seed")
        # Never conflate the NEW auto-optimizer training suite with the OLD
        # portability-backfill AdamW suite. Case and seed leaves stay intact.
        require(not any(r["target_run_dir"] == dst for r in rows),
                "Journal B1 seed_44 destination collides with another case")
        best = f"{src}/weights/best.pt"
        require(not any(r.get("sha256") == sha_by_seed[seed]
                        and r["seed"] != seed for r in rows),
                "Journal B1 seed_44 SHA coincides with another training seed")
        rows.append({
            "dataset": "LEVIR-Ship", "scope": "journal_experiments",
            "detail": "ftsc_off_hf_first_v1", "arch": "",
            "case": "V8_B1_P23_NOFTSC", "seed": seed,
            "seed_folder": f"seed_{seed}",
            "source_run_dir": src, "target_run_dir": dst,
            "best_path": best, "sha256": sha_by_seed[seed],
            "bytes": None, "has_marker": True,
            "canonical_sha_source": best,
            "canonical_target_checkpoint": dst + "/weights/best.pt",
            "action": "STAGE_PRIMARY_CHECKPOINT",
        })
    plan["summary"]["best_total"] = len(rows)
    plan["summary"]["with_marker_sha"] = sum(
        bool(r.get("sha256")) and not r.get("sha256_source") for r in rows
    )
    plan["summary"]["with_any_sha"] = sum(bool(r.get("sha256")) for r in rows)
    plan["summary"]["unique_sha"] = len({
        r["sha256"] for r in rows if r.get("sha256")
    })
    plan["summary"]["unresolved_sha"] = sum(
        not bool(r.get("sha256")) for r in rows)
    require_journal_b1_complete(rows)
    assert_plan_invariants(rows)
    return {
        "case": "V8_B1_P23_NOFTSC",
        "namespace": JOURNAL_B1_PREFIX,
        "verified_seeds": list(JOURNAL_B1_SEEDS),
        "best_sha256_by_seed": sha_by_seed,
        "indexed_best_pt": len(rows),
        "source_modified": False,
    }


def validate_remote_sources(api: Any, entries: list[dict[str, Any]],
                            revision: str, files: set[str]) -> dict[str, Any]:
    require_journal_b1_complete(entries)
    source_paths = [r["best_path"] for r in entries]
    source_set = set(source_paths)
    missing = [p for p in source_paths if p not in files]
    require(not missing, f"HF source changed; missing checkpoint paths: {missing[:5]}")
    # New experiments may have arrived after the dated audit. Never silently
    # leave any newly uploaded original checkpoint outside the migration plan.
    target_dataset_roots = {"LEVIR-Ship", "TinyPerson", "Varroa", "VisDrone"}
    untracked = sorted(
        p for p in files
        if p.endswith("/weights/best.pt")
        and p not in source_set
        and p.split("/", 1)[0] not in target_dataset_roots
    )
    require(not untracked,
            f"New HF checkpoints appeared after the audit; update inventory "
            f"before migration: {untracked[:10]}")
    actual = remote_sha256(api, source_paths, revision=revision)
    disagreements, unverifiable = [], []
    for row in entries:
        expected = row.get("sha256")
        got = actual.get(row["best_path"])
        if not got:
            unverifiable.append(row["best_path"])
        elif expected and got != expected:
            disagreements.append({"path": row["best_path"],
                                  "marker_sha": expected, "lfs_sha": got})
    require(not disagreements,
            f"Checkpoint SHA mismatch vs trusted archive marker: {disagreements[:5]}")
    require(not unverifiable,
            f"Source checkpoint bytes cannot be independently identified from "
            f"the Hugging Face LFS index: {unverifiable[:5]}")
    return {"remote_revision": revision,
            "source_count": len(source_paths),
            "lfs_sha_verified": sum(bool(actual.get(p)) for p in source_paths),
            "unverifiable_lfs_sha": unverifiable,
            "marker_sha_conflicts": disagreements}


def reference_data(row: dict[str, Any], revision: str) -> bytes:
    return (json.dumps({
        "schema": "ftsc_checkpoint_reference_v1",
        "dataset": row["dataset"], "case": row["case"], "seed": row["seed"],
        "source_run_dir": row["source_run_dir"],
        "source_best_pt": row["best_path"],
        "source_repo": REPO_ID,
        "best_sha256": row.get("sha256"),
        "canonical_best_pt": row["canonical_target_checkpoint"],
        "selection": row["action"],
        "meaning": ("Same checkpoint binary; store evaluation/provenance only."
                    if row["action"] == "METADATA_ONLY_ALIAS"
                    else "Independent run or canonical checkpoint."),
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_copy_jobs(entries: list[dict[str, Any]], files: set[str],
                    revision: str) -> list[tuple[str, str, bytes | None]]:
    """For aliases: copy only metadata, not any .pt model weights.

    The source archive upload_complete.json is renamed to
    legacy_upload_complete.json because the old completion marker does not
    certify the newly constructed directory. The old source remains untouched.
    """
    jobs = []
    for row in entries:
        source_prefix = row["source_run_dir"].rstrip("/") + "/"
        target_prefix = row["target_run_dir"].rstrip("/") + "/"
        source_files = sorted(s for s in files if s.startswith(source_prefix))
        require(row["best_path"] in source_files,
                f"Missing best checkpoint in inventory: {row['best_path']}")
        is_alias = row["action"] == "METADATA_ONLY_ALIAS"
        for src in source_files:
            rel = src[len(source_prefix):]
            if is_alias and rel.startswith("weights/") and rel.endswith(".pt"):
                continue
            if rel == "upload_complete.json":
                rel = "legacy_upload_complete.json"
            dest = target_prefix + rel
            jobs.append((src, dest, None))
        reference_path = target_prefix + "checkpoint_reference.json"
        jobs.append(("", reference_path, reference_data(row, revision)))
    dests = [d for _, d, _ in jobs]
    require(len(dests) == len(set(dests)), "Destination file collisions in reorganization")
    return jobs


def validate_registry_against_plan(plan: dict[str, Any]) -> None:
    """Hard-stop if the three registry files do not match the plan exactly.

    Important for resuming after an interrupted local hydrate/write: registry
    content must describe the SAME canonical checkpoint for every case/seed.
    """
    import csv

    registry = AUDIT_DIR / "registry"
    csv_path = registry / "checkpoint_index.csv"
    provenance_path = registry / "provenance_map.json"
    manifest_path = registry / "migration_manifest.json"
    for path in (csv_path, provenance_path, manifest_path):
        require(path.is_file(), f"Missing required migration registry: {path}")

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        index_rows = list(csv.DictReader(stream))
    expected_rows = plan["entries"]
    require(len(index_rows) == len(expected_rows),
            "Registry checkpoint count differs from the migration plan")
    csv_index = {r["source_best_pt"]: r for r in index_rows}
    require(len(csv_index) == len(index_rows), "Registry contains duplicate source rows")

    for r in expected_rows:
        actual = csv_index.get(r["best_path"])
        require(actual is not None, f"Source missing from registry: {r['best_path']}")
        for key, expected in (
            ("dataset", r["dataset"]),
            ("case", r["case"]),
            ("seed", str(r["seed"])),
            ("sha256", r.get("sha256") or ""),
            ("action", r["action"]),
            ("target_run_dir", r["target_run_dir"]),
            ("canonical_target_checkpoint", r["canonical_target_checkpoint"]),
        ):
            require(actual[key] == expected,
                    f"Registry mismatch ({key}) for {r['best_path']}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(provenance.get("schema") == "ftsc_checkpoint_provenance_map_v1",
            "Unexpected checkpoint provenance registry schema")
    by_sha = provenance.get("by_sha", {})
    require(set(by_sha) == {r["sha256"] for r in expected_rows if r.get("sha256")},
            "Provenance registry SHA set differs from the migration plan")
    for r in expected_rows:
        if not r.get("sha256"):
            continue
        record = by_sha[r["sha256"]]
        require(record["canonical_target_checkpoint"] == r["canonical_target_checkpoint"],
                f"Provenance registry canonical target mismatch: {r['best_path']}")
        require(r["best_path"] in record["source_aliases"],
                f"Provenance registry missing source alias: {r['best_path']}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "ftsc_checkpoint_migration_manifest_v1"
            and manifest.get("repo") == REPO_ID,
            "Unexpected migration registry manifest")
    n_primary = sum(r["action"] == "STAGE_PRIMARY_CHECKPOINT" for r in expected_rows)
    require(manifest.get("target_primary_checkpoint_count") == n_primary,
            "Registry canonical checkpoint count differs from the migration plan")
    require(manifest.get("source_best_count") == len(expected_rows),
            "Registry source checkpoint count differs from the plan")


def registry_jobs() -> list[tuple[str, str, bytes | None]]:
    """Stage the global index alongside the per-dataset reorganized tree."""
    registry = AUDIT_DIR / "registry"
    names = ("checkpoint_index.csv", "provenance_map.json", "migration_manifest.json")
    for name in names:
        require((registry / name).is_file(),
                f"Migration registry artifact is missing: {registry / name}")
    return [("", "registry/" + name, (registry / name).read_bytes())
            for name in names]


def file_identities(api: Any, paths: list[str], *, revision: str) -> dict[str, tuple[str, str]]:
    """Read byte identities without downloading entire large checkpoints."""
    out: dict[str, tuple[str, str]] = {}
    for offset in range(0, len(paths), 75):
        chunk = paths[offset:offset + 75]
        infos = api.get_paths_info(repo_id=REPO_ID, repo_type="dataset",
                                   paths=chunk, revision=revision, expand=True)
        for info in infos:
            lfs = getattr(info, "lfs", None)
            meta = lfs if isinstance(lfs, dict) else vars(lfs) if lfs else {}
            if meta.get("sha256"):
                out[info.path] = ("lfs", meta["sha256"])
            elif getattr(info, "blob_id", None):
                out[info.path] = ("git", info.blob_id)
    return out


def git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def check_existing_targets(api: Any, jobs: list[tuple[str, str, bytes | None]],
                           *, revision: str, files: set[str]) -> int:
    """Refuse an existing destination unless its bytes equal the proposed source.

    Enables an interrupted migration to resume without overwriting or treating
    conflicting checkpoint/metric content as complete.
    """
    existing = [(s, d, b) for s, d, b in jobs if d in files]
    if not existing:
        return 0
    check_paths = sorted({p for s, d, _ in existing for p in (s, d) if p})
    ids = file_identities(api, check_paths, revision=revision)
    conflicts = []
    for src, dest, payload in existing:
        dest_id = ids.get(dest)
        if payload is not None:
            # A repo-level .gitattributes rule may store JSON/CSV in LFS.
            valid_ids = {
                ("git", git_blob_oid(payload)),
                ("lfs", hashlib.sha256(payload).hexdigest()),
            }
            expected = sorted(valid_ids)
            matches = dest_id in valid_ids
        else:
            expected = ids.get(src)
            matches = bool(expected) and dest_id == expected
        if not matches:
            conflicts.append((src, dest, expected, dest_id))
    require(not conflicts,
            f"Existing target conflicts (will not overwrite): {conflicts[:4]}")
    return len(existing)


def execute_stage(api: Any, jobs: list[tuple[str, str, bytes | None]],
                  *, revision: str, files: set[str], batch_size: int,
                  max_ops: int | None, execute: bool) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy
    already_verified = check_existing_targets(
        api, jobs, revision=revision, files=files)
    planned = [(s, d, b) for s, d, b in jobs if d not in files]
    summary = {
        "source_revision": revision,
        "total_expected_files": len(jobs),
        "verified_existing_files": already_verified,
        "remaining_operations": len(planned),
        "checkpoint_copies_remaining": sum(
            d.endswith("/weights/best.pt") for _, d, _ in planned),
        "status": "DRY_RUN" if not execute else "STAGING",
    }
    if not execute or not planned:
        if not planned:
            summary["status"] = "ALREADY_STAGED_VERIFIED"
        return summary
    require(1 <= batch_size <= 70, "Batch size must be 1..70")
    todo = len(planned) if max_ops is None else max_ops
    require(0 < todo <= len(planned),
            "max_ops must be positive and <= remaining operations")
    for start in range(0, todo, batch_size):
        batch = planned[start:min(start + batch_size, todo)]
        operations = [
            CommitOperationAdd(path_in_repo=dest, path_or_fileobj=io.BytesIO(payload))
            if payload is not None else
            CommitOperationCopy(src_path_in_repo=src, path_in_repo=dest,
                                src_revision=revision)
            for src, dest, payload in batch
        ]
        # Staging only; originals and old namespaces are untouched.
        api.create_commit(
            repo_id=REPO_ID, repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Stage FTSC HF archive reorganization "
                f"({start + 1}-{start + len(batch)} / {len(planned)} pending)"))
        print(f"[stage] copied {start + len(batch)} previously missing paths",
              flush=True)
    summary["status"] = "PARTIAL_STAGE" if todo < len(planned) else "STAGED"
    summary["ops_executed"] = todo
    return summary


def verify_targets(api: Any, entries: list[dict[str, Any]], revision: str,
                   files: set[str]) -> dict[str, Any]:
    jobs = build_copy_jobs(entries, files, revision) + registry_jobs()
    missing = [dest for _, dest, _ in jobs if dest not in files]
    require(not missing, f"Missing target migration files: {missing[:5]}")
    match_count = check_existing_targets(
        api, jobs, revision=revision, files=files)
    canonical = [r for r in entries
                 if r["action"] == "STAGE_PRIMARY_CHECKPOINT"
                 or r["action"].startswith("REVIEW_")]
    expected = {r["target_run_dir"] + "/weights/best.pt": r for r in canonical}
    actual = remote_sha256(api, list(expected), revision=revision)
    mismatches = [
        (path, row["sha256"], actual.get(path))
        for path, row in expected.items()
        if row.get("sha256") and actual.get(path) != row["sha256"]
    ]
    require(not mismatches, f"Canonical best.pt SHA differs: {mismatches[:5]}")
    return {
        "all_expected_target_files_verified": match_count,
        "new_canonical_checkpoints": len(expected),
        "known_marker_sha_verified": sum(bool(r.get("sha256")) for r in canonical),
        "still_unverified_marker_sha": sum(not bool(r.get("sha256")) for r in canonical),
        "legacy_namespaces_untouched": True,
    }


def hydrate_unverified_sha(plan: dict[str, Any], live_sha: dict[str, str | None]) -> dict[str, Any]:
    """Complete unknown hashes from real HF LFS metadata and rebuild alias links.

    Source equivalence is identified by LFS SHA-256, not filename, model
    configuration, reported AP or checkpoint size.
    """
    import csv

    entries = plan["entries"]
    priorities = {
        "original_runs": 0,
        "journal_experiments": 10,
        "historical_reconstructions": 20,
        "cross_yolo_portability": 30,
        "resolution_1536": 40,
        "additional_ablations": 50,
        "derived_evaluations": 90,
    }
    for row in entries:
        got = live_sha.get(row["best_path"])
        require(isinstance(got, str) and len(got) == 64,
                f"Cannot hydrate best.pt without HF LFS SHA: {row['best_path']}")
        require(not row.get("sha256") or row["sha256"] == got,
                f"Conflict between archive provenance and actual best.pt: {row['best_path']}")
        if not row.get("sha256"):
            row["sha256"] = got
            row["sha256_source"] = "live_hf_lfs_verified"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in entries:
        grouped.setdefault(row["sha256"], []).append(row)
    duplicates = []
    for sha, records in grouped.items():
        require(len({r["seed"] for r in records}) == 1,
                f"SHA used across multiple training seeds: {sha}")
        require(len({r["dataset"] for r in records}) == 1,
                f"SHA used across datasets: {sha}")
        records.sort(key=lambda r: (priorities.get(r["scope"], 99), r["best_path"]))
        primary = records[0]
        for index, row in enumerate(records):
            row["canonical_sha_source"] = primary["best_path"]
            row["canonical_target_checkpoint"] = primary["target_run_dir"] + "/weights/best.pt"
            row["action"] = "STAGE_PRIMARY_CHECKPOINT" if index == 0 else "METADATA_ONLY_ALIAS"
        if len(records) > 1:
            duplicates.append({
                "sha256": sha, "count": len(records), "seed": primary["seed"],
                "dataset": primary["dataset"], "primary": primary["best_path"],
                "aliases": [r["best_path"] for r in records[1:]],
                "cross_seed": False,
            })
    plan["duplicate_groups"] = duplicates
    plan["summary"].update({
        "with_any_sha": len(entries),
        "unique_sha": len(grouped),
        "duplicate_aliases": len(entries) - len(grouped),
        "duplicate_groups": len(duplicates),
        "unresolved_sha": 0,
        "cross_seed_same_sha_groups": 0,
    })
    assert_plan_invariants(entries)
    return plan


def write_hydrated_registry(plan: dict[str, Any], path: Path) -> None:
    """Atomically update local audit/registry; does not change the HF repo."""
    import csv
    import tempfile

    registry = AUDIT_DIR / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    rows = plan["entries"]

    fields = (
        "dataset", "scope", "detail", "arch", "case", "seed", "source_best_pt",
        "sha256", "sha256_source", "action", "target_run_dir",
        "canonical_target_checkpoint",
    )
    import io as _io
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields))
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "dataset": row["dataset"], "scope": row["scope"],
            "detail": row.get("detail", ""), "arch": row.get("arch", ""),
            "case": row["case"], "seed": row["seed"],
            "source_best_pt": row["best_path"], "sha256": row["sha256"],
            "sha256_source": row.get("sha256_source", "upload_complete_marker"),
            "action": row["action"], "target_run_dir": row["target_run_dir"],
            "canonical_target_checkpoint": row["canonical_target_checkpoint"],
        })

    provenance: dict[str, Any] = {
        "schema": "ftsc_checkpoint_provenance_map_v1",
        "repository": REPO_ID, "policy": plan["policy"], "by_sha": {},
    }
    for row in rows:
        data = provenance["by_sha"].setdefault(row["sha256"], {
            "canonical_source_checkpoint": row["canonical_sha_source"],
            "canonical_target_checkpoint": row["canonical_target_checkpoint"],
            "dataset": row["dataset"], "seed": row["seed"],
            "source_aliases": [], "source_evaluations": [],
        })
        data["source_aliases"].append(row["best_path"])
        data["source_evaluations"].append({
            "source_run_dir": row["source_run_dir"], "scope": row["scope"],
            "case": row["case"], "action": row["action"],
        })
    # Preserve a complete local source inventory when the journal backfill
    # seed_44 finishes after the original audit. Do not treat it as a separate
    # unrelated experiment or silently leave it out of the inventory.
    inventory_path = AUDIT_DIR / "checkpoint_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    existing_best = {r["path"] for r in inventory["checkpoint_rows"]}
    existing_last = {r["path"] for r in inventory["non_best_pt"]}
    existing_marker = set(inventory["completion_markers"])
    for row in rows:
        if not row["source_run_dir"].startswith(JOURNAL_B1_PREFIX + "/"):
            continue
        prefix = row["source_run_dir"]
        best = prefix + "/weights/best.pt"
        last = prefix + "/weights/last.pt"
        marker = prefix + "/upload_complete.json"
        if best not in existing_best:
            inventory["checkpoint_rows"].append({
                "path": best, "bytes": None, "lfs": True, "kind": "best",
                "checkpoint_dir": prefix, "has_marker": True,
                "marker_sha256": row["sha256"],
                "marker_schema": "ftsc_seed_archive_v1",
            })
            existing_best.add(best)
        if last not in existing_last:
            inventory["non_best_pt"].append({
                "path": last, "bytes": None, "lfs": True, "kind": "last",
            })
            existing_last.add(last)
        if marker not in existing_marker:
            inventory["completion_markers"].append(marker)
            existing_marker.add(marker)
    inventory["checkpoint_rows"].sort(key=lambda r: r["path"])
    inventory["non_best_pt"].sort(key=lambda r: r["path"])
    inventory["completion_markers"].sort()
    require(len(inventory["checkpoint_rows"]) == len(rows),
            "Local inventory/plan checkpoint totals differ; cannot publish")
    manifest = {
        "schema": "ftsc_checkpoint_migration_manifest_v1",
        "repo": REPO_ID, "source_best_count": len(rows),
        "target_primary_checkpoint_count": len(provenance["by_sha"]),
        "known_duplicate_aliases": sum(r["action"] == "METADATA_ONLY_ALIAS" for r in rows),
        "unverified_rows": [],
        "required_phases": [
            "HF write scope", "preflight LFS SHA parity",
            "copy to provenance layout", "verify byte identity and consumers",
            "explicit retirement review",
        ],
        "remote_status": "AUDITED_STAGING_PENDING",
    }
    contents = (
        (inventory_path, json.dumps(inventory, indent=2) + "\n"),
        (path, json.dumps(plan, indent=2) + "\n"),
        (registry / "checkpoint_index.csv", buf.getvalue()),
        (registry / "provenance_map.json", json.dumps(provenance, indent=2) + "\n"),
        (registry / "migration_manifest.json", json.dumps(manifest, indent=2) + "\n"),
    )
    for target, data in contents:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, encoding="utf-8",
            delete=False, prefix=target.name + ".tmp.") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--mode", choices=("inspect", "preflight", "hydrate", "stage", "verify", "retire"),
                        default="inspect")
    parser.add_argument("--execute", action="store_true",
                        help="Required for staging operations. Never deletes legacy paths.")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-ops", type=int, default=None,
                        help="Explicit number of operations authorized for this invocation")
    parser.add_argument("--expected-revision",
                        help="Optional 40-hex SHA to pin the audited source revision")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = read_plan(args.plan)
    report = {"plan_sha256": sha256_json(plan), **summarize(plan)}
    if args.mode == "inspect":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.mode == "retire":
        raise MigrationError(
            "Legacy deletion is intentionally disabled: notebook/recovery scripts "
            "still reference old HF paths. First stage, verify and migrate consumers.")
    if args.mode in ("stage", "verify"):
        validate_registry_against_plan(plan)
    api = api_from_env()
    revision, files = get_remote_snapshot(api)
    if args.expected_revision:
        require(revision == args.expected_revision, "Remote revision changed since audit")
    if args.mode in ("preflight", "hydrate", "stage"):
        report["source_preflight"] = validate_remote_sources(
            api, plan["entries"], revision, files)
    if args.mode == "hydrate":
        all_sha = remote_sha256(api, [r["best_path"] for r in plan["entries"]], revision=revision)
        plan = hydrate_unverified_sha(plan, all_sha)
        report["hydrated_summary"] = plan["summary"]
        if args.execute:
            write_hydrated_registry(plan, args.plan)
        else:
            report["hydrated_summary"]["status"] = "PREVIEW_ONLY"
    if args.mode == "stage":
        require(all(r.get("sha256") for r in plan["entries"]),
                "All source SHA must be hydrated before staging; run --mode hydrate --execute")
        jobs = build_copy_jobs(plan["entries"], files, revision) + registry_jobs()
        report["stage"] = execute_stage(
            api, jobs, revision=revision, files=files,
            batch_size=args.batch_size,
            max_ops=args.max_ops,
            execute=args.execute)
    if args.mode == "verify":
        report["target_verification"] = verify_targets(
            api, plan["entries"], revision, files)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MigrationError as exc:
        print("[FTSC HF migration BLOCKED]", str(exc), file=sys.stderr)
        sys.exit(2)
