"""Strict LEVIR RECON_H2_V1 reuse gate before any baseline-only training.

Read-only on source files, checkpoints and HF; the historical dataset hashing
helper may update its .ftsc_paper_content_hash_cache_v1.json cache.
A FAIL means DO NOT present newly trained OFF as a strictly matched control.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_COMMIT = "87a0a200ca8212bda20c48e325d95936dfe17630"
REF_CONFIG = "93c90262592a15962553a7f4ffe7f9cf9545679d8dbc702c614fae9d97842676"
REF_PRETRAINED = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
REF_CONTENT = "de46f406f91273301411b79edf6ba0512cfbb9f2b23bf6c470136f1013ed79a4"
REF_MANIFEST = "13b6c1fa3e06243fcff7ba70ccc42ab0d8db7756b5352d3b08bb308f2061c8ce"
REF_SEMANTIC = "5cc32060a550f1dfac9dc322528c05242cf3b1b4be3188ece58c9671d2eb2192"
REF_GRAPH = "598308dd8d0f2eeca9b9180447aa486a8737abe8465969a7296747e8ae52fcb5"
REF_TRAIN = {
    42: "fbac0802d86660e1b74c7b5ec89fb419fd5e119b054d3a2f67449ac01864a59b",
    43: "f9d45d16474aac522ced0842bc4bc3dc84224ad567b2663cc232ce4f4d1db68a",
    44: "b3b2e5f38f181c17398d20713358eba5bf62699819ec82f329ef5b6a32cda6f6",
}
SOURCE_FILES = (
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
    "models_related/ultralytics/ultralytics/utils/loss.py",
    "models_related/ultralytics/ultralytics/utils/tal.py",
    "models_related/ultralytics/ultralytics/engine/trainer.py",
    "models_related/ultralytics/ultralytics/nn/modules/head.py",
    "models_related/ultralytics/ultralytics/nn/tasks.py",
    "models_related/ultralytics/ultralytics/cfg/default.yaml",
    "train_levir_scripts/train_all_levir_yolov8n_p2_ftsc_head_pruning.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py",
)
REF_CASE = "LEVIR-Ship/historical_reconstructions/RECON_H2_V1"
REF_HF_REV = "81edf3de222a65d3bb825036e61c1def9c5684ea"
REPO = "TLMHoang/FTSC_results"


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def historical_code(root, name):
    for candidate in (root, root.parent):
        try:
            p = subprocess.run(["git", "show", f"{REF_COMMIT}:{name}"], cwd=candidate,
                               capture_output=True, check=True, timeout=30)
            return p.stdout, str(candidate)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return None, None


def source_identity(root):
    rows = {}
    for name in SOURCE_FILES:
        local = root / name
        previous, git_root = historical_code(root, name)
        rows[name] = {
            "local_sha256": sha_file(local) if local.is_file() else None,
            "reconstruction_commit_sha256": sha_bytes(previous) if previous is not None else None,
            "historical_git_root": git_root,
        }
        rows[name]["matches_reconstruction_commit"] = (
            rows[name]["local_sha256"] is not None and previous is not None
            and rows[name]["local_sha256"] == rows[name]["reconstruction_commit_sha256"]
        )
    return rows


def routing_function_identity(root):
    """Compare training-relevant routing functions rather than full file byte hash.

    A changed validator is not an accuracy intervention once the entire dataset
    and split manifest byte hashes are confirmed identical. Changes to seeding,
    source loading or split generation must still block baseline-only reuse.
    """
    name = "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py"
    old, _ = historical_code(root, name)
    now = root / name
    if old is None or not now.is_file():
        return {"error": "Historical routing source unavailable", "checks": {}}
    def functions(content):
        module = ast.parse(content)
        return {node.name: ast.dump(node, include_attributes=False)
                for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    before, after = functions(old), functions(now.read_bytes())
    tracked = ("local_ultralytics", "seed_everything", "create_fixed_split",
               "prepare_fixed_split", "validate_split")
    checks = {key: key in before and key in after and before[key] == after[key]
              for key in tracked}
    return {"checks": checks, "identical_functions": sorted(k for k,v in checks.items() if v),
            "different_functions": sorted(k for k,v in checks.items() if not v),
            "note": "Different validation/prepare functions can be tolerated ONLY if "
                    "frozen split manifest plus full image/label content hashes match; "
                    "training seed/source loading/split creation must remain identical."}


def pretrained_candidates(root, explicit=None):
    """Check known local locations; never download or substitute a different .pt."""
    paths = ([Path(explicit)] if explicit else []) + [
        root / "yolov8n.pt", root / "weights/yolov8n.pt",
        root.parent / "yolov8n.pt", root.parent / "weights/yolov8n.pt",
        Path("/marimo/yolov8n.pt"),
    ]
    results, seen = [], set()
    for candidate in paths:
        absolute = candidate.expanduser().resolve()
        if str(absolute) in seen:
            continue
        seen.add(str(absolute))
        exists = absolute.is_file()
        digest = sha_file(absolute) if exists else None
        results.append({"path": str(absolute), "exists": exists,
                        "sha256": digest, "matches": digest == REF_PRETRAINED})
    return results


def remote_contracts():
    from huggingface_hub import hf_hub_download
    result = {}
    for seed in (42, 43, 44):
        base = f"{REF_CASE}/seed_{seed}"
        entries = {}
        for name in ("run_contract.json", "pretrained_transfer.json", "training_attempt.json"):
            path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                        filename=f"{base}/{name}", revision=REF_HF_REV))
            entries[name] = json.loads(path.read_text())
        result[str(seed)] = entries
    return result


def check_report(root, pretrained_path=None):
    from utils.ftsc_paper_run_integrity import _split_identity
    from utils.ftsc_paper_preflight import load_yaml, layers
    from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as h
    from train_levir_scripts import train_levir_ftsc_head_factorial_v1 as fac
    config = fac.CASES["P23_FTSC"]
    control = fac.CASES["P23_NOFTSC"]
    data = root/"datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    pretrained_sources = pretrained_candidates(root, pretrained_path)
    remote = remote_contracts()
    checks = {}
    checks["ftsc_config_sha256"] = config.is_file() and sha_file(config) == REF_CONFIG
    checks["control_no_ftsc_same_graph"] = (control.is_file() and config.is_file()
        and fac.graph(load_yaml(config)) == fac.graph(load_yaml(control))
        and load_yaml(control).get("ftsc", {}).get("enabled") is False)
    checks["ftsc_graph_sha256"] = (config.is_file() and
        sha_bytes(json.dumps(layers(load_yaml(config)), sort_keys=True,
                             separators=(",", ":")).encode()) == REF_GRAPH)
    checks["pretrained_bytes_match"] = any(p["matches"] for p in pretrained_sources)
    if data.is_file():
        split = _split_identity("LEVIR-Ship", data)
    else:
        split = {"error": "Missing frozen dataset YAML"}
    checks["dataset_full_contents_match"] = split.get("content_inventory_sha256") == REF_CONTENT
    checks["dataset_split_manifest_match"] = split.get("split_manifest_sha256") == REF_MANIFEST
    checks["dataset_yaml_semantics_match"] = split.get("data_yaml_semantic_sha256") == REF_SEMANTIC
    checks["dataset_file_count_match"] = split.get("content_file_count") == 7792
    args = argparse.Namespace(epochs=100, imgsz=512, batch_size=8, patience=20,
                              device="cuda:0", workers=4)
    # Compare the scientific recipe generated by the actual current runner with
    # the remotely preserved reconstruction contract, not only args.yaml.
    recipe = {}
    for seed in (42, 43, 44):
        run = h.train_kwargs(args, data, seed, True)
        from utils.ftsc_paper_run_integrity import EXECUTION_ONLY_KEYS
        scientific = {k: v for k, v in run.items() if k not in EXECUTION_ONLY_KEYS}
        recipe[str(seed)] = scientific
        checks[f"recon_contract_training_seed_{seed}"] = (
            remote[str(seed)]["run_contract.json"]["training"] == scientific
            and remote[str(seed)]["run_contract.json"]["training_sha256"] == REF_TRAIN[seed]
        )
        checks[f"recon_contract_pretrained_seed_{seed}"] = (
            remote[str(seed)]["run_contract.json"]["pretrained_sha256"] == REF_PRETRAINED
            and remote[str(seed)]["pretrained_transfer.json"]["source_sha256"] == REF_PRETRAINED
            and remote[str(seed)]["pretrained_transfer.json"]["status"] == "PASS"
        )
        checks[f"recon_contract_amp_seed_{seed}"] = (
            remote[str(seed)]["training_attempt.json"]["effective_amp"] is True
            and remote[str(seed)]["training_attempt.json"]["environment"].get("git_commit")
            == REF_COMMIT
        )
    sources = source_identity(root)
    routing = routing_function_identity(root)
    critical_sources = {k: v for k, v in sources.items() if not k.endswith('/train_all_levir_yolov8n_p2_routing.py')}
    checks["core_training_source_bytes_match_recon_commit"] = all(
        v["matches_reconstruction_commit"] for v in critical_sources.values()
    )
    # If the split bytes are identical, only non-training validation and
    # prepared-dataset path checking changes may be accepted in routing.py.
    # No relaxation applies to random seeding, loader selection or split creation.
    essential_routing = ("seed_everything", "local_ultralytics", "create_fixed_split")
    checks["routing_training_functions_match_recon_commit"] = (
        all(routing.get("checks", {}).get(k) is True for k in essential_routing)
        and all(checks[k] for k in ("dataset_full_contents_match",
                                      "dataset_split_manifest_match",
                                      "dataset_yaml_semantics_match"))
    )
    checks["scientific_source_bytes_match_recon_commit"] = (
        checks["core_training_source_bytes_match_recon_commit"]
        and checks["routing_training_functions_match_recon_commit"]
    )
    # Control graph, scientific recipe and full source must pass before claiming
    # a future baseline-only training as a strictly matched ablation.
    blockers = sorted(k for k, value in checks.items() if not value)
    return {
        "schema": "levir_h2_reuse_gate_v1",
        "remote_revision": REF_HF_REV,
        "reconstruction_commit": REF_COMMIT,
        "checks": checks,
        "blockers": blockers,
        "status": "READY_FOR_MATCHED_CONTROL_ONLY" if not blockers else "BLOCKED_FOR_STRICT_MATCH",
        "source_revision_comparison": sources,
        "routing_function_comparison": routing,
        "pretrained_candidates": pretrained_sources,
        "selected_pretrained": next((x["path"] for x in pretrained_sources if x["matches"]), None),
        "dataset_identity": split,
        "training_recipes": recipe,
        "action": "Baseline-only training is strictly matched only on READY. "
                  "On BLOCKED, inspect differences or pin archived reconstruction source. "
                  "Do not splice new baseline into historical table."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=ROOT/"runs/levir_h2_reuse_gate_v1/gate.json")
    p.add_argument("--pretrained", type=Path, default=None,
                   help="Optional path to the exact original yolov8n.pt; SHA256 is enforced")
    args = p.parse_args()
    result = check_report(ROOT, pretrained_path=args.pretrained)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        old = json.loads(args.report.read_text())
        if old != result:
            raise RuntimeError(f"Existing gate differs; write to a NEW --report: {args.report}")
    else:
        with args.report.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({"status": result["status"], "blockers": result["blockers"],
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
