"""LEVIR H2 matched-control-only training; RECON_H2_V1 FTSC is immutable HF reference.

This is NOT a new four-cell factorial. Training is strictly limited to P23_NOFTSC
seeds 42/43/44, with strict provenance check against RECON_H2_V1 first.
The defaults perform a preflight ONLY. No automatic upload.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from train_levir_scripts import train_levir_ftsc_head_factorial_v1 as fac
from utils import ftsc_levir_h2_reuse_gate_v1 as gate
from utils.ftsc_paper_run_integrity import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/levir_h2_matched_noftsc_control_v1"
CASE = "P23_NOFTSC"
SEEDS = (42, 43, 44)
REMOTE_PREFIX = "LEVIR-Ship/historical_reconstructions/RECON_H2_V1"
METRICS = fac.METRICS


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def write_once(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        require(prior == data, f"Immutable output differs; choose new namespace: {path}")
    else:
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")


def preflight(args):
    require(args.project.resolve() != fac.RUNS.resolve(), "Never write into original factorial run directory")
    require(not fac.RUNS.resolve().is_relative_to(args.project.resolve()),
            "Project cannot be ancestor of archived factorial runs")
    require(args.split_seed == 42 and args.epochs == 100 and args.imgsz == 512
            and args.batch_size == 8 and args.patience == 20, "Historical LEVIR recipe changed")
    require(tuple(args.cases) == (CASE,), "Only P23_NOFTSC may train in this suite")
    require(set(args.seeds) <= set(SEEDS) and len(set(args.seeds)) == len(args.seeds),
            "Invalid seed selection")
    require(args.workers == 4, "Match reconstruction worker count (4) for maximal control")
    require(args.pretrained.is_file() and sha256(args.pretrained) == gate.REF_PRETRAINED,
            "Local yolov8n.pt missing or its SHA differs from reconstructed training")
    report = gate.check_report(ROOT, pretrained_path=args.pretrained)
    require(report["status"] == "READY_FOR_MATCHED_CONTROL_ONLY",
            f"Historical reuse gate is blocked: {report['blockers']}")
    require(Path(report["selected_pretrained"]).resolve() == args.pretrained.resolve(),
            "Selected pretrained source differs from requested training file")
    source = fac.source_preflight()
    require(fac.graph(fac.load(fac.CASES["P23_FTSC"])) ==
            fac.graph(fac.load(fac.CASES[CASE])), "FTSC vs control graph differs")
    require(fac.load(fac.CASES[CASE]).get("ftsc", {}).get("enabled") is False,
            "Control FTSC enabled unexpectedly")
    data = fac.dataset(args)
    require(report["dataset_identity"]["content_inventory_sha256"] == gate.REF_CONTENT,
            "Dataset source no longer matches reconstruction")
    # Guard against accidental writes onto older completed experiment namespaces.
    plan = {
        "schema": "levir_h2_matched_noftsc_control_v1",
        "action": "CONTROL_ONLY",
        "cases": [CASE],
        "seeds": list(SEEDS),
        "reused_treatment": {"cohort": "RECON_H2_V1",
                             "repo": gate.REPO, "revision": gate.REF_HF_REV,
                             "checkpoint_roots": [f"{REMOTE_PREFIX}/seed_{s}/weights/best.pt" for s in SEEDS]},
        "output_root": str(args.project.resolve()),
        "pretrained_sha256": sha256(args.pretrained),
        "data_identity": report["dataset_identity"],
        "reconstruction_git_commit": gate.REF_COMMIT,
        "source_static_sha": source["source_sha256"],
        "control_yaml_sha256": sha256(fac.CASES[CASE]),
        "ftsc_yaml_sha256": gate.REF_CONFIG,
        "scientific_recipe": {str(s): report["training_recipes"][str(s)] for s in SEEDS},
        "no_training_in_preflight": True,
        "no_source_overwrite": True,
    }
    return report, source, data, plan


def reconstructed_scores():
    from huggingface_hub import hf_hub_download
    result = {}
    for seed in SEEDS:
        file = Path(hf_hub_download(
            repo_id=gate.REPO, repo_type="dataset",
            filename=f"{REMOTE_PREFIX}/seed_{seed}/evaluation_metrics_extended.json",
            revision=gate.REF_HF_REV,
        ))
        obj = json.loads(file.read_text(encoding="utf-8"))
        vals = {metric: float(obj[key]) for metric, key in METRICS.items()}
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in vals.values()),
                f"Invalid reconstruction reference metric seed {seed}")
        result[seed] = vals
    return result


def aggregate(args, data, source):
    from evaluate_test.standard_detection_metrics import load_merged_metrics
    ftsc = reconstructed_scores()
    base, paired = {}, []
    for seed in SEEDS:
        run = args.project / CASE / f"seed_{seed}"
        require(fac.current_state(run, fac.contract(args, CASE, seed, data, source)) == "COMPLETE",
                f"Cannot aggregate incomplete baseline: {run}")
        obj = load_merged_metrics(run)
        base[seed] = {metric: float(obj[key]) for metric, key in METRICS.items()}
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in base[seed].values()),
                f"Invalid baseline result seed {seed}")
        for metric in METRICS:
            paired.append({"seed": seed, "metric": metric,
                           "recon_ftsc_pct": ftsc[seed][metric] * 100,
                           "new_noftsc_pct": base[seed][metric] * 100,
                           "delta_ftsc_minus_control_pp": (ftsc[seed][metric] - base[seed][metric]) * 100})
    summary = []
    for metric in METRICS:
        diffs = [next(x["delta_ftsc_minus_control_pp"] for x in paired
                      if x["seed"] == seed and x["metric"] == metric) for seed in SEEDS]
        summary.append({"metric": metric, "n": len(SEEDS),
                        "recon_ftsc_mean_pct": 100 * statistics.fmean(ftsc[s][metric] for s in SEEDS),
                        "new_control_mean_pct": 100 * statistics.fmean(base[s][metric] for s in SEEDS),
                        "paired_mean_delta_pp": statistics.fmean(diffs),
                        "paired_sample_std_delta_pp": statistics.stdev(diffs)})
    # Cross-generation comparison stays explicitly labelled, never masquerades as same cohort.
    result = {
        "status": "AGGREGATED_HISTORICAL_RECON_TREATMENT_VS_NEW_CONTROL",
        "provenance": "RECOVERED_RECON_H2_TREATMENT + NEW_MATCHED_CONTRACT_CONTROL; NOT 2x2 FACTORIAL",
        "seeds": list(SEEDS),
        "reference_commit": gate.REF_COMMIT,
        "hf_revision": gate.REF_HF_REV,
        "comparison": paired,
        "summary": summary,
        "warning": "Both arms run same scientific config, matching source and data hashes; "
                   "they are distinct experiment generations, not common-run stochastic pair. "
                   "Report this transparently and compare against existing contemporaneous factorial."
    }
    write_once(args.project / "recon_vs_new_control_summary.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cases", nargs="+", default=[CASE])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--pretrained", type=Path, default=ROOT/"yolov8n.pt")
    parser.add_argument("--project", type=Path, default=OUT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT/"datasets")
    parser.add_argument("--data-root", type=Path, default=ROOT/"LevirShipData")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(not (args.confirm_run and args.aggregate), "Choose train or aggregate, not both")
    require(not args.resume or args.confirm_run, "--resume requires --confirm-run")
    args.project = args.project.expanduser().resolve()
    args.pretrained = args.pretrained.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    _, source, data, plan = preflight(args)
    write_once(args.project / "control_only_plan.json", plan)
    if args.aggregate:
        require(set(args.seeds) == set(SEEDS), "Aggregate requires seeds 42/43/44")
        result = aggregate(args, data, source)
    elif args.confirm_run:
        # Invoke existing hardened factorial trainer but restrict to this one case,
        # a fresh isolated project and matching reconstruction workers.
        fac.model_preflight(args.pretrained)
        for seed in args.seeds:
            fac.train_and_evaluate(args, CASE, seed, data, source)
        result = {"status": "CONTROL_ONLY_RUNS_PROCESSED",
                  "seeds": args.seeds, "case": CASE,
                  "project": str(args.project), "no_ftsc_retraining": True}
    else:
        result = {"status": "CONTROL_ONLY_PREFLIGHT_PASSED_NO_TRAINING",
                  "plan": str(args.project/"control_only_plan.json"),
                  "cases": args.cases, "seeds": args.seeds}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
