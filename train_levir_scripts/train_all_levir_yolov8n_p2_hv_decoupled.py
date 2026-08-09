#!/usr/bin/env python3
"""Train and evaluate the seed-42 LEVIR P2-only HV-decoupled DFL head."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
_WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "_hv_decoupled_workflow", ROOT / "train_all_levir_yolov8n_p2_routing.py"
)
workflow = importlib.util.module_from_spec(_WORKFLOW_SPEC)
sys.modules[_WORKFLOW_SPEC.name] = workflow
_WORKFLOW_SPEC.loader.exec_module(workflow)
workflow.EXPERIMENT = "levir_yolov8n_p2_hv_decoupled"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-hv-decoupled-seed42"
workflow.VARIANTS = {
    "hv_decoupled": ROOT.parent
    / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_hv_decoupled.yaml"
}


def parse_args(argv=None):
    args = workflow.parse_args(argv)
    args.runner = Path(__file__)
    args.seeds = [42]
    return args


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for variant in args.variants:
        run_dir = workflow.train(variant, 42, data_yaml, amp[variant], args)
        workflow.evaluate(run_dir, data_yaml, args)
        workflow.write_summaries(args)
        if uploader:
            uploader.upload_run(run_dir, variant, 42)
            uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
