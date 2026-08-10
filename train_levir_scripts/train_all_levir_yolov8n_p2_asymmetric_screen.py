#!/usr/bin/env python3
"""Train and evaluate the seed-42 LEVIR P2-only asymmetric-head screen."""

from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow


ROOT = Path(__file__).resolve().parent
workflow.EXPERIMENT = "levir_yolov8n_p2_asymmetric_screen"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-asymmetric-screen-seed42"
workflow.VARIANTS = {
    "plain_p2_only": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml",
    "cls_context_mid_cbam": ROOT.parent
    / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cls_context_mid_cbam.yaml",
    "reg_local": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_reg_local.yaml",
}


def parse_args(argv=None):
    args = workflow.parse_args(argv)
    args.runner = Path(__file__)
    args.seeds = [42]
    args.no_upload = True
    return args


def evaluate(run_dir, data_yaml, args):
    metrics = workflow.evaluate(run_dir, data_yaml, args)
    if "model/params" not in metrics:
        workflow.local_ultralytics()
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops

        model = YOLO(run_dir / "weights/best.pt").model
        metrics.update(
            {"model/params": sum(parameter.numel() for parameter in model.parameters()), "model/GFLOPs": get_flops(model, args.imgsz)}
        )
        (run_dir / "evaluation_metrics.json").write_text(
            workflow.json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metrics


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for variant in args.variants:
        run_dir = workflow.train(variant, 42, data_yaml, amp[variant], args)
        evaluate(run_dir, data_yaml, args)
        workflow.write_summaries(args)


if __name__ == "__main__":
    main()
