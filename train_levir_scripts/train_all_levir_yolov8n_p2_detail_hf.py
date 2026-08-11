#!/usr/bin/env python3
"""Train/evaluate the LEVIR YOLOv8n random-HF and masked-P2-detail variants."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_detail_hf"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-detail-hf"
workflow.VARIANTS = {
    "hf_atten_aug": CONFIG_ROOT / "yolov8n_p2_levir_hf_atten_aug.yaml",
    "p2_detail_recon": CONFIG_ROOT / "yolov8n_p2_levir_masked_detail_recon.yaml",
}

_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for
_base_smoke = workflow.smoke
_base_train = workflow.train


def train_kwargs(args, data_yaml, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "_active_variant", "")
    if variant == "hf_atten_aug":
        kwargs.update(
            hf_atten_prob=0.5,
            hf_atten_min_alpha=0.0,
            hf_atten_max_alpha=0.75,
            hf_atten_blur_kernel=5,
            hf_atten_mask_grid=16,
        )
    elif variant == "p2_detail_recon":
        kwargs["p2_detail_rec_gain"] = 0.1
    return kwargs


def model_for(variant: str, pretrained: str):
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import Detect, MaskedP2DetailReconstruction

    layers = model.model.model
    head = layers[-1]
    if not isinstance(head, Detect) or head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"{variant}: expected P2-P5 Detect strides, got {type(head).__name__} {head.stride}")
    if variant == "hf_atten_aug" and head.f != [19, 22, 25, 28]:
        raise ValueError(f"{variant}: unexpected Detect inputs {head.f}")
    if variant == "p2_detail_recon":
        if not isinstance(layers[20], MaskedP2DetailReconstruction) or head.f != [20, 23, 26, 29]:
            raise ValueError(f"{variant}: masked detail layer/Detect inputs did not resolve as specified")
    return model


def smoke(variant: str, data_yaml: Path, args, amp: bool = True) -> bool:
    args._active_variant = variant
    return _base_smoke(variant, data_yaml, args, amp)


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args) -> Path:
    args._active_variant = variant
    return _base_train(variant, seed, data_yaml, amp, args)


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.smoke = smoke
    workflow.train = train
    args = workflow.parse_args()
    args.runner = Path(__file__)
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
