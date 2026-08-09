#!/usr/bin/env python3
"""Train seed-42 shared-CBAM P2 with TAL-positive confidence rescue."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_cbam_shared_confidence_rescue_workflow", ROOT / "train_all_levir_yolov8n_p2_routing.py"
)
workflow = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = workflow
_SPEC.loader.exec_module(workflow)
workflow.EXPERIMENT = "levir_yolov8n_p2_cbam_shared_confidence_rescue"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-cbam-shared-confidence-rescue-seed42"
workflow.VARIANTS = {
    "cbam_shared_confidence_rescue": ROOT.parent
    / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_block.yaml"
}
_base_train_kwargs = workflow.train_kwargs
_base_parse_args = workflow.parse_args


def train_kwargs(args, data_yaml, seed, amp):
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        positive_confidence_rescue_gain=0.25,
        positive_confidence_rescue_gamma=1.0,
        cls_iou_target=False,
        vfl=False,
        loc_assign=False,
        box_consensus_gain=0.0,
        positive_support_dropout=False,
    )
    return kwargs


workflow.train_kwargs = train_kwargs


def parse_args(argv=None):
    args = _base_parse_args(argv)
    args.runner = Path(__file__)
    args.seeds = [42]
    return args


def main():
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
