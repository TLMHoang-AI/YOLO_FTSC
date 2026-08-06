#!/usr/bin/env python3
"""Train the clean YOLOv8n P2 TAL-positive box-consensus causal test."""

from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
VARIANT = "p2_consensus_g01"
workflow.EXPERIMENT = "levir_yolov8n_p2_consensus"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-consensus-seed42"
workflow.VARIANTS = {
    VARIANT: ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_consensus.yaml"
}
_baseline_train_kwargs = workflow.train_kwargs


def consensus_train_kwargs(args, data_yaml, seed, amp):
    kwargs = _baseline_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        loc_assign=False,
        box_consensus_gain=0.1,
        box_consensus_warmup_start=5,
        box_consensus_warmup_end=15,
        box_consensus_log_grad_ratio=True,
    )
    return kwargs


workflow.train_kwargs = consensus_train_kwargs


if __name__ == "__main__":
    workflow.main()
