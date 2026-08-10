#!/usr/bin/env python3
"""Train a matched bare-KVCA classification-only P2 placement control."""

from pathlib import Path

import train_all_levir_yolov8n_p2_kvca_sr_ablation as runner

runner.VARIANTS = {
    "kvca_block_clsonly": runner.CONFIG_ROOT / "yolov8n_p2_fpn_only_kvca_block_clsonly.yaml",
}
runner.VARIANT_META = {
    "kvca_block_clsonly": {
        "num_heads": 4,
        "sr_ratio": 8,
        "mode": "group_weight",
        "placement": "classification_only",
    },
}
runner.RUNNER_PATH = Path(__file__)

if __name__ == "__main__":
    runner.main()
