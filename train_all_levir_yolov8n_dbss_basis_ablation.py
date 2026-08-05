#!/usr/bin/env python3
"""Run the YOLOv8n LEVIR DBSS-P2-Aware basis-count ablation."""

from misc.train_levir_dbss_hit import main


BASIS_VARIANTS = ("k4", "k12", "k16", "k20")
EXPERIMENT_SLUG = "levir_yolov8n_dbss_basis_ablation"


if __name__ == "__main__":
    main(
        default_mechanisms=BASIS_VARIANTS,
        default_project=EXPERIMENT_SLUG,
        default_hf_repo_name="levir-yolov8n-dbss-basis-ablation",
        default_seeds=(42,),
    )
