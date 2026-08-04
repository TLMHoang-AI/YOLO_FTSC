# GCTS v2 diagnostic record

## Fixed v1 baseline

| Model | AP50 | AP75 | Selector quadrant accuracy | Selector x-bias | Global gamma |
|---|---:|---:|---:|---:|---:|
| YOLOv10n | 69.55 | 13.53 | - | - | - |
| GCTS v1 onehot-w01 | 74.29 | 12.41 | 66.8% | +0.0665 P3 cell | -0.075 |

The inference splice on the v1 checkpoint produced AP50/AP75 of `73.73/11.73` with `gamma=0`, `74.02/11.56` for classification-only, `74.04/12.58` for regression-only, and `74.29/12.41` for the full branch. The selected content is useful, but feeding the same appearance correction into regression does not restore independent-baseline AP75.

## Reproduction

`diagnose_gcts_v2.py` verifies pixel-unshuffle order, evaluates AP50/AP75, performs Hungarian GT/prediction matching, and writes matched IoU, signed P3-cell center offsets, width/height ratios, `<20 px` and `>=20 px` buckets, selector coordinate bias/MAE/entropy, and tiny/large/background gate means.

```bash
python diagnose_gcts_v2.py \
  --weights runs/levir_gcts_v2/v2_e05/weights/best.pt \
  --images datasets/levir_gcts_seed42/images/test \
  --labels datasets/levir_gcts_seed42/labels/test \
  --data datasets/levir_gcts_seed42/levir_ship.yaml \
  --output runs/levir_gcts_v2/v2_e05/diagnostics.json
```

Run the same command for `v2_e10` and `v2_e05_nogate`. Keep `imgsz=512`, `conf=0.001`, the seed-42 split, and the same test set for all comparisons.

## Acceptance criteria

- AP50 is at least `69.55` and AP75 is at least `13.53`.
- IoU>=0.75 for matched objects with diagonal `>=20 px` falls by no more than one percentage point.
- The `<20 px` bucket improves mean matched IoU or its IoU>=0.75 rate.
- Selector x-bias clearly falls below `0.0665` P3 cell and coordinate MAE below approximately `0.19` cell.

This file records the frozen comparison protocol. Run-specific measurements belong in each generated `diagnostics.json` and the matrix `summary.csv`.
