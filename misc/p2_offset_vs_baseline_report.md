# YOLOv8 P2 Offset Regression vs Baseline

## Setup

- Dataset: LEVIR-Ship, seed 42.
- Models: YOLOv8n with P2/P3/P4/P5 detection levels.
- `p2_offset`: P2 uses `P2OffsetRegression`; P3/P4/P5 remain unchanged.
- `p2_baseline`: P2 exists, but uses the original regression head; no offset sampling.
- Checkpoints:
  - `yolov8n_p2_offset_seed42/weights/best.pt`
  - `yolov8n_p2_baseline_seed42/weights/best.pt`

## Test-set results

Both models were evaluated sequentially on the same 788-image test split.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| P2 offset | 0.7550 | 0.6968 | 0.7407 | 0.2894 |
| P2 baseline | 0.7824 | 0.6681 | 0.7453 | 0.2924 |

## Interpretation

- Baseline is higher by `+0.0046` mAP50 and `+0.0030` mAP50-95.
- Offset has `+0.0287` recall.
- On this seed and test split, the offset head improves recall but does not improve overall mAP.

## Validation reference

The earlier validation run was not used as the final comparison. It produced:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| P2 offset | 0.8372 | 0.6959 | 0.7834 | 0.3215 |
| P2 baseline | 0.8165 | 0.6731 | 0.7654 | 0.3165 |

Final conclusions above are based on the test split.
