# GCTS approach report

## Goal

Ground-Truth Constrained Token Selection (GCTS) preserves the high-resolution P2 token that contains a tiny object before the P2-to-P3 stride-2 operation mixes it with neighboring background tokens. It does not estimate energy, reconstruction error, residual hardness, or background objectness.

## Architecture

The original YOLOv10n P2-to-P3 `Conv` remains the base path. `GCTS` subclasses that `Conv`, so its pretrained parameter names (`conv`, `bn`, and `act`) and its layer index remain unchanged.

For projected P2 features, `pixel_unshuffle(2)` exposes candidates in TL, TR, BL, BR order. A `1x1` selector predicts four routing weights:

```text
alpha = softmax(selector(concat(T0, T1, T2, T3)))
D = sum_k alpha_k * T_k
P3_out = P3_base + gamma * output_projection(D)
```

`gamma` is a learned scalar initialized to zero. A newly constructed GCTS model therefore starts with exactly the base convolution output while the auxiliary loss can immediately train the selector.

## Ground-truth constraint

Only annotated centers contribute to the selection loss. Background P3 cells are ignored.

- `onehot`: the target is the P2 quadrant containing the center.
- `bilinear`: the fractional center coordinates within its P3 cell interpolate the four quadrant targets.

```text
L_select = -loss_weight * mean_i(sum_k q_ik * log(alpha_ik))
```

Multiple objects in the same P3 cell remain separate loss samples. An image or batch without boxes contributes a differentiable zero. The existing model auxiliary-loss hook adds `L_select` to training and exposes `loss_gcts_select` through `model.mechanism_metrics`.

## Files changed

- `models_related/ultralytics/ultralytics/nn/modules/block.py`: GCTS implementation and target/loss construction.
- `models_related/ultralytics/ultralytics/nn/modules/__init__.py`: public module export.
- `models_related/ultralytics/ultralytics/nn/tasks.py`: YAML parser registration and auxiliary-loss integration.
- `models_related/ultralytics/tests/test_levir_mechanisms.py`: routing, identity, target, empty-GT, and gradient tests.
- `models_related/ultralytics/ultralytics/cfg/models/v10/yolov10n-gcts-*.yaml`: four controlled ablation configurations.
- `diagnose_gcts_v2.py` and `gcts_v2_diagnostic_report.md`: reproducible localization and routing diagnostics for v2.

## Ablation configurations

| Config | Target | Loss weight |
|---|---|---:|
| `yolov10n-gcts-bilinear-w01.yaml` | bilinear | 0.1 |
| `yolov10n-gcts-bilinear-w02.yaml` | bilinear | 0.2 |
| `yolov10n-gcts-onehot-w01.yaml` | one-hot | 0.1 |
| `yolov10n-gcts-onehot-w02.yaml` | one-hot | 0.2 |

All other architecture and training settings are identical to YOLOv10n.

## Pretrained initialization

Use the custom YAML and partially load the standard checkpoint. Because GCTS replaces layer 3 in place and inherits `Conv`, the original P2-to-P3 convolution keys remain compatible. Only the new detail projection, selector, output projection, and `gamma` lack pretrained values.

```bash
python models_related/train_eval/train.py \
  --model-yaml models_related/ultralytics/ultralytics/cfg/models/v10/yolov10n-gcts-bilinear-w01.yaml \
  --pretrained yolov10n.pt \
  --data-yaml /path/to/data.yaml \
  --epochs 100 \
  --name yolov10n_gcts_bilinear_w01
```

Evaluate the resulting checkpoint with the existing evaluation entry point:

```bash
python models_related/train_eval/eval.py \
  --weights runs/detect/yolov10n_gcts_bilinear_w01/weights/best.pt \
  --data-yaml /path/to/data.yaml
```

Run the four configurations with the same seed, dataset split, image size, augmentation, and training hyperparameters. Compare AP50, AP75, tiny-object recall, parameter count, FLOPs, and latency; AP75 is the primary localization-sensitive metric.

For the controlled LEVIR-Ship matrix (seed 42), use `train_gcts_levir.py`. It trains the four configurations sequentially, resumes partial runs, evaluates validation and test splits, writes `runs/levir_gcts/summary.csv`, and uploads each completed run to the configured Hugging Face dataset.

## GCTS v2: head-local appearance and geometry

The first matrix improved classification-sensitive AP50 but not strict localization: baseline AP50/AP75 was `69.55/13.53`, while the best v1 checkpoint reached `74.29/12.41`. Its selector was correct in `66.8%` of quadrants, had an x-bias of `+0.0665` P3 cells, and learned a global `gamma=-0.075`. GCTS v2 therefore leaves the backbone and neck unchanged and adds a `v10GCTSDetect` adapter at the detection head.

The head consumes `[P2, P3, P4, P5]`. P2 candidates retain TL/TR/BL/BR identity. Classification receives position-preserving selected appearance:

```text
Cpos = concat(alpha0*T0, alpha1*T1, alpha2*T2, alpha3*T3)
Fcls = P3 + Proj_cls0(Cpos)
```

Regression never receives collapsed appearance. It receives only selector geometry:

```text
x_hat = alpha1 + alpha3
y_hat = alpha2 + alpha3
P = concat(alpha0..alpha3, x_hat-0.5, y_hat-0.5, normalized_entropy(alpha))
Freg = P3 + g * epsilon * tanh(Proj_pos0(P))
```

Both projections are zero-initialized, so a new v2 model is exactly the standard YOLOv10 head at initialization. There is no scalar `gamma`. P4/P5 are untouched, while both YOLOv10 one-to-many and detached one-to-one heads use the same routed P3 features.

The bilinear target has expected coordinates equal to the fractional GT center. At GT cells, v2 optimizes `0.1 * (KL(q || alpha) + SmoothL1((x_hat,y_hat), (xfrac,yfrac)))`. The optional gate uses bbox diagonal after augmentation: `<20 px` is positive, `20-24 px` is ignored, and `>24 px` is negative. Tiny wins cell collisions; an equal number of deterministic background cells are sampled as negatives. The no-gate variant fixes `g=1` and omits gate loss.

| Config | epsilon | Tiny gate |
|---|---:|---:|
| `yolov10n-gcts-v2-e05.yaml` | 0.05 | yes |
| `yolov10n-gcts-v2-e10.yaml` | 0.10 | yes |
| `yolov10n-gcts-v2-e05-nogate.yaml` | 0.05 | no |

Standard `v10Detect` state keys (`cv2`, `cv3`, and one-to-one copies) are retained and load from `yolov10n.pt`; only the P2 adapter is new. Run the seed-42 matrix with:

```bash
python train_gcts_levir.py --matrix v2 --seed 42 --pretrained yolov10n.pt
```

The default v2 output directory is `runs/levir_gcts_v2`, and completed runs upload to `<HF user>/levir-yolov10n-gcts-v2-ablation`. Selection requires AP50 >= `69.55`, AP75 >= `13.53`, no more than one percentage-point loss in IoU>=0.75 for objects at least 20 px, improved tiny localization, and reduced selector x-bias/coordinate MAE.

## P3 non-uniform DFL

The controlled P3 codebook experiment retains 16 logits per box side but replaces the P3 projection values with `[0, .25, .5, .75, 1, 1.25, 1.5, 2, 3, 4, 5, 7, 9, 11, 13, 15]`. Targets interpolate between the two surrounding physical values and decoding uses their expected value. P4/P5 retain uniform `0..15` bins, and standard YAMLs do not enter the new path.

Two seed-42 configs isolate the codebook effect: `yolov10n-p3-nudfl.yaml` and `yolov10n-gcts-v2-e05-p3-nudfl.yaml`. Both preserve the standard head parameter shapes and load `yolov10n.pt`; the codebook is a fixed buffer. Run them with `python train_gcts_levir.py --matrix p3_nudfl`. Each fresh run records pretrained validation in `epoch0_metrics.json` before training.

The YOLOv8 control matrix uses the same codebook and loss/decode path through `P3NUDFLDetect`, while retaining standard one-to-many training and NMS inference. `yolov8n-levir-baseline.yaml` and `yolov8n-p3-nudfl.yaml` load `yolov8n.pt` and run with `python train_gcts_levir.py --matrix yolov8_p3_nudfl --pretrained yolov8n.pt`. This comparison separates the codebook effect from YOLOv10's one-to-one end-to-end head.

## Experiment status and verified results

All numbers below are post-training evaluations on the fixed LEVIR-Ship split unless explicitly marked as partial. AP values are shown as percentages. A run configured for 100 or 200 epochs may finish earlier only when its configured early-stopping patience permits it.

### GCTS v1 multi-seed matrix

Seed 42 used 100 epochs with patience 20. Seed 43 and seed 44 use the later 200-epoch protocol with patience 200, so their training budget is not directly identical to seed 42. Do not calculate a final three-seed mean until seed 44 has completed all four variants and produced test evaluations.

| Seed | Variant | Test AP50 | Test mAP50-95 | Test AP75 | Status |
|---:|---|---:|---:|---:|---|
| 42 | `bilinear_w01` | 70.17 | 27.33 | not recorded | complete |
| 42 | `bilinear_w02` | 68.75 | 26.42 | not recorded | complete |
| 42 | `onehot_w01` | 74.29 | 29.00 | 12.41 (diagnostic evaluation) | complete |
| 42 | `onehot_w02` | 71.33 | 27.62 | not recorded | complete |
| 43 | `bilinear_w01` | 76.46 | 30.15 | 13.88 | complete |
| 43 | `bilinear_w02` | 79.01 | 30.07 | 14.29 | complete |
| 43 | `onehot_w01` | 76.53 | 30.02 | 14.70 | complete |
| 43 | `onehot_w02` | 77.69 | 30.92 | 14.53 | complete |
| 44 | `bilinear_w01` | pending | pending | pending | training; no test result uploaded yet |
| 44 | remaining variants | pending | pending | pending | queued after `bilinear_w01` |

Artifact sources:

- Seed 42: `duyle2408/levir-yolov10n-gcts-ablation`.
- Seed 43: `duyle2408/levir-yolov10n-gcts-v1-seed43`.
- Seed 44: `duyle2408/levir-yolov10n-gcts-v1-seed44`; the repository contained only `.gitattributes` at the last check because the first variant had not completed.

The seed-44 job is a detached process. Notebook/kernel availability must not be used as proof that training is active. Verify activity using two successive `results.csv` observations plus GPU utilization. At the last verified observation, the first variant advanced from epoch 181 to 182 within 20 seconds and the GPU reported 39-43% utilization. These are training-progress observations, not final metrics.

### YOLOv8 P3 non-uniform DFL control

Both seed-42 runs completed and were evaluated on the 788-image test split. The baseline stopped after 77 epochs and NUDFL after 57 epochs because patience was 20.

| Model | Test AP50 | Test AP75 | Test mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| YOLOv8n baseline | 69.92 | 9.07 | 25.56 | 75.00 | 61.65 |
| YOLOv8n P3 NUDFL | 67.87 | 10.42 | 25.21 | 71.46 | 61.35 |
| NUDFL minus baseline | -2.05 | +1.35 | -0.35 | -3.55 | -0.30 |

The non-uniform codebook improved strict AP75 but reduced AP50 and precision. Since the same trade-off appears in YOLOv8, it cannot be attributed solely to YOLOv10's one-to-one head. Artifacts are stored in `duyle2408/levir-yolov8n-p3-nudfl-ablation`.

### Result-handling rules

- `results.csv` contains per-epoch validation metrics, not final test metrics.
- Add a seed to aggregate mean and standard deviation only after `best.pt` has been evaluated on both validation and test splits and `summary.csv` has been uploaded.
- Never treat a live PID, an existing `/proc` entry, or a reachable notebook as sufficient evidence of training progress; require file advancement over time and non-idle GPU activity.
- Preserve seed, split, epoch budget, patience, image size, batch size, and pretrained checkpoint in any aggregate table so unequal protocols remain visible.
