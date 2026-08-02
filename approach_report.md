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
