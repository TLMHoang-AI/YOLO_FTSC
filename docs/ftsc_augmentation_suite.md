# FTSC H2 augmentation suite

**PREPARED — NOT YET RUN**

All five cases use
`models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml`.
The model topology, FTSC calibrator/evidence, loss, and TAL implementation are
shared and unchanged. Only the training augmentation configuration differs.

| Case | Augmentation | Transform order |
|---|---|---|
| `A0_FTSC` | none of OACP/M5/CP2 | disabled Mosaic, then RandomPerspective |
| `A1_OACP_R2` | corrected single-pass R2 | OACP, disabled Mosaic, RandomPerspective |
| `A2_M5` | hard-negative Mosaic | M5 Mosaic, RandomPerspective |
| `A3_CP2` | two copies of one raw detection object | disabled Mosaic, RandomPerspective, detection CP2 |
| `A4_OACP_R2_M5` | M5 plus corrected single-pass R2 | M5 Mosaic, RandomPerspective, OACP |

The A4 order intentionally follows Duy's `levir_m5_oacp_r2` runner. That
runner enables R2 but does not set `OACP_PLACEMENT`; the patched source pipeline
therefore uses its `post_mosaic` default. A1 explicitly sets
`OACP_PLACEMENT=pre_transform`.

The runner is fail-closed. Without `--confirm-run` it performs a local
configuration/model preflight and exits without training. Before a future M5
run, create a real bank with `tools/mine_mosaic_hard_negatives.py`; the runner
will reject A2/A4 if the specified bank file does not exist.

Future local launch (not executed during preparation):

```bash
/home/htmlai/miniconda3/bin/conda run -n myenv python \
  train_levir_scripts/train_levir_ftsc_augmentation_suite.py \
  --confirm-run \
  --cases A0_FTSC A1_OACP_R2 A2_M5 A3_CP2 A4_OACP_R2_M5 \
  --seeds 42 43 44 \
  --data-root /absolute/path/to/LevirShipData \
  --dataset-root /absolute/path/to/local/datasets \
  --pretrained /absolute/path/to/local/yolov8n.pt \
  --hard-negative-bank /absolute/path/to/hard_negative_bank.json \
  --project /absolute/path/to/local/runs/levir_ftsc_h2_augmentation_suite
```

The runner evaluates both validation and test splits, records AP50,
mAP50-95, AP75, precision, and recall, then writes per-run and mean/std summary
CSVs from real evaluation artifacts only.
