# GAP seed42 feature-map and missed-GT diagnostic

Checkpoint: `hf_gap/runs/gap/seed_42/weights/best.pt`
HF repo: `duyle2408/levir-yolov8n-p2-channel-descriptor-seed42`

Test images: **788**, GT: **696**, detected: **525**, missed: **171**, recall@conf0.25/IoU0.5: **0.754**

## Miss reasons
- `classification_low_conf`: **141**
- `localization_no_candidate`: **30**

## Size buckets

| bucket | total | missed | miss % |
|---|---:|---:|---:|
| `large_ge1000` | 21 | 2 | 9.5% |
| `small_medium_400_1000` | 274 | 30 | 10.9% |
| `tiny_lt100` | 14 | 14 | 100.0% |
| `very_small_100_400` | 387 | 125 | 32.3% |

## Notes

- Feature maps hook layer 19: `ChannelAttention(avg)`, after P2 FPN and before `Detect([19])`, stride 4.
- Overlay colors: green = matched GT, orange = miss with candidate IoU >= 0.3 but conf < 0.25, red = no good candidate.
- This is a diagnostic match, not Ultralytics mAP evaluation; use it to inspect misses and representations.

## Top FN images

| image | GT | detected | missed |
|---|---:|---:|---:|
| `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png` | 43 | 23 | 20 |
| `GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png` | 8 | 0 | 8 |
| `GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216.png` | 9 | 4 | 5 |
| `GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png` | 6 | 1 | 5 |
| `GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png` | 5 | 0 | 5 |
| `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_12305_8192.png` | 5 | 2 | 3 |
| `GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_4096.png` | 3 | 0 | 3 |
| `GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png` | 3 | 0 | 3 |
| `GF6_WFV_E131.8_N38.0_20200910_L1A1120034675-2_0_17285.png` | 3 | 0 | 3 |
| `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_6656_17285.png` | 8 | 6 | 2 |
| `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_11264_7680.png` | 5 | 3 | 2 |
| `GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png` | 3 | 1 | 2 |

## Visual artifacts

### `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png`

![overlay](overlays/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824_overlay.png)

![feature map](feature_maps/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824_feature_map.png)

### `GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png`

![overlay](overlays/GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240_overlay.png)

![feature map](feature_maps/GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240_feature_map.png)

### `GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216.png`

![overlay](overlays/GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216_overlay.png)

![feature map](feature_maps/GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216_feature_map.png)

### `GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png`

![overlay](overlays/GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120_overlay.png)

![feature map](feature_maps/GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120_feature_map.png)

### `GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png`

![overlay](overlays/GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632_overlay.png)

![feature map](feature_maps/GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632_feature_map.png)

### `GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_12305_8192.png`

![overlay](overlays/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_12305_8192_overlay.png)

![feature map](feature_maps/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_12305_8192_feature_map.png)

## Classification-low-confidence score bins

This section uses only `tables/gap_seed42_test_gt_diagnostic.csv`; no model rerun. Population: **141** GT with `reason == classification_low_conf`.

Summary:

- `best_candidate_conf >= 0.15`: **49/141** (34.8%)
- `best_candidate_conf >= 0.20`: **33/141** (23.4%)
- `best_candidate_iou >= 0.50`: **113/141** (80.1%)
- `best_candidate_iou >= 0.75`: **27/141** (19.1%)

### Confidence bins

| conf_bin | n | pct | median_iou | mean_iou | median_area |
| --- | --- | --- | --- | --- | --- |
| <0.05 | 53 | 37.6% | 0.664 | 0.642 | 300.161 |
| 0.05-0.10 | 28 | 19.9% | 0.631 | 0.636 | 268.000 |
| 0.10-0.15 | 11 | 7.8% | 0.726 | 0.712 | 287.413 |
| 0.15-0.20 | 16 | 11.3% | 0.655 | 0.658 | 315.693 |
| 0.20-0.25 | 18 | 12.8% | 0.682 | 0.684 | 220.810 |

### Confidence x IoU bins

| conf_bin | 0.30-0.50 | 0.50-0.75 | >=0.75 |
| --- | --- | --- | --- |
| nan | 15 | 0 | 0 |
| <0.05 | 8 | 37 | 8 |
| 0.05-0.10 | 4 | 17 | 7 |
| 0.10-0.15 | 0 | 6 | 5 |
| 0.15-0.20 | 1 | 12 | 3 |
| 0.20-0.25 | 0 | 14 | 4 |

### Size bucket x confidence bins

| bucket | <0.05 | 0.05-0.10 | 0.10-0.15 | 0.15-0.20 | 0.20-0.25 |
| --- | --- | --- | --- | --- | --- |
| large_ge1000 | 1 | 0 | 0 | 0 | 0 |
| small_medium_400_1000 | 14 | 2 | 2 | 3 | 3 |
| tiny_lt100 | 0 | 0 | 0 | 0 | 1 |
| very_small_100_400 | 38 | 26 | 9 | 13 | 14 |

### Images with most low-confidence FNs

| image | low_conf_fn | median_conf | median_iou | median_area |
| --- | --- | --- | --- | --- |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png | 17 | 0.065 | 0.687 | 182.000 |
| GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216.png | 5 | 0.050 | 0.634 | 273.000 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png | 4 | 0.045 | 0.612 | 126.554 |
| GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png | 3 | 0.002 | 0.622 | 127.713 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png | 3 | 0.007 | 0.563 | 138.607 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_12305_8192.png | 3 | 0.022 | 0.673 | 182.000 |
| GF6_WFV_E131.8_N38.0_20200910_L1A1120034675-2_0_17285.png | 3 | 0.176 | 0.494 | 180.000 |
| GF1_WFV3_E112.3_N21.4_20190806_L2A0004164428_11776_9216.png | 2 | 0.067 | 0.534 | 492.000 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_6656_17285.png | 2 | 0.078 | 0.698 | 308.000 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_11264_7680.png | 2 | 0.099 | 0.593 | 235.500 |
| GF1_WFV3_E112.3_N21.4_20190806_L2A0004164428_12593_7680.png | 2 | 0.107 | 0.621 | 415.000 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_10769_8192.png | 2 | 0.130 | 0.725 | 198.000 |

Interpretation: most low-confidence FNs are not just under the threshold. The largest mass is below `0.10`, so a pure threshold move would be expensive. Still, **49** GT are in the softer `>=0.15` band and are the cleanest target for calibration/ranking rescue before touching regression.

## Border and touching-image misses

Definition:

- `near_border_10pct`: GT center has `border_dist < 51.2 px` on a 512x512 image.
- `touching_border`: GT box touches the image boundary.

Full-test miss population: **171**.

- near-border misses: **60/171** (35.1%)
- touching-border misses: **11/171** (6.4%)
- near-border reasons: `classification_low_conf=54`, `localization_no_candidate=6`
- touching-border reasons: `classification_low_conf=10`, `localization_no_candidate=1`

### Images with most near-border misses

| image | near_border_miss | touching_miss | classification_low_conf | localization_no_candidate | median_border_dist | median_conf | median_iou |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png | 5 | 1 | 5 | 0 | 12.500 | 0.050 | 0.773 |
| GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png | 3 | 0 | 2 | 1 | 8.300 | 0.002 | 0.494 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png | 2 | 1 | 2 | 0 | 6.334 | 0.017 | 0.563 |
| GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216.png | 2 | 0 | 2 | 0 | 31.750 | 0.049 | 0.664 |
| GF6_WFV_E131.8_N38.0_20200910_L1A1120034675-2_0_17285.png | 2 | 0 | 2 | 0 | 33.250 | 0.247 | 0.552 |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_512_11776.png | 1 | 1 | 1 | 0 | 10.500 | 0.217 | 0.516 |
| GF1_WFV3_E112.3_N21.4_20190806_L2A0004164428_2560_11264.png | 1 | 1 | 1 | 0 | 7.500 | 0.186 | 0.668 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_4096.png | 1 | 1 | 0 | 1 | 9.500 | 0.054 | 0.000 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png | 1 | 1 | 1 | 0 | 12.500 | 0.005 | 0.888 |
| GF1_WFV3_E121.9_N35.6_20190805_L2A0004161897_11776_13312.png | 1 | 1 | 1 | 0 | 9.500 | 0.249 | 0.698 |
| GF1_WFV4_E114.5_N11.7_20200703_L2A0004902460_13312_4608.png | 1 | 1 | 1 | 0 | 12.500 | 0.018 | 0.678 |
| GF6_WFV_E131.8_N38.0_20200910_L1A1120034675-2_18944_11264.png | 1 | 1 | 1 | 0 | 7.000 | 0.187 | 0.755 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_11264_7680.png | 1 | 1 | 1 | 0 | 6.000 | 0.004 | 0.512 |
| GF6_WFV_E137.7_N29.1_20200521_L1A1119999499-2_21904_7168.png | 1 | 1 | 1 | 0 | 4.500 | 0.001 | 0.526 |
| GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_14973_6656.png | 1 | 0 | 1 | 0 | 30.500 | 0.071 | 0.563 |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png | 1 | 0 | 0 | 1 | 34.949 | 0.025 | 0.133 |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_3072_10752.png | 1 | 0 | 1 | 0 | 37.500 | 0.468 | 0.480 |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_3072_7680.png | 1 | 0 | 1 | 0 | 20.500 | 0.003 | 0.721 |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_512_12288.png | 1 | 0 | 1 | 0 | 39.000 | 0.192 | 0.644 |
| GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_1024_7168.png | 1 | 0 | 0 | 1 | 15.652 | 0.301 | 0.200 |

### Touching-border missed GT cases

| image | gt_id | reason | bucket | area | border_dist | best_candidate_conf | best_candidate_iou |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_512_11776.png | 2 | classification_low_conf | small_medium_400_1000 | 777.000 | 10.500 | 0.217 | 0.516 |
| GF1_WFV3_E112.3_N21.4_20190806_L2A0004164428_2560_11264.png | 1 | classification_low_conf | small_medium_400_1000 | 403.000 | 7.500 | 0.186 | 0.668 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_4096.png | 2 | localization_no_candidate | very_small_100_400 | 361.000 | 9.500 | 0.054 | 0.000 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png | 0 | classification_low_conf | small_medium_400_1000 | 500.000 | 12.500 | 0.005 | 0.888 |
| GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png | 2 | classification_low_conf | very_small_100_400 | 111.257 | 5.679 | 0.002 | 0.453 |
| GF1_WFV3_E121.9_N35.6_20190805_L2A0004161897_11776_13312.png | 0 | classification_low_conf | small_medium_400_1000 | 418.000 | 9.500 | 0.249 | 0.698 |
| GF1_WFV4_E114.5_N11.7_20200703_L2A0004902460_13312_4608.png | 0 | classification_low_conf | small_medium_400_1000 | 475.000 | 12.500 | 0.018 | 0.678 |
| GF6_WFV_E131.8_N38.0_20200910_L1A1120034675-2_18944_11264.png | 0 | classification_low_conf | very_small_100_400 | 196.000 | 7.000 | 0.187 | 0.755 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png | 26 | classification_low_conf | very_small_100_400 | 195.000 | 6.500 | 0.097 | 0.773 |
| GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_11264_7680.png | 1 | classification_low_conf | very_small_100_400 | 156.000 | 6.000 | 0.004 | 0.512 |
| GF6_WFV_E137.7_N29.1_20200521_L1A1119999499-2_21904_7168.png | 0 | classification_low_conf | very_small_100_400 | 198.000 | 4.500 | 0.001 | 0.526 |

Interpretation: border/touching objects contribute materially to the miss set, but the dominant failure mode remains low classification confidence. Most border misses still have usable candidates; the score is suppressed rather than the box disappearing.

## GAP Gradient Coupling
Checkpoint: `hf_gap/runs/gap/seed_42/weights/best.pt`
Layer: `model.model[19] ChannelAttention(avg), P2 pre-Detect`
Split: `test`; images: **788**; loss: **classification**; device: `cuda`

## Gradient Decomposition

| metric | p25 | median | p75 |
|---|---:|---:|---:|
| `indirect_over_full` | 0.024056 | 0.040136 | 0.058621 |
| `cos_full_detach` | 0.998282 | 0.999194 | 0.999711 |
| `indirect_object_bg_ratio` | 0.999999 | 1.000000 | 1.000001 |

Max forward absolute diff across normal vs detached passes: `0`.
Interpretation: **mixed_or_small_gradient_coupling**.

## Gate/Failure Correlations

| relation | Pearson | Spearman | n |
|---|---:|---:|---:|
| `gate_mean_vs_low_conf_fn` | -0.0635 | -0.1302 | 788 |
| `gate_mean_vs_median_positive_conf` | 0.1690 | 0.2494 | 340 |
| `gate_std_vs_low_conf_fn` | -0.0944 | -0.0648 | 788 |
| `gate_std_vs_median_positive_conf` | -0.0541 | -0.0290 | 340 |
| `p2_rms_vs_low_conf_fn` | 0.0485 | 0.0859 | 788 |
| `p2_rms_vs_median_positive_conf` | -0.0043 | -0.0936 | 340 |

Artifacts:
- `investigate_gap_feature_miss/tables/gap_gradient_coupling_per_image.csv`
- `investigate_gap_feature_miss/gap_gradient_coupling_summary.json`
