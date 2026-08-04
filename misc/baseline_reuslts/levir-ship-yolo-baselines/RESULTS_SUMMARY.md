# LEVIR-Ship YOLO baseline results

Five YOLO baselines were trained and evaluated with dataset/training seeds 42,
43, and 44. Each split contains 2,320 training, 788 validation, and 788 test
images. Models use one detection class, an input size of 512, and batch size 8.

Validation values are taken from the epoch with the highest validation
mAP50-95 in each training `results.csv`. Test values come from explicitly
loading that run's `best.pt` and calling Ultralytics validation with
`split="test"`. Aggregate values are mean +/- sample standard deviation over
the three seeds.

## Aggregate validation results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv5nu | 0.718 +/- 0.103 | 0.639 +/- 0.057 | 0.674 +/- 0.087 | 0.258 +/- 0.039 |
| YOLOv8n | **0.800 +/- 0.017** | 0.700 +/- 0.030 | **0.752 +/- 0.022** | 0.286 +/- 0.015 |
| YOLOv9t | 0.781 +/- 0.040 | **0.702 +/- 0.014** | 0.744 +/- 0.049 | 0.286 +/- 0.022 |
| YOLOv10n | 0.761 +/- 0.042 | 0.674 +/- 0.006 | 0.737 +/- 0.035 | **0.298 +/- 0.022** |
| YOLO11n | 0.781 +/- 0.034 | 0.668 +/- 0.047 | 0.724 +/- 0.062 | 0.285 +/- 0.038 |

## Aggregate test results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv5nu | 0.732 +/- 0.054 | 0.601 +/- 0.078 | 0.657 +/- 0.082 | 0.243 +/- 0.028 |
| YOLOv8n | 0.782 +/- 0.019 | 0.668 +/- 0.029 | 0.718 +/- 0.031 | 0.264 +/- 0.013 |
| YOLOv9t | **0.784 +/- 0.031** | **0.687 +/- 0.034** | **0.738 +/- 0.026** | 0.277 +/- 0.011 |
| YOLOv10n | 0.743 +/- 0.050 | 0.661 +/- 0.022 | 0.729 +/- 0.030 | **0.289 +/- 0.015** |
| YOLO11n | 0.742 +/- 0.042 | 0.666 +/- 0.032 | 0.707 +/- 0.049 | 0.266 +/- 0.025 |

## Per-run validation results

| Run | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| yolov5nu_seed42 | 93 | 0.8321 | 0.7046 | 0.7739 | 0.3025 |
| yolov5nu_seed43 | 25 | 0.6313 | 0.6160 | 0.6283 | 0.2360 |
| yolov5nu_seed44 | 28 | 0.6918 | 0.5972 | 0.6201 | 0.2346 |
| yolov8n_seed42 | 68 | 0.8174 | 0.7156 | 0.7769 | 0.3029 |
| yolov8n_seed43 | 56 | 0.7825 | 0.7184 | 0.7415 | 0.2780 |
| yolov8n_seed44 | 65 | 0.7998 | 0.6657 | 0.7374 | 0.2766 |
| yolov9t_seed42 | 65 | 0.7803 | 0.6985 | 0.7599 | 0.3045 |
| yolov9t_seed43 | 67 | 0.7413 | 0.6897 | 0.6897 | 0.2611 |
| yolov9t_seed44 | 100 | 0.8215 | 0.7176 | 0.7827 | 0.2921 |
| yolov10n_seed42 | 77 | 0.7747 | 0.6815 | 0.7712 | 0.3233 |
| yolov10n_seed43 | 67 | 0.7140 | 0.6697 | 0.7021 | 0.2825 |
| yolov10n_seed44 | 99 | 0.7939 | 0.6713 | 0.7364 | 0.2885 |
| yolo11n_seed42 | 87 | 0.8196 | 0.7216 | 0.7917 | 0.3273 |
| yolo11n_seed43 | 28 | 0.7544 | 0.6421 | 0.6700 | 0.2530 |
| yolo11n_seed44 | 49 | 0.7699 | 0.6393 | 0.7098 | 0.2752 |

## Per-run test results

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| yolov5nu_seed42 | 0.7937 | 0.6853 | 0.7519 | 0.2750 |
| yolov5nu_seed43 | 0.6967 | 0.5311 | 0.6063 | 0.2257 |
| yolov5nu_seed44 | 0.7048 | 0.5880 | 0.6131 | 0.2269 |
| yolov8n_seed42 | 0.7697 | 0.6436 | 0.7094 | 0.2602 |
| yolov8n_seed43 | 0.8040 | 0.7008 | 0.7522 | 0.2780 |
| yolov8n_seed44 | 0.7717 | 0.6607 | 0.6913 | 0.2530 |
| yolov9t_seed42 | 0.7629 | 0.6610 | 0.7221 | 0.2702 |
| yolov9t_seed43 | 0.7681 | 0.6737 | 0.7231 | 0.2725 |
| yolov9t_seed44 | 0.8198 | 0.7259 | 0.7678 | 0.2894 |
| yolov10n_seed42 | 0.6954 | 0.6365 | 0.6956 | 0.2739 |
| yolov10n_seed43 | 0.7391 | 0.6693 | 0.7383 | 0.2906 |
| yolov10n_seed44 | 0.7958 | 0.6786 | 0.7542 | 0.3033 |
| yolo11n_seed42 | 0.7881 | 0.7026 | 0.7545 | 0.2924 |
| yolo11n_seed43 | 0.7049 | 0.6416 | 0.6559 | 0.2417 |
| yolo11n_seed44 | 0.7321 | 0.6542 | 0.7107 | 0.2628 |

## Main observations

- YOLOv10n has the highest mean mAP50-95 on both validation and test.
- YOLOv9t has the highest mean test precision, recall, and mAP50.
- YOLOv8n has the smallest test mAP50-95 variation across seeds after YOLOv9t
  (0.013 versus 0.011), and the highest mean validation precision and mAP50.
- Validation and test rankings are broadly consistent, but individual-seed
  gaps show that reporting the three-seed mean and deviation is important.
