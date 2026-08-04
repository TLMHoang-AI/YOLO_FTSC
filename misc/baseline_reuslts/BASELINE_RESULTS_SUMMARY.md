# Baseline YOLO Results

- Download folder: `baseline_reuslts/`
- Test summary rows: 75
- Train `results.csv` files: 79

## Downloaded Repositories

| Repo | Folder | Test rows | Train logs |
| --- | --- | --- | --- |
| duyle2408/varroa-yolo-baselines-missing-part1-missing | varroa-yolo-baselines-missing-part1-missing | 9 | 9 |
| duyle2408/varroa-yolo-baselines-missing-part2-missing | varroa-yolo-baselines-missing-part2-missing | 12 | 12 |
| duyle2408/varroa-yolo-baselines-missing-part3-missing | varroa-yolo-baselines-missing-part3-missing | 9 | 13 |
| duyle2408/varroa-yolo-baselines-part2-full | varroa-yolo-baselines-part2-full | 24 | 24 |
| duyle2408/varroa-yolo-baselines-part1-full | varroa-yolo-baselines-part1-full | 21 | 21 |

## Aggregate By Model Scale

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLO11-l | 42,43,44 | 3 | 0.8959 +- 0.0060 | 0.3340 +- 0.0011 | 0.8985 +- 0.0037 | 0.8986 +- 0.0068 |
| full | YOLO11-n | 42,43,44 | 3 | 0.9009 +- 0.0149 | 0.3260 +- 0.0098 | 0.9108 +- 0.0149 | 0.8986 +- 0.0094 |
| full | YOLO11-s | 42,43,44 | 3 | 0.9067 +- 0.0124 | 0.3326 +- 0.0050 | 0.9053 +- 0.0086 | 0.9048 +- 0.0123 |
| missing | YOLO11-m | 42,43,44 | 3 | 0.8996 +- 0.0090 | 0.3275 +- 0.0092 | 0.9154 +- 0.0073 | 0.8960 +- 0.0054 |
| missing | YOLO11-x | 42,43,44 | 3 | 0.8999 +- 0.0054 | 0.3268 +- 0.0045 | 0.8928 +- 0.0146 | 0.8894 +- 0.0081 |
| full | YOLOv10-l | 42,43,44 | 3 | 0.8762 +- 0.0101 | 0.3338 +- 0.0096 | 0.8475 +- 0.0176 | 0.8270 +- 0.0010 |
| full | YOLOv10-n | 42,43,44 | 3 | 0.8800 +- 0.0089 | 0.3333 +- 0.0049 | 0.8439 +- 0.0147 | 0.8328 +- 0.0156 |
| full | YOLOv10-s | 42,43,44 | 3 | 0.8708 +- 0.0069 | 0.3247 +- 0.0057 | 0.8570 +- 0.0146 | 0.8256 +- 0.0073 |
| missing | YOLOv10-m | 42,43,44 | 3 | 0.8655 +- 0.0174 | 0.3315 +- 0.0028 | 0.8414 +- 0.0261 | 0.8091 +- 0.0242 |
| missing | YOLOv10-x | 42,43,44 | 3 | 0.8754 +- 0.0034 | 0.3366 +- 0.0077 | 0.8565 +- 0.0152 | 0.8417 +- 0.0280 |
| full | YOLOv5-l | 42,43,44 | 3 | 0.9111 +- 0.0098 | 0.3301 +- 0.0022 | 0.9085 +- 0.0074 | 0.8904 +- 0.0144 |
| full | YOLOv5-n | 42,43,44 | 3 | 0.8961 +- 0.0133 | 0.3227 +- 0.0063 | 0.9032 +- 0.0075 | 0.8966 +- 0.0082 |
| full | YOLOv5-s | 42,43,44 | 3 | 0.8956 +- 0.0085 | 0.3277 +- 0.0013 | 0.8958 +- 0.0142 | 0.8959 +- 0.0031 |
| missing | YOLOv5-m | 42,43,44 | 3 | 0.9203 +- 0.0077 | 0.3387 +- 0.0043 | 0.9056 +- 0.0106 | 0.9002 +- 0.0141 |
| missing | YOLOv5-x | 42,43,44 | 3 | 0.9035 +- 0.0055 | 0.3297 +- 0.0045 | 0.8940 +- 0.0075 | 0.8960 +- 0.0080 |
| full | YOLOv8-l | 42,43,44 | 3 | 0.8992 +- 0.0077 | 0.3287 +- 0.0022 | 0.8944 +- 0.0134 | 0.8843 +- 0.0126 |
| full | YOLOv8-n | 42,43,44 | 3 | 0.9002 +- 0.0081 | 0.3262 +- 0.0051 | 0.8998 +- 0.0046 | 0.8965 +- 0.0128 |
| full | YOLOv8-s | 42,43,44 | 3 | 0.8972 +- 0.0069 | 0.3295 +- 0.0013 | 0.8947 +- 0.0094 | 0.8806 +- 0.0081 |
| missing | YOLOv8-m | 42,43,44 | 3 | 0.9127 +- 0.0088 | 0.3355 +- 0.0025 | 0.9057 +- 0.0051 | 0.8950 +- 0.0074 |
| missing | YOLOv8-x | 42,43,44 | 3 | 0.9065 +- 0.0186 | 0.3323 +- 0.0090 | 0.9005 +- 0.0127 | 0.8880 +- 0.0146 |
| full | YOLOv9-c | 42,43,44 | 3 | 0.9011 +- 0.0089 | 0.3298 +- 0.0050 | 0.8998 +- 0.0101 | 0.9020 +- 0.0058 |
| full | YOLOv9-s | 42,43,44 | 3 | 0.9011 +- 0.0065 | 0.3327 +- 0.0065 | 0.8983 +- 0.0124 | 0.8979 +- 0.0062 |
| full | YOLOv9-t | 42,43,44 | 3 | 0.9056 +- 0.0172 | 0.3293 +- 0.0051 | 0.9036 +- 0.0070 | 0.8951 +- 0.0071 |
| missing | YOLOv9-e | 42,43,44 | 3 | 0.9029 +- 0.0176 | 0.3285 +- 0.0108 | 0.9081 +- 0.0052 | 0.8909 +- 0.0056 |
| missing | YOLOv9-m | 42,43,44 | 3 | 0.9222 +- 0.0043 | 0.3304 +- 0.0038 | 0.9162 +- 0.0112 | 0.9139 +- 0.0129 |

## YOLO11

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall | Repo folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLO11-l | 42,43,44 | 3 | 0.8959 +- 0.0060 | 0.3340 +- 0.0011 | 0.8985 +- 0.0037 | 0.8986 +- 0.0068 | varroa-yolo-baselines-part2-full |
| full | YOLO11-n | 42,43,44 | 3 | 0.9009 +- 0.0149 | 0.3260 +- 0.0098 | 0.9108 +- 0.0149 | 0.8986 +- 0.0094 | varroa-yolo-baselines-part2-full |
| full | YOLO11-s | 42,43,44 | 3 | 0.9067 +- 0.0124 | 0.3326 +- 0.0050 | 0.9053 +- 0.0086 | 0.9048 +- 0.0123 | varroa-yolo-baselines-part2-full |
| missing | YOLO11-m | 42,43,44 | 3 | 0.8996 +- 0.0090 | 0.3275 +- 0.0092 | 0.9154 +- 0.0073 | 0.8960 +- 0.0054 | varroa-yolo-baselines-missing-part3-missing |
| missing | YOLO11-x | 42,43,44 | 3 | 0.8999 +- 0.0054 | 0.3268 +- 0.0045 | 0.8928 +- 0.0146 | 0.8894 +- 0.0081 | varroa-yolo-baselines-missing-part3-missing |

## YOLOv10

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall | Repo folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLOv10-l | 42,43,44 | 3 | 0.8762 +- 0.0101 | 0.3338 +- 0.0096 | 0.8475 +- 0.0176 | 0.8270 +- 0.0010 | varroa-yolo-baselines-part2-full |
| full | YOLOv10-n | 42,43,44 | 3 | 0.8800 +- 0.0089 | 0.3333 +- 0.0049 | 0.8439 +- 0.0147 | 0.8328 +- 0.0156 | varroa-yolo-baselines-part2-full |
| full | YOLOv10-s | 42,43,44 | 3 | 0.8708 +- 0.0069 | 0.3247 +- 0.0057 | 0.8570 +- 0.0146 | 0.8256 +- 0.0073 | varroa-yolo-baselines-part2-full |
| missing | YOLOv10-m | 42,43,44 | 3 | 0.8655 +- 0.0174 | 0.3315 +- 0.0028 | 0.8414 +- 0.0261 | 0.8091 +- 0.0242 | varroa-yolo-baselines-missing-part2-missing |
| missing | YOLOv10-x | 42,43,44 | 3 | 0.8754 +- 0.0034 | 0.3366 +- 0.0077 | 0.8565 +- 0.0152 | 0.8417 +- 0.0280 | varroa-yolo-baselines-missing-part2-missing |

## YOLOv5

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall | Repo folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLOv5-l | 42,43,44 | 3 | 0.9111 +- 0.0098 | 0.3301 +- 0.0022 | 0.9085 +- 0.0074 | 0.8904 +- 0.0144 | varroa-yolo-baselines-part1-full |
| full | YOLOv5-n | 42,43,44 | 3 | 0.8961 +- 0.0133 | 0.3227 +- 0.0063 | 0.9032 +- 0.0075 | 0.8966 +- 0.0082 | varroa-yolo-baselines-part1-full |
| full | YOLOv5-s | 42,43,44 | 3 | 0.8956 +- 0.0085 | 0.3277 +- 0.0013 | 0.8958 +- 0.0142 | 0.8959 +- 0.0031 | varroa-yolo-baselines-part1-full |
| missing | YOLOv5-m | 42,43,44 | 3 | 0.9203 +- 0.0077 | 0.3387 +- 0.0043 | 0.9056 +- 0.0106 | 0.9002 +- 0.0141 | varroa-yolo-baselines-missing-part1-missing |
| missing | YOLOv5-x | 42,43,44 | 3 | 0.9035 +- 0.0055 | 0.3297 +- 0.0045 | 0.8940 +- 0.0075 | 0.8960 +- 0.0080 | varroa-yolo-baselines-missing-part1-missing |

## YOLOv8

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall | Repo folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLOv8-l | 42,43,44 | 3 | 0.8992 +- 0.0077 | 0.3287 +- 0.0022 | 0.8944 +- 0.0134 | 0.8843 +- 0.0126 | varroa-yolo-baselines-part1-full |
| full | YOLOv8-n | 42,43,44 | 3 | 0.9002 +- 0.0081 | 0.3262 +- 0.0051 | 0.8998 +- 0.0046 | 0.8965 +- 0.0128 | varroa-yolo-baselines-part1-full |
| full | YOLOv8-s | 42,43,44 | 3 | 0.8972 +- 0.0069 | 0.3295 +- 0.0013 | 0.8947 +- 0.0094 | 0.8806 +- 0.0081 | varroa-yolo-baselines-part1-full |
| missing | YOLOv8-m | 42,43,44 | 3 | 0.9127 +- 0.0088 | 0.3355 +- 0.0025 | 0.9057 +- 0.0051 | 0.8950 +- 0.0074 | varroa-yolo-baselines-missing-part1-missing |
| missing | YOLOv8-x | 42,43,44 | 3 | 0.9065 +- 0.0186 | 0.3323 +- 0.0090 | 0.9005 +- 0.0127 | 0.8880 +- 0.0146 | varroa-yolo-baselines-missing-part3-missing |

## YOLOv9

| Group | Model | Seeds | Runs | mAP50 | mAP50-95 | Precision | Recall | Repo folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | YOLOv9-c | 42,43,44 | 3 | 0.9011 +- 0.0089 | 0.3298 +- 0.0050 | 0.8998 +- 0.0101 | 0.9020 +- 0.0058 | varroa-yolo-baselines-part2-full |
| full | YOLOv9-s | 42,43,44 | 3 | 0.9011 +- 0.0065 | 0.3327 +- 0.0065 | 0.8983 +- 0.0124 | 0.8979 +- 0.0062 | varroa-yolo-baselines-part2-full |
| full | YOLOv9-t | 42,43,44 | 3 | 0.9056 +- 0.0172 | 0.3293 +- 0.0051 | 0.9036 +- 0.0070 | 0.8951 +- 0.0071 | varroa-yolo-baselines-part1-full |
| missing | YOLOv9-e | 42,43,44 | 3 | 0.9029 +- 0.0176 | 0.3285 +- 0.0108 | 0.9081 +- 0.0052 | 0.8909 +- 0.0056 | varroa-yolo-baselines-missing-part2-missing |
| missing | YOLOv9-m | 42,43,44 | 3 | 0.9222 +- 0.0043 | 0.3304 +- 0.0038 | 0.9162 +- 0.0112 | 0.9139 +- 0.0129 | varroa-yolo-baselines-missing-part2-missing |

## Notes

- duyle2408/varroa-yolo-baselines-missing-part3-missing: train results without test summary rows: yolo11n_seed42, yolo11n_seed43, yolo11n_seed44, yolo11s_seed42
