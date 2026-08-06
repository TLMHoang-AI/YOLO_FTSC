# LEVIR DBSS P2-aware generalization audit

Phân tích này chỉ dùng artifact sẵn có: `best.pt`, training `results.csv`, fixed split seed 42 và các summary đã upload. Không retrain, không thay architecture/loss/gamma và không suy diễn test-by-epoch vì không có checkpoint trung gian.

## Kết luận chính

- Không có pattern “cả ba seed tăng trên val nhưng không tăng trên test”. Seed 42 và 44 tăng trên cả hai split; seed 43 giảm trên cả hai. Dấu của Δ mAP50-95 khớp val/test ở **3/3 seed**.
- Mean Δ mAP50-95 gần 0 và variance giữa seed lớn: val `+0.00265 ± 0.02628`, test `-0.00040 ± 0.03040` (sample std). Vì vậy aggregate `0.3312` trên val chủ yếu do seed 42/44 bù seed 43, chưa phải bằng chứng cải thiện ổn định.
- DBSS statistics của cùng checkpoint rất gần nhau giữa val và test. Không có DBSS val/test gap rõ để kết luận correction phụ thuộc background split.
- Correction vẫn tăng separation ở seed 43 nhưng AP giảm mạnh. Với experiment hiện tại, residual separation là proxy chưa đủ mạnh cho detection quality.
- `fP2` của aware **giảm**, không tăng, so với baseline ở cả ba seed. Không có bằng chứng TAL bị kéo sang P2 chỉ do classification score; assigned IoU chỉ tăng rõ ở seed 42.
- Split có scene leakage lớn: 85 scene xuất hiện trong cả train/val/test. Vì val và test đều overlap mạnh với train, đây không phải evaluation scene-independent.

## Paired comparison

Giá trị dưới đây là `aware - baseline`; P/R/mAP là metric từ cùng seed và cùng fixed split.

| Seed | Split | ΔP | ΔR | ΔmAP50 | ΔAP75 | ΔmAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 42 | val | +0.02811 | +0.04402 | +0.03576 | +0.02015 | +0.01321 |
| 42 | test | +0.00764 | +0.03999 | +0.02212 | +0.02328 | +0.01305 |
| 43 | val | -0.07590 | -0.04841 | -0.05288 | -0.00743 | -0.02727 |
| 43 | test | -0.06794 | -0.03014 | -0.04737 | -0.02370 | -0.03657 |
| 44 | val | +0.03825 | +0.04387 | +0.03506 | +0.01524 | +0.02201 |
| 44 | test | +0.03137 | +0.01871 | +0.02769 | +0.02099 | +0.02234 |

Across seed, aware thắng `2/3` cho mọi metric trên cả val và test. Mean ± sample std đầy đủ nằm trong [paired_summary.csv](diagnostics/levir_dbss_generalization/paired_summary.csv), còn các giá trị model gốc nằm trong [paired_by_seed.csv](diagnostics/levir_dbss_generalization/paired_by_seed.csv).

## Training dynamics có thể quan sát

Best validation epoch của aware là 98/100 (seed 42), 48/68 (seed 43), và 98/100 (seed 44). Tương quan theo epoch giữa val mAP50-95 và `delta_q_pos` lần lượt là `0.840`, `0.777`, `0.867`; với displacement ratio là `0.921`, `0.744`, `0.804`. Correction mạnh dần đồng thời val tăng, nhưng thiếu checkpoint quanh best nên không thể biết test peak sớm hay muộn hơn.

Đường cong và dữ liệu: [aware_epoch_curves.png](diagnostics/levir_dbss_generalization/aware_epoch_curves.png), [aware_epoch_curves.csv](diagnostics/levir_dbss_generalization/aware_epoch_curves.csv).

## DBSS và assignment diagnostics

Diagnostics chạy no-grad, BatchNorm ở eval, trên `best.pt`; cả sáu checkpoint đều có Detect strides `[4, 8, 16, 32]`.

| Seed | Split | aware Δq_pos | aware G pre → post | aware displacement | fP2 base → aware | TAL score P2 base → aware | assigned IoU P2 base → aware |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | val | 0.02815 | 0.28576 → 0.30656 | 0.07144 | 0.9775 → 0.7185 | 0.4657 → 0.4820 | 0.7125 → 0.7238 |
| 42 | test | 0.02623 | 0.28397 → 0.30264 | 0.07112 | 0.9795 → 0.7165 | 0.4625 → 0.4831 | 0.7104 → 0.7213 |
| 43 | val | 0.02502 | 0.27218 → 0.28868 | 0.06270 | 0.8485 → 0.7364 | 0.4872 → 0.4642 | 0.7230 → 0.7163 |
| 43 | test | 0.02340 | 0.28728 → 0.30218 | 0.06254 | 0.8477 → 0.7369 | 0.4830 → 0.4584 | 0.7159 → 0.7083 |
| 44 | val | 0.02061 | 0.32147 → 0.33026 | 0.06687 | 0.8836 → 0.6922 | 0.4847 → 0.4837 | 0.7230 → 0.7216 |
| 44 | test | 0.01942 | 0.32275 → 0.33070 | 0.06714 | 0.8818 → 0.6844 | 0.4840 → 0.4819 | 0.7186 → 0.7182 |

Raw values, gồm `q_pos_pre/post` và `q_neg_pre/post`, ở [checkpoint_diagnostics.csv](diagnostics/levir_dbss_generalization/checkpoint_diagnostics.csv).

## Dataset, scene và background

| Split | Images | Objects | Empty ratio | Objects/image | mean w | mean h | mean sqrt(area) |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2,320 | 1,862 | 0.4987 | 0.8026 | 20.63 | 19.86 | 20.00 |
| val | 788 | 661 | 0.4975 | 0.8388 | 20.44 | 19.67 | 19.80 |
| test | 788 | 696 | 0.4746 | 0.8832 | 20.21 | 19.83 | 19.76 |

Scene counts là train/val/test = `113/100/92`; overlap train-val `99`, train-test `91`, val-test `86`, three-way `85`, union `114`. Đây là leakage theo scene rõ ràng, dù membership ảnh giữa ba split vẫn riêng biệt.

Background aggregate khá gần nhau. Mean black-border ratio train/val/test là `0.0243/0.0203/0.0279`; grayscale mean `74.98/76.86/75.19`; gradient mean `1.101/1.183/1.128`; entropy 32-bin `1.384/1.441/1.401`. Variance theo ảnh lớn hơn các chênh lệch mean này, nên chưa thấy distribution shift global mạnh.

Chi tiết: [split_summary.csv](diagnostics/levir_dbss_generalization/split_summary.csv), [scene_overlap.csv](diagnostics/levir_dbss_generalization/scene_overlap.csv), [scene_crop_counts.csv](diagnostics/levir_dbss_generalization/scene_crop_counts.csv), [background_statistics.csv](diagnostics/levir_dbss_generalization/background_statistics.csv), [objects.csv](diagnostics/levir_dbss_generalization/objects.csv).

## Subgroup findings

TP/FP/FN được xuất theo seed, split, IoU 0.50/0.75 cho empty, one-object, multiple-object, very-tiny `<16 px`, tiny `16–24 px`, larger `≥24 px`, và theo scene. Một vài pattern micro-aggregate qua ba seed:

- Aware tăng recall IoU50 trên test cho very-tiny (`0.887 → 0.900`) và multiple-object (`0.938 → 0.945`), nhưng tạo nhiều FP hơn.
- Ở IoU75, aware tăng recall test cho very-tiny (`0.390 → 0.434`) và multiple-object (`0.510 → 0.525`), nhưng giảm larger (`0.669 → 0.643`).
- FP/empty-image tăng từ `6.17 → 7.72` trên val và `6.47 → 7.66` trên test ở diagnostic threshold. Vì vậy gain không đến từ giảm false positive trên empty images.
- Precision tuyệt đối của bảng subgroup thấp vì diagnostic cố ý dùng confidence threshold `0.001` để giữ candidate cho matching; dùng bảng này để so relative behavior, không thay metric COCO/Ultralytics chính.

Toàn bộ breakdown nằm trong [subgroup_metrics.csv](diagnostics/levir_dbss_generalization/subgroup_metrics.csv).

## Quyết định theo rule đã đặt

1. Chưa có DBSS gap val/test lớn, nên chưa đủ bằng chứng background-dependent correction để giảm `gamma_max` hoặc regularize basis.
2. DBSS separation tăng ở seed 43 trong khi detection giảm trên cả val/test: objective residual separation hiện là proxy yếu hoặc chưa đủ.
3. `fP2` không tăng; nhánh “TAL bị kéo về P2 mà assigned IoU không tăng” không xảy ra theo đúng dạng giả thuyết.
4. Chưa thể đề xuất decay auxiliary loss dựa trên test-peak timing vì không có checkpoint trung gian. Bước hợp lý tiếp theo, nếu chạy experiment mới, là lưu checkpoint định kỳ trước khi thay loss/gamma.

## Reproducibility và artifact

- Runner: [analyze_levir_dbss_generalization.py](misc/analyze_levir_dbss_generalization.py).
- Output local: [diagnostics/levir_dbss_generalization](diagnostics/levir_dbss_generalization/).
- Output HF: `duyle2408/levir_dbss_p2_aware/diagnostics/generalization/`.
- Fixed split được kiểm tra đúng `2320/788/788`; paired seeds đúng `42/43/44`; checkpoint strides đúng `[4,8,16,32]`.
- Aggregate trong report được tính lại từ CSV thô và khớp HF summary trong tolerance `1e-4`.
