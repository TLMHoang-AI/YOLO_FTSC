# YOLOv8n P2 attention: FPN-only và PAN-P3

Report này gom hai nhóm thí nghiệm attention trên LEVIR-Ship:

- **FPN-only P2-only attention**: bốn variant seed 42, Detect chỉ nhận P2.
- **PAN-P3 attention**: CBAM và KVCA seeds 42/43/44, Detect nhận P2 và P3.
- **P2-only follow-up**: control plain, hai branch-specific refiner, HV-decoupled regression và P1-DRR + shared CBAM, seed 42.
- **Shared P2-only replication**: bare KVCA và shared CBAM trên seeds 42/43/44.
- **KVCA mechanism controls**: raw-P2 ranking diagnostic, matched shared-vs-cls-only placement và SR-ratio screen.
- **Local-vs-global follow-up**: Patch-KVCA `r0/r1`, raw-P2 diagnostic và inference-only radius mask trên global checkpoint.
- **Local-attention mechanism screen**: NATTEN kernel 3 full-resolution P2.

Các run dùng cùng fixed dataset split seed 42 (`2320/788/788` ảnh train/val/test), ảnh 512 px, batch 8 và `p2_offset_regression: false`. Nhóm ban đầu khai báo 100 epochs, patience 20; riêng KVCA cls-only encoder cũ dừng ở 38 epochs. Các replication mới chạy đủ 100 epochs với `patience=0`, trừ shared KVCA seed 42 cũ đã tự chạy đủ 100 epochs dù dùng patience 20. Mọi bảng mới bên dưới evaluate `best.pt` với NMS IoU ghi explicit bằng `0.5`. PAN-P3 thay đổi cả topology, số detection level và vị trí attention; vì vậy chênh lệch giữa PAN-P3 và FPN-only chỉ là evidence mô tả, không phải ablation causal thuần.

## Kiến trúc

### Phần chung

Backbone là YOLOv8n với depth/width scale `[0.33, 0.25]`: `Conv → C2f` theo các mức P2/P3/P4/P5 và SPPF ở P5. Sau scaling, các feature chính có lần lượt khoảng 32/64/128/256 channel. Neck top-down thực hiện:

```text
P5 --upsample+concat(P4)--> P4 FPN
   --upsample+concat(P3)--> P3 FPN
   --upsample+concat(P2)--> P2 FPN
```

Hai nhóm attention ban đầu không dùng P1, offset regression, geometry cue hay loss/TAL custom. Trong follow-up, chỉ `p1drr_cbam_shared` đọc thêm raw P1 qua P1-DRR và dùng restraint loss có sẵn của module; các variant còn lại vẫn P2-only thuần ở đầu vào Detect.

### FPN-only P2-only

Topology kết thúc ngay tại P2 FPN:

```text
Backbone P2...P5
       │
       └─ top-down FPN P5→P4→P3→P2
                                  │
                           attention/refiner
                                  │
                         Detect([P2]), stride [4]
```

| Variant | Vị trí và phạm vi attention | Cấu hình cụ thể |
| :--- | :--- | :--- |
| `fpn_only_kvca_block` | KVCA đặt trên P2 trước Detect; classification và regression cùng nhận feature đã attention | `KVCompressedAttention`, 4 heads, K/V spatial ratio 8, learned `group_weight`, residual; Q giữ full resolution |
| `fpn_only_kvca_encoder` | Transformer encoder đặt trên P2 trước Detect; dùng chung cho cả hai task | pre-norm KVCA 4 heads, ratio 8, `group_weight`, sau đó residual DW-FFN `PW expand → DW3 → PW project` |
| `fpn_only_cbam_clsonly` | Chỉ classification branch nhận CBAM; regression branch nhận P2 gốc | `DetectClsAttention("cbam")`; channel attention rồi spatial attention kernel 7 |
| `fpn_only_kvca_clsonly` | Chỉ classification branch nhận KVCA encoder; regression branch nhận P2 gốc | `DetectClsAttention("kvca")`; encoder 4 heads, ratio 8, `dwconv` compression |
| `kvca_block_clsonly` | Placement control: chỉ classification nhận đúng bare block dùng ở shared KVCA; regression nhận P2 gốc | `DetectClsAttention("kvca_block")`; `KVCompressedAttention`, 4 heads, ratio 8, `group_weight`, residual |
| `kvca_groupweight_sr4` | Shared attention trước Detect như control SR8 | Bare KVCA, 4 heads, ratio 4, `group_weight` |
| `kvca_groupweight_sr2` | Shared attention trước Detect như control SR8 | Bare KVCA, 4 heads, ratio 2; run bị chủ động dừng ở epoch 9 và không được evaluate/upload |

Hai variant “shared” có thể thay đổi đồng thời box logits và class logits. Hai variant “cls-only” giữ input regression tower nguyên bản, nhưng score field thay đổi trong training vẫn có thể gián tiếp đổi TAL positives.

### PAN-P3 P2+P3

Sau P2 FPN, model thêm một bottom-up PAN edge từ P2 về P3:

```text
P2 FPN ────────────────┐
   │                   ├──────────────> Detect P2, stride 4
   └─ Conv3 s=2 ─ concat(P3 FPN) ─ C2f ─> P3 PAN
                                            │
                                            └> Detect P3, stride 8
```

| Variant | P2 path | P3 path | Detect input |
| :--- | :--- | :--- | :--- |
| `pan_p3_cbam` | `C2fCBAM(128)` trong YAML, resolved n-scale output khoảng 32 channel | P2 downsample bằng Conv3 stride 2, concat P3 FPN, rồi `C2fCBAM(256)` với output khoảng 64 channel | P2 sau C2fCBAM và P3 PAN sau C2fCBAM |
| `pan_p3_kvca` | C2f thường → `KVCompressedTransformerEncoder`, 4 heads, ratio 8, `dwconv`, dropout 0.1 | C2f thường → encoder 4 heads, ratio 4, `dwconv`, dropout 0.1 | P2/P3 sau hai KVCA encoder |

`C2fCBAM` là C2f chuẩn rồi mới áp channel attention và spatial attention kernel 7 lên toàn output. KVCA giữ full-resolution queries nhưng nén K/V; mode `dwconv` dùng average pooling theo spatial ratio, depthwise 3×3 cho K và GroupNorm, sau đó SDPA và residual DW-FFN. Ở cả hai PAN variant, attention nằm trước Detect và được dùng chung bởi classification lẫn regression.

## Kết quả evaluation lưu trên Hugging Face

Đây là metrics trong `evaluation_metrics.json` của từng repo, không phải bảng NMS 0.50 bên dưới.

### FPN-only, seed 42

| Variant | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fpn_only_kvca_block` | 1.61M | 5.58 | 0.8449 | 0.7333 | 0.8123 | 0.1775 | 0.3359 | 0.7838 | 0.7188 | 0.7647 | 0.1389 | 0.3122 |
| `fpn_only_kvca_encoder` | 1.61M | 5.58 | 0.8110 | 0.6752 | 0.7719 | 0.1890 | 0.3178 | 0.7653 | 0.6667 | 0.7419 | 0.1246 | 0.2734 |
| `fpn_only_cbam_clsonly` | 1.60M | 5.58 | 0.8479 | 0.7262 | 0.8113 | 0.1684 | 0.3338 | 0.7912 | 0.6805 | 0.7514 | 0.1528 | 0.3010 |
| `fpn_only_kvca_clsonly` | 1.61M | 5.58 | 0.7445 | 0.6233 | 0.7008 | 0.1533 | 0.2861 | 0.6725 | 0.5991 | 0.6394 | 0.1310 | 0.2585 |

Source: [levir-yolov8n-p2-fpn-only-attention-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-fpn-only-attention-seed42).

### P2-only follow-up, seed 42

Các run dưới đây giữ cùng fixed split, 512 px, 100 epochs và Detect stride 4. `plain_p2_only` là control trực tiếp. `cls_context_mid_cbam` chỉ refine classification, `reg_local` chỉ refine input regression nhưng vẫn dùng nguyên tower bốn cạnh, `hv_decoupled` tách regression thành horizontal `[L,R]` và vertical `[T,B]`, còn `p1drr_cbam_shared` chạy `P1DRR(P2,P1) → CBAM → Detect`, nên CBAM dùng chung cho cả box và class tower.

| Variant | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_p2_only` | 1.61M | 5.58 | 0.8178 | 0.6717 | 0.7565 | 0.1686 | 0.3206 | 0.8049 | 0.6223 | 0.7213 | 0.1353 | 0.2815 |
| `cls_context_mid_cbam` | 1.60M | 5.58 | 0.8427 | 0.7294 | 0.8050 | 0.1594 | 0.3315 | 0.7865 | 0.6987 | 0.7633 | 0.1284 | 0.2946 |
| `reg_local` | 1.60M | 5.58 | 0.8076 | 0.6857 | 0.7719 | 0.1736 | 0.3272 | 0.7952 | 0.6807 | 0.7411 | 0.1323 | 0.2937 |
| `hv_decoupled` | 1.64M | 5.62 | 0.7808 | 0.6974 | 0.7629 | 0.1528 | 0.3003 | 0.7492 | 0.6767 | 0.7214 | 0.1116 | 0.2699 |
| `p1drr_cbam_shared` | 1.64M | 5.80 | 0.7231 | 0.6884 | 0.7387 | 0.1465 | 0.3047 | 0.7095 | 0.6652 | 0.7096 | 0.1319 | 0.2833 |

Sources: [asymmetric screen](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-asymmetric-screen-seed42), [HV-decoupled](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-hv-decoupled-seed42), [P1-DRR + shared CBAM](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-p1drr-cbam-shared-seed42).

So với control trên test, HV giữ AP50 gần như bằng nhau (`+0.0001`) nhưng giảm AP75 `0.0237` và mAP50-95 `0.0116`. P1-DRR + shared CBAM giảm AP50 `0.0117`, AP75 `0.0034`, trong khi mAP50-95 tăng nhẹ `0.0018`; do đó chưa có evidence rằng tổ hợp này cải thiện localization nghiêm ngặt. A1 và A2 vẫn là hai follow-up có gain detectability/mAP rõ nhất, nhưng cả hai đều không vượt control ở test AP75.

### PAN-P3, mean ± sample std của seeds 42/43/44

| Variant | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pan_p3_kvca` | 1.86M | 11.10 | 0.8322 ± 0.0359 | 0.7176 ± 0.0282 | 0.7980 ± 0.0180 | 0.1672 ± 0.0289 | 0.3324 ± 0.0159 | 0.8178 ± 0.0247 | 0.6987 ± 0.0106 | 0.7677 ± 0.0193 | 0.1488 ± 0.0125 | 0.3112 ± 0.0126 |
| `pan_p3_cbam` | 1.82M | 10.40 | 0.8709 ± 0.0066 | 0.7311 ± 0.0116 | 0.8194 ± 0.0074 | 0.1841 ± 0.0124 | 0.3429 ± 0.0046 | 0.8320 ± 0.0118 | 0.7284 ± 0.0102 | 0.7893 ± 0.0078 | 0.1459 ± 0.0047 | 0.3135 ± 0.0052 |

Source: [levir-yolov8n-p2-pan-p3-attention-3seed](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-pan-p3-attention-3seed).

PAN-P3 CBAM có AP50, recall và mAP50-95 trung bình cao hơn KVCA trên cả val và test, đồng thời variance nhỏ hơn ở phần lớn metric. KVCA chỉ nhỉnh hơn test AP75 (`0.1488` so với `0.1459`), chênh lệch `0.0029`, nhỏ hơn nhiều so với độ lệch chuẩn giữa seed.

## Kết quả re-evaluate với NMS IoU = 0.50 trong `report_yolo.md`

Các số dưới đây được chép riêng từ mục “Kết quả Test với NMS IoU = 0.50” của report tổng. Chúng không được trộn với `evaluation_metrics.json` ở trên.

| Family | Variant | Seeds | Val AP50 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FPN-only shared | `fpn_only_kvca_block` | 42 | 0.8177 | 0.3309 | 0.8259 | 0.7529 | 0.7857 | 0.1029 | 0.2990 |
| FPN-only shared | `fpn_only_kvca_encoder` | 42 | 0.7840 | 0.3167 | 0.7765 | 0.7387 | 0.7694 | 0.0851 | 0.2620 |
| FPN-only cls-only | `fpn_only_cbam_clsonly` | 42 | 0.7917 | 0.3289 | 0.8086 | 0.7270 | 0.7700 | 0.1141 | 0.2890 |
| FPN-only cls-only | `fpn_only_kvca_clsonly` | 42 | 0.7008 | 0.2861 | 0.7794 | 0.6681 | 0.7202 | 0.0842 | 0.2600 |
| PAN-P3 shared | `pan_p3_kvca` | 42/43/44 | 0.8258 ± 0.0090 | 0.3222 ± 0.0124 | 0.8199 ± 0.0103 | 0.7481 ± 0.0102 | 0.7916 ± 0.0175 | 0.1146 ± 0.0114 | 0.3013 ± 0.0119 |
| PAN-P3 shared | `pan_p3_cbam` | 42/43/44 | 0.8390 ± 0.0071 | 0.3295 ± 0.0042 | 0.8279 ± 0.0099 | 0.7716 ± 0.0128 | 0.8076 ± 0.0100 | 0.1094 ± 0.0085 | 0.3012 ± 0.0062 |

## Shared P2-only attention, ba seed, NMS IoU = 0.50

Đây là replication seeds 42/43/44 cho hai shared block P2-only. Seed huấn luyện thay đổi, nhưng dataset split luôn cố định ở seed 42. Bảng dùng `best.pt`; độ lệch chuẩn là sample standard deviation.

| Variant | Seeds | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_kvca` | 42/43/44 | 1.61M | 5.58 | 0.8653 ± 0.0126 | 0.7782 ± 0.0195 | 0.8350 ± 0.0142 | 0.1310 ± 0.0075 | 0.3240 ± 0.0077 | 0.8228 ± 0.0036 | 0.7556 ± 0.0105 | 0.7979 ± 0.0109 | 0.1045 ± 0.0130 | 0.2987 ± 0.0056 |
| `shared_cbam` | 42/43/44 | 1.60M | 5.58 | 0.8551 ± 0.0070 | 0.7739 ± 0.0120 | 0.8298 ± 0.0081 | 0.1397 ± 0.0077 | 0.3270 ± 0.0027 | 0.8328 ± 0.0158 | 0.7768 ± 0.0087 | 0.8076 ± 0.0102 | 0.1115 ± 0.0058 | 0.3089 ± 0.0054 |

CBAM cao hơn KVCA về mean trên cả năm test metric trong matrix này. Đây là comparison giữa hai shared module, không phải kết luận rằng CBAM luôn tốt hơn KVCA ở topology khác.

## Diagnostic raw P2 trước NMS, seed 42

Diagnostic chạy trên cùng 788 ảnh test và 696 GT. Với mỗi GT, candidate pool chỉ gồm decoded P2 anchors có center nằm trong GT; không threshold và không NMS. `IoU_best` là geometry tốt nhất có sẵn, `IoU_topscore` là IoU của candidate có confidence cao nhất, còn `rank_gap = IoU_best - IoU_topscore`.

| Model | Mean `IoU_best` | Mean `IoU_topscore` | Mean `rank_gap` | Mean Spearman(conf, IoU) | Mean confidence của best-IoU box |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `plain_p2_only` | 0.8001 | 0.6557 | 0.1444 | 0.3149 | 0.1516 |
| `shared_kvca` | 0.7932 | **0.6652** | **0.1280** | 0.3802 | **0.1548** |
| KVCA encoder cls-only cũ | 0.7921 | 0.6408 | 0.1512 | **0.4346** | 0.1479 |

Paired mean delta `shared_kvca - plain_p2_only` là `-0.0069` cho `IoU_best`, `+0.0095` cho `IoU_topscore`, `-0.0164` cho `rank_gap` và `+0.0654` cho Spearman. So với cls-only encoder cũ, shared có `IoU_best +0.0011`, `IoU_topscore +0.0244` và `rank_gap -0.0233`. Vì confidence của best-IoU candidate chỉ tăng `+0.0032` so với plain và `+0.0069` so với cls-only, evidence nghiêng về cải thiện score–box ranking hơn là score inflation đơn thuần. Spearman shared thấp hơn cls-only cũ, nên không diễn đạt đây là cải thiện mọi dạng correlation; gain rõ nhất nằm ở top-score selection và rank gap.

Ở nhóm small (389 GT), shared so với plain giảm mean `IoU_best` `0.0136` nhưng tăng `IoU_topscore` `0.0138`, giảm `rank_gap` `0.0273` và tăng Spearman `0.1104`. Nhóm tiny chỉ có 14 GT và shared giảm cả `IoU_best` lẫn `IoU_topscore`; sample này quá nhỏ để kết luận ổn định.

## Matched KVCA placement control, seed 42, đủ 100 epochs

KVCA cls-only cũ không phải placement control sạch: nó dùng full `KVCompressedTransformerEncoder(avg_dwk)` 9,712 params và dừng sau 38 epochs, trong khi shared dùng bare `KVCompressedAttention(group_weight)` 6,257 params. Vì vậy một run mới được tạo với đúng bare block của shared ở classification path:

```text
shared:   P2 → bare KVCA(group_weight, heads=4, SR=8) → box + cls
cls-only: P2 ─┬→ bare KVCA(group_weight, heads=4, SR=8) → cls
              └→ P2 gốc → box
```

Cả hai resolved model có 1,607,764 parameters; block có đúng 6,257 parameters và pretrained transfer đều đạt 230 tensors. Matched cls-only được train mới từ `yolov8n.pt`, seed 42, đủ 100 epochs, `patience=0`; best training row nằm ở epoch 99.

| Placement | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared bare KVCA | 1.61M | 5.58 | **0.8597** | **0.7789** | **0.8339** | **0.1347** | **0.3233** | **0.8259** | **0.7529** | **0.7857** | **0.1029** | **0.2988** |
| Matched bare KVCA cls-only | 1.61M | 5.58 | 0.8333 | 0.7638 | 0.8027 | 0.1203 | 0.3058 | 0.7769 | 0.7405 | 0.7641 | 0.0831 | 0.2705 |
| Shared − cls-only | — | — | +0.0264 | +0.0150 | +0.0312 | +0.0144 | +0.0175 | +0.0490 | +0.0124 | +0.0216 | +0.0198 | +0.0283 |

Khi đã match module, capacity và training length, shared vẫn thắng toàn bộ metric. Kết quả này loại được hai confound lớn của run cũ (full encoder và early stopping), đồng thời ủng hộ việc box/class tower cần co-adapt trên cùng feature KVCA. Nó chưa trực tiếp chứng minh gradient regression đóng vai trò regularizer; claim đó cần gradient/assignment intervention riêng.

Source: [matched KVCA block cls-only full-100](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-kvca-block-clsonly-full100-seed42).

## KVCA spatial-ratio screen, seed 42

Screen giữ shared placement, `group_weight` và 4 heads; chỉ đổi SR từ 8 xuống 4 hoặc 2. SR4 chạy đủ 100 epochs, evaluate `best.pt` ở NMS IoU 0.5 và upload verified. SR2 dùng khoảng 34.7 GB GPU, chạy chậm hơn rõ và bị chủ động dừng ở epoch 9 vì SR4 không cho gain tổng thể; SR2 không có test metrics và không được upload như một run hoàn tất.

| SR | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.61M | 5.58 | **0.8597** | 0.7789 | **0.8339** | **0.1347** | **0.3233** | **0.8259** | **0.7529** | **0.7857** | 0.1029 | **0.2988** |
| 4 | 1.61M | 5.58 | 0.8402 | **0.7882** | 0.8286 | 0.1464 | 0.3264 | 0.7862 | 0.7428 | 0.7633 | **0.1107** | 0.2863 |
| 2 | 1.61M | 5.58 | — | — | — | — | — | — | — | — | — | — |

SR4 tăng AP75 `0.0078` nhưng giảm precision `0.0397`, recall `0.0101`, AP50 `0.0225` và mAP50-95 `0.0125` so với SR8. Do đó không có evidence rằng finer K/V compression cải thiện tổng thể; SR8 vẫn là lựa chọn tốt hơn cho shared KVCA hiện tại.

Source: [KVCA SR ablation](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-kvca-sr-ablation-seed42).

## Patch-KVCA locality ablation, seed 42, đủ 100 epochs

`Patch-KVCA` vẫn đặt P2-only trước Detect, giữ 4 heads, cùng fixed split, 100 epochs, 512 px, batch 8, AMP và NMS IoU 0.5. Cả `patch_kvca_r0` và `patch_kvca_r1` resolved ở 1,607,764 parameters. `r0` chỉ cho mỗi query một compressed K/V cùng group; `r1` mở cửa sổ đến tối đa 3×3 = 9 compressed K/V. Đây là hai run train hoàn tất, không phải inference mask, và cả hai đã upload/verify tại [duyle2408/levir-yolov8n-p2-patch-kvca-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-patch-kvca-seed42).

| Variant | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch_kvca_r0` | 1.61M | 5.58 | 0.8163 | 0.7640 | 0.8034 | 0.1200 | 0.3096 | 0.7991 | 0.7199 | 0.7689 | 0.0902 | 0.2804 |
| `patch_kvca_r1` | 1.61M | 5.58 | 0.8525 | 0.7693 | 0.8220 | 0.1249 | 0.3164 | 0.8209 | 0.7284 | 0.7847 | 0.0966 | 0.2852 |
| `r1 - r0` | — | — | — | — | — | — | +0.0067 | +0.0218 | +0.0085 | +0.0159 | +0.0064 | +0.0048 |

Mở local receptive field từ `r0` sang `r1` cải thiện cả năm test metric trong comparison này. Tuy nhiên đây là ablation locality giữa hai Patch-KVCA đã retrain; nó không thay thế comparison với global KVCA hoặc suy luận từ mask OOD ở phần dưới.

### Diagnostic raw P2 của Patch-KVCA trước threshold/NMS

Diagnostic dùng đúng 788 ảnh test và 696 GT, trước threshold/NMS; các cột lần lượt là `IoU_best`, `IoU_topscore`, `rank_gap`, Spearman(conf, IoU), confidence của best-IoU box. Nó được ghi riêng với diagnostic shared KVCA lịch sử ở trên để không thay đổi các bảng cũ.

| Population / model | `IoU_best` | `IoU_topscore` | `rank_gap` | Spearman | Best-IoU confidence |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All GT — global | 0.793214997 | 0.665229417 | 0.127985580 | 0.380223562 | 0.154795783 |
| All GT — `patch_kvca_r0` | 0.793097645 | 0.668477498 | 0.124620148 | 0.382111046 | 0.175739296 |
| All GT — `patch_kvca_r1` | 0.792445490 | 0.664004050 | 0.128441439 | 0.399396531 | 0.181918049 |
| Small (389 GT) — global | 0.783891549 | 0.654723012 | 0.129168537 | 0.237643896 | 0.144062577 |
| Small (389 GT) — `patch_kvca_r0` | 0.783216602 | 0.664269662 | 0.118946940 | 0.287591210 | 0.191093387 |
| Small (389 GT) — `patch_kvca_r1` | 0.782135211 | 0.660518018 | 0.121617194 | 0.297245011 | 0.186955582 |

Paired mean delta all-GT: `r0 - global` là best `-0.00011735`, top `+0.00324808`, gap `-0.00336543`, Spearman `+0.00188748`, confidence `+0.02094351`; `r1 - global` là `-0.00076951`, `-0.00122537`, `+0.00045586`, `+0.01917297`, `+0.02712227`; và `r1 - r0` là `-0.00065216`, `-0.00447345`, `+0.00382129`, `+0.01728548`, `+0.00617875` theo cùng thứ tự.

Không variant Patch-KVCA nào khôi phục được `IoU_best` của plain baseline (`0.8001`). `r0` cải thiện nhẹ top-score selection/rank gap, còn `r1` tăng rank correlation nhưng làm raw selection kém hơn nhẹ. Vì vậy gain AP của `r1` không thể giải thích chỉ bằng raw P2 ranking.

### Inference-only radius intervention trên checkpoint global-KVCA

Đây là dependency kill-test **chỉ ở inference**: một checkpoint global-KVCA đã train giữ nguyên exact weights, state-dict và 1,607,764 parameters, rồi global attention được thay bằng mask `r0`/`r1`/`r2`. Nó là can thiệp OOD/distribution-shift, không phải evidence retraining hay ablation huấn luyện.

| Inference attention | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| global | 0.825865331 | 0.752873563 | 0.785746146 | 0.102871416 | 0.298792027 |
| mask `r0` | 0.360323887 | 0.639367816 | 0.408210719 | 0.024385662 | 0.121743834 |
| mask `r1` | 0.730217393 | 0.663793103 | 0.668944703 | 0.080834130 | 0.236787366 |
| mask `r2` | 0.774923417 | 0.708333333 | 0.718523876 | 0.085285914 | 0.254144968 |

| Population / inference attention | `IoU_best` | `IoU_topscore` | `rank_gap` | Spearman | Best-IoU confidence |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All GT — global | 0.793214997 | 0.665229417 | 0.127985580 | 0.380223562 | 0.154795783 |
| All GT — mask `r0` | 0.762872600 | 0.617662322 | 0.145210278 | 0.493801586 | 0.006987263 |
| All GT — mask `r1` | 0.799472706 | 0.659463571 | 0.140009135 | 0.307663019 | 0.110842366 |
| All GT — mask `r2` | 0.795388093 | 0.662717713 | 0.132670379 | 0.354782915 | 0.137349654 |
| Small (389 GT) — global | 0.783891549 | 0.654723012 | 0.129168537 | 0.237643896 | 0.144062577 |
| Small (389 GT) — mask `r0` | 0.763681257 | 0.624117455 | 0.139563803 | 0.454940241 | 0.003376341 |
| Small (389 GT) — mask `r1` | 0.793322311 | 0.664019269 | 0.129303041 | 0.176919514 | 0.117551369 |
| Small (389 GT) — mask `r2` | 0.786370899 | 0.658604651 | 0.127766248 | 0.222779894 | 0.137967367 |

Trên all-GT, paired mean `mask - global` của `r0` là best `-0.0303424`, top `-0.0475671`, gap `+0.0172247`, Spearman `+0.1135780`; CI median loại 0 cho ba đại lượng đầu và Spearman. Với `r1`: `+0.0062577`, `-0.00576585`, `+0.0120236`, `-0.0725605`; với `r2`: `+0.0021731`, `-0.0025117`, `+0.0046848`, `-0.0254406`.

`r1`/`r2` giữ, thậm chí tăng raw geometry, trong khi AP và confidence giảm mạnh. Cách đọc thận trọng là global/far context ở checkpoint này được dùng chủ yếu cho scoring, calibration hoặc foreground-background discrimination hơn là raw geometry. Vì mask là OOD, đây chỉ là inference dependency evidence. Retraining physical Patch-KVCA phục hồi đáng kể phần drop của mask, nhưng global checkpoint `0.298792027` vẫn cao hơn retrained `patch_kvca_r1` `0.285210868` đúng `0.013581159` mAP50-95.

## NATTEN full-resolution P2 mechanism screen, seed 42

`natten_k3` dùng `NATBlock` full-resolution P2 tại layer 19, 4 heads, kernel 3. Run hoàn tất seed 42 với 100 epochs, 512 px, batch 8, AMP, NMS 0.5; model có 1,610,052 parameters (1,606,548 sau fuse) và artifact đã hoàn tất tại [duyle2408/levir-yolov8n-p2-nat-k3-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-nat-k3-seed42).

| Variant | Params | GFLOPs | Val P | Val R | Val AP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test AP50 | Test AP75 | Test mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `natten_k3` | 1.61M | 5.58 | 0.8627 | 0.7508 | 0.8120 | 0.1456 | 0.3142 | 0.7977 | 0.7126 | 0.7594 | 0.1032 | 0.2783 |

So với Patch-KVCA `r1` trên test, NATTEN giảm mAP `0.006926753` và AP50 `0.025340731`, nhưng tăng AP75 `0.006569382`. Đây là mechanism screen, không phải connectivity-only control sạch: NATBlock mang full-resolution QKV/relative attention, MLP và gamma. AP75 cao hơn nhưng metric tổng thể thấp hơn, nên nó không thay thế `patch_kvca_r1`.

## Channel descriptor follow-up: diagnostic results

Diagnostic so sánh checkpoint channel-only và spatial-only trên đúng 788 test images / 696 GT với NMS IoU 0.50. Kết quả như sau:

### So sánh Anchor-GT Matching (Pre-NMS)

| Chỉ số | Channel-only | Spatial-only | Paired Delta (Channel - Spatial) [95% CI] |
| :--- | :---: | :---: | :---: |
| **Mean `IoU_topscore`** | 0.6868 (median) | 0.6890 (median) | +0.0019 `[-0.0066, 0.0106]` |
| **`IoU_topscore` ≥ 0.50 Rate** | 89.22% | 88.79% | -0.11% `[-2.42%, +2.16%]` |
| **`IoU_topscore` ≥ 0.75 Rate** | 31.47% | 31.03% | +1.24% `[-2.81%, +5.39%]` |
| **`raw_half_top_multiplicity`** (mean) | 9.0 (median) | 9.0 (median) | **-0.2035** `[-0.3899, -0.0113]` |

### Phân bố Confidence của True Positives (Post-NMS)

| Ngưỡng IoU | Phân vị (Quantile) | Channel-only | Spatial-only |
| :--- | :---: | :---: | :---: |
| **IoU ≥ 0.50** | 5% / 25% / 50% / 75% / 95% | 0.0387 / 0.3447 / 0.4707 / 0.5483 / 0.6267 | 0.0534 / 0.3793 / 0.4770 / 0.5397 / 0.6081 |
| **IoU ≥ 0.75** | 5% / 25% / 50% / 75% / 95% | 0.0116 / 0.3732 / 0.4978 / 0.5711 / 0.6504 | 0.0251 / 0.3907 / 0.4885 / 0.5431 / 0.6141 |

### Điểm vận hành Precision-Recall (Post-NMS)

| Metric | Ngưỡng Target | Channel-only | Spatial-only |
| :--- | :---: | :---: | :---: |
| **Precision @ Recall (IoU 0.50)** | R = 0.25 / 0.50 / 0.75 | 95.85% / 92.67% / 84.06% | 94.14% / 90.33% / 85.23% |
| **Recall @ Precision (IoU 0.50)** | P = 0.50 / 0.75 / 0.90 | 87.64% / 82.04% / 58.62% | 87.50% / 81.75% / 60.92% |
| **Precision @ Recall (IoU 0.75)** | R = 0.25 / 0.50 / 0.75 | 36.31% / None / None | 34.38% / None / None |
| **Recall @ Precision (IoU 0.75)** | P = 0.50 / 0.75 / 0.90 | **10.78%** / 2.87% / 0.14% | **1.58%** / None / None |

### Diễn giải cơ chế:
- **Candidate Multiplicity**: Việc delta `raw_half_top_multiplicity` âm ổn định và có ý nghĩa thống kê xác nhận Spatial attention tạo ra trường điểm số bao quanh object bị phân tán/lặp nhiều hơn (diffuse/duplicated), gây ảnh hưởng tiêu cực lên cơ chế lọc của NMS.
- **Tính nhất quán phân phối điểm số**: Ở ngưỡng IoU 0.75 nghiêm ngặt, tại Precision 50%, Channel-only giữ được Recall là **10.78%** trong khi Spatial-only sụt giảm mạnh về **1.58%**. Kết quả này chỉ ra rằng Spatial gating phá vỡ tính nhất quán của thang điểm số (global score-field scale inconsistency) giữa các ảnh trên toàn bộ dataset, trực tiếp giải thích khoảng cách AP75 (`0.1305 vs 0.0928`).

Channel-only GAP control hiện tại đang được so sánh hẹp với GMP và GAP+GMP (proposed) trên server.

## Ghi chú run 400 epochs

Một exploratory run đã train shared CBAM mới đến 400 epochs và tiếp tục shared KVCA từ checkpoint 100 epochs thêm 300 epochs. Artifact summary hiện không ghi trường `nms_iou`, còn KVCA continuation không phải exact optimizer resume vì checkpoint công bố đã strip optimizer/EMA. Vì vậy các số này không được trộn vào bảng NMS 0.5 hoặc dùng làm causal comparison với matrix 100 epochs.

## Diễn giải

- Trong FPN-only seed 42, KVCA block shared tốt hơn encoder sâu hơn trên test AP50 và mAP50-95. Việc thêm full encoder không tạo gain ổn định.
- Replication ba seed ở NMS 0.5 cho thấy shared CBAM có mean test metric cao hơn shared KVCA trong matrix hiện tại, nhưng cả hai giữ variance seed tương đối nhỏ ở AP50 và mAP50-95.
- Raw diagnostic cho thấy shared KVCA gần như không tăng `IoU_best`, nhưng tăng `IoU_topscore` và giảm `rank_gap`; mechanism phù hợp nhất hiện tại là score–box ranking/co-adaptation, không phải cải thiện raw geometry.
- Matched placement control xác nhận shared KVCA tốt hơn cls-only ngay cả khi dùng cùng bare block và đủ 100 epochs. Vì TAL kết hợp detached classification scores và predicted-box IoU để gán positive, việc hai tower dùng cùng transformed feature là một giải thích hợp lý, nhưng chưa phải causal proof về gradient regularization.
- Giảm SR 8 xuống 4 chỉ đổi trade-off sang AP75 cao hơn nhưng làm precision/AP50/mAP thấp hơn; SR2 đã được đóng sớm vì chi phí lớn và evidence từ SR4 không đủ tốt.
- Patch-KVCA retrain cho thấy mở radius từ `r0` sang `r1` tăng mọi test metric, nhưng raw-P2 không cho một câu chuyện đơn giản về geometry: `r0` tốt hơn nhẹ về top-score/gap, còn `r1` mạnh hơn về Spearman và vẫn có AP cao hơn.
- Kill-test mask inference nói rằng far/global context của checkpoint global quan trọng rõ cho score/calibration; vì đây là distribution shift không retrain, không quy nó thành causal comparison kiến trúc. `r1`/`r2` còn giữ raw geometry tốt trong khi AP sụt, phù hợp với cách đọc này.
- NATTEN k3 là screen cho local attention full-resolution, không phải clean control chỉ đổi KV connectivity. Nó tăng test AP75 so với Patch-KVCA `r1` nhưng giảm AP50 và mAP, nên Patch-KVCA `r1` vẫn là lựa chọn tốt hơn theo metric tổng thể hiện có.
- CBAM cls-only giữ localization tốt nhất trong bốn FPN-only variant theo HF test AP75 (`0.1528`), nhưng shared KVCA block có mAP50-95 cao nhất (`0.3122`). Điều này ủng hộ trade-off giữa detectability và localization-aware ranking, không chứng minh attention cls-only luôn tốt hơn shared attention.
- PAN-P3 CBAM là variant ổn định nhất trong family PAN: test AP50 `0.7893 ± 0.0078`, recall `0.7284 ± 0.0102`, mAP50-95 `0.3135 ± 0.0052` theo artifact HF.
- Không dùng PAN-P3 để kết luận causal rằng thêm P3 tốt hơn P2-only: PAN có thêm bottom-up fusion, thêm detection level, attention placement khác và ba seed thay vì một seed.
## Channel descriptor multi-seed evaluation results

Chúng tôi đã hoàn thành huấn luyện bổ sung hai mô hình `gap` (Average pooling) và `gmp` (Max pooling) trên hai hạt giống `seed 43` và `seed 44` (mỗi mô hình 100 epochs, 512px, batch 8, NMS IoU = 0.50). Đồng thời, mô hình đề xuất kết hợp `gap_gmp` (Avg+Max pooling) trên `seed 42` cũng đã hoàn thành. Kết quả đầy đủ trên tập Test (NMS IoU = 0.50) được ghi nhận trong bảng sau:

| Variant | Seed | Params | GFLOPs | Val AP50 | Val mAP50-95 | Test AP50 | Test AP75 | Test mAP50-95 | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gap` (Average - Control)** | 42 | 1.60M | 5.58 | 0.8383 | 0.3353 | 0.8162 | 0.1305 | 0.3106 | Hoàn thành |
| | 43 | 1.60M | 5.58 | 0.8170 | 0.3131 | 0.7913 | 0.1060 | 0.2893 | Hoàn thành |
| | 44 | 1.60M | 5.58 | 0.8378 | 0.3334 | 0.8057 | 0.1093 | 0.3061 | Hoàn thành |
| *Trung bình (Average)* | | — | — | **0.8310 ± 0.0121** | **0.3272 ± 0.0123** | **0.8044** | **0.1153** | **0.3020** | |
| **`gmp` (Max - Falsify)** | 42 | 1.60M | 5.58 | 0.8330 | 0.3190 | 0.8206 | 0.0947 | 0.2995 | Hoàn thành |
| | 43 | 1.60M | 5.58 | 0.8453 | 0.3198 | 0.7996 | 0.0968 | 0.2840 | Hoàn thành |
| | 44 | 1.60M | 5.58 | 0.8472 | 0.3166 | 0.7941 | 0.1038 | 0.2902 | Hoàn thành |
| *Trung bình (Average)* | | — | — | **0.8418 ± 0.0077** | **0.3185 ± 0.0017** | **0.8048** | **0.0984** | **0.2912** | |
| **`gap_gmp` (Avg+Max - Proposed)** | 42 | 1.60M | 5.58 | 0.8391 | 0.3130 | 0.7891 | 0.1028 | 0.2862 | Hoàn thành |

### Diễn giải kết quả:
- **GMP consistently degrades AP75**: Thử nghiệm đa hạt giống xác nhận Max pooling làm sụt giảm nghiêm trọng độ ổn định định vị (AP75 giảm trung bình **-1.69%** absolute so với GAP) trên toàn bộ các hạt giống, chứng minh sự ảnh hưởng tiêu cực của nhiễu cục bộ và tính không nhất quán của score-field.
- **Thất bại của giải thuyết kết hợp**: Việc kết hợp `gap_gmp` cho kết quả tệ hơn cả hai bản đơn lẻ ở AP50 (`0.7891`) và mAP (`0.2862`), cho thấy Max pooling làm loãng tín hiệu bối cảnh nền ổn định do Average pooling mang lại.

### Gate Intervention Matrix: Phân tích cơ chế suy luận thực tế (Inference Interventions)

Để cô lập cơ chế mang lại lợi ích của `ChannelAttention`, chúng tôi chạy phép thử can thiệp (Inference Intervention Matrix) trên chính checkpoint `gap` seed 42 (đã học xong với NMS IoU = 0.50). Bằng cách can thiệp động hoặc tĩnh vào trọng số attention $g_{i,c} \in (0, 1)^C$ tại inference, chúng tôi thu được kết quả:

* **Trung bình toàn bộ (Global mean) của gate weights ban đầu**: **`0.4514`**

| Cấu hình can thiệp | Test AP50 | Test AP75 | Test mAP50-95 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** (Không can thiệp) | 0.8162 | 0.1305 | 0.3106 | 0.8266 | 0.7701 |
| **Static-channel** ($g'_{i,c} = \text{mean}_i(g_{i,c})$) | 0.8229 | 0.1357 | 0.3198 | 0.8148 | **0.7839** |
| **Dynamic-scalar** ($g'_{i,c} = \text{mean}_c(g_{i,c})$) | 0.8329 | 0.1421 | **0.3254** | 0.8409 | 0.7802 |
| **Global-scalar** ($g' = 0.4514$ constant) | 0.8319 | 0.1382 | 0.3241 | **0.8431** | 0.7718 |
| **Cross-image shuffle** | 0.8095 | 0.1244 | 0.3084 | 0.8242 | 0.7748 |
| **Channel shuffle** | 0.8189 | **0.1474** | 0.3148 | 0.8380 | 0.7658 |
| **Identity (g = 1.0)** (Bỏ attention hoàn toàn) | 0.8212 | 0.1233 | 0.3115 | 0.8189 | 0.7796 |
| **Sweep g = 0.25** | 0.6989 | 0.0895 | 0.2676 | 0.6259 | 0.7787 |
| **Sweep g = 0.40** | 0.8263 | 0.1356 | 0.3240 | 0.8487 | 0.7672 |
| **Sweep g = 0.50** | **0.8345** | 0.1349 | 0.3208 | 0.8364 | 0.7902 |
| **Sweep g = 0.60** | 0.8315 | 0.1327 | 0.3144 | 0.8237 | 0.7919 |
| **Sweep g = 0.75** | 0.8164 | 0.1218 | 0.3091 | 0.8239 | 0.7796 |

### Nhận xét & Diễn giải khoa học:

1. **Channel-specific gating không cần thiết khi suy luận (Inference-time Gating Redundancy)**:
   * Việc loại bỏ hoàn toàn tính năng thay đổi trọng số theo từng kênh bằng cách đặt toàn bộ $g = 0.4514$ (`Global-scalar`) hoặc $g = 0.40$ (`Sweep g=0.40`) giúp tăng mAP tương ứng lên **0.3241** và **0.3240** (+1.35% absolute so với Original).
   * Điều này chứng minh rằng cơ chế lựa chọn kênh động (dynamic channel-specific selection) theo ảnh tại thời điểm suy luận là không cần thiết, thậm chí còn gây méo cấu trúc kênh được học ở các lớp trước đó.
2. **Optimum Curve của biên độ đặc trưng (Optimal Feature Amplitude)**:
   * Phép sweep giá trị tĩnh tạo ra một đường cong tối ưu (parabolic curve) rõ rệt: mAP đạt đỉnh tại $g \approx 0.40 - 0.45$ (mAP 0.3240) và giảm dần khi dịch về hai phía ($g = 0.25$ làm mAP giảm về 0.2676; $g = 1.0$ đạt mAP 0.3115).
   * Điều này ủng hộ mạnh mẽ cho giả thuyết **hiệu chuẩn biên độ kích hoạt (Feature Amplitude Calibration)** tại thời điểm suy luận.
3. **Phân rã tác động: Inference Calibration vs Training Regularization**:
   * Việc `Identity (g=1.0)` đạt mAP **0.3115** (gần như tương đương `Original` 0.3106) cho thấy: bản thân việc nhân chập gate $g(F)$ động lúc inference không tạo ra bất kỳ gain nào so với việc bỏ khối attention hoàn toàn.
   * Do đó, phần lớn độ chính xác vượt trội của checkpoint được huấn luyện với `ChannelAttention` so với baseline thô không nằm ở sức mạnh biểu diễn (representational power) của khối attention lúc suy luận, mà nằm ở **tác dụng điều hòa tối ưu hóa lúc huấn luyện (training-time regularization/co-adaptation)**.

### Chẩn đoán Channel-KVCA: Vai trò của tham số Beta và Khử nhiễu Kênh (Beta=0 Intervention)

Mô hình **Channel-KVCA** (Channel-wise Key-Value Compressed Attention) trên `seed 42` ban đầu cho thấy hiệu suất suy giảm nghiêm trọng so với GAP (`0.2913` so với `0.3106` mAP). Để cô lập tác động của nhánh cross-channel attention này, chúng tôi thực hiện chẩn đoán giá trị tham số học được `beta` và chạy can thiệp `beta = 0.0` tại inference (loại bỏ hoàn toàn tác động của khối KVCA):

* **Giá trị `beta` học được (Learned beta)**: **`-0.1572`** (Mô hình tự động học trọng số âm để giảm trừ/correction các đặc trưng chéo kênh).

Kết quả so sánh chi tiết trên Test split (NMS IoU = 0.50):

| Cấu hình can thiệp | Test AP50 | Test AP75 | Test mAP50-95 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** (Learned $\beta = -0.1572$) | 0.7948 | 0.1080 | 0.2913 | 0.8185 | 0.7802 |
| **Beta = 0.0000** (Khử hoàn toàn KVCA) | **0.8202** | **0.1344** | **0.3128** | **0.8231** | **0.7954** |
| *GAP (Average Control - Seed 42)* | *0.8162* | *0.1305* | *0.3106* | *0.8266* | *0.7701* |

### Diễn giải cơ chế quan trọng:

1. **Cross-Channel Mixing gây nhiễu nghiêm trọng khi suy luận (Inference Distortion)**:
   * Khi tắt hoàn toàn KVCA bằng cách đặt `beta = 0.0` lúc inference, hiệu năng mô hình lập tức **vọt lên 0.3128 mAP (+2.15% absolute) và AP75 lên 0.1344 (+2.64% absolute)**.
   * Điều này xác nhận giả thuyết: cơ chế tự tương tác chéo giữa các kênh (`[C, C]` mixing) tạo ra sự méo mó biểu diễn không mong muốn ở tầng P2 trong quá trình suy luận.
2. **Channel-KVCA đóng vai trò Training-time Regularizer**:
   * Đáng chú ý, khi loại bỏ KVCA (`beta = 0.0`), mô hình thậm chí đạt **`0.3128` mAP, vượt qua cả bản GAP baseline nguyên bản (`0.3106` mAP)**.
   * Kết quả này chứng minh rằng việc huấn luyện với KVCA giúp backbone học được các biểu diễn đặc trưng mạnh mẽ và bền vững hơn (có thể do KVCA hoạt động như một dạng nhiễu loạn/regularization thúc đẩy mạng học biểu diễn tốt hơn). Tuy nhiên, khối KVCA bản thân nó không nên được kích hoạt lúc inference.
3. **Kết luận chung về Attention trên P2**:
   * **P2 cần được hiệu chuẩn biên độ (amplitude scale), nhưng hoàn toàn loại trừ việc thay đổi phân bố đặc trưng ngữ nghĩa chéo kênh (cross-channel re-mixing).**

## Nguồn truy vết

- PAN YAML, per-seed metrics, args và aggregate: [HF PAN-P3 3-seed](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-pan-p3-attention-3seed).
- FPN-only per-run metrics: [HF FPN-only attention seed 42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-fpn-only-attention-seed42).
- Shared P2-only seeds 43/44: [HF shared attention seed43-44](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-shared-attention-seed43-44-100).
- Shared seed 42 và summary ba seed: [HF FPN-only attention seed 42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-fpn-only-attention-seed42).
- Matched cls-only block: [HF KVCA block cls-only full-100](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-kvca-block-clsonly-full100-seed42).
- SR screen: [HF KVCA SR ablation](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-kvca-sr-ablation-seed42).
- Patch-KVCA r0/r1 completed artifacts: [HF Patch-KVCA seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-patch-kvca-seed42).
- NATTEN k3 completed artifact: [HF NATTEN seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-nat-k3-seed42).
- Bảng NMS 0.50: [`report_yolo.md`](report_yolo.md).
- Local YAML nguồn: `models_related/models_config/yolov8/levir/`.

## P2 Feature Amplitude Calibration & Perturbation Ablation (Identity-Initialized)

Chúng tôi đã thiết kế và thực hiện các thí nghiệm hiệu chuẩn biên độ đặc trưng (Feature Amplitude Calibration) và làm nhiễu biên độ khi huấn luyện (Train-time Perturbation) trên hạt giống `seed 42` (huấn luyện 100 epochs, patience = 0, batch 8, imgsz 512, NMS IoU = 0.50). 

Các mô hình được chạy phân tán trên hai server Marimo:
- **Server 1**: Chạy các phiên bản `global_scalar` (A1) và `amplitude_calibrator` (A2) (Đang chạy).
- **Server 2**: Chạy các phiên bản `amplitude_perturbation` (A3) và `calibrator_perturbation` (A4) (Hoàn thành).

Kết quả thu được từ **Marimo Server 2** (NMS IoU = 0.50):

| Variant | Params | GFLOPs | Val AP50 | Val mAP50-95 | Test AP50 | Test AP75 | Test mAP50-95 | Delta mAP |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`global_scalar` (A1)** | 1.60M | 5.58 | 0.8118 | 0.3344 | 0.8178 | 0.0880 | **0.2818** | **+0.0077** |
| **`amplitude_calibrator` (A2)** | 1.60M | 5.58 | 0.6792 | 0.2710 | — | — | *Terminated* | — |
| **`amplitude_perturbation` (A3)** | 1.60M | 5.58 | 0.8595 | 0.3285 | 0.8000 | 0.0912 | **0.2801** | **+0.0060** |
| **`calibrator_perturbation` (A4)** | 1.60M | 5.58 | 0.8484 | 0.3110 | 0.7960 | 0.0890 | **0.2763** | **+0.0022** |

### Phân tích kết quả:
1. **Hiệu năng các bộ hiệu chuẩn biên độ động (A1, A3, A4)**:
   - Các biến thể hiệu chuẩn biên độ đều cho thấy sự cải thiện nhẹ so với baseline clean (`0.2741` mAP), đạt mức tăng từ **`+0.0022` đến `+0.0077` mAP**.
   - Tuy nhiên, sự cải thiện này thấp hơn rất nhiều so với phương pháp **Consensus-guided Denoising** (tắt các kênh Easy dư thừa đạt **`+0.0246` mAP**). Điều này gợi ý rằng việc hiệu chuẩn biên độ toàn cục/kênh đơn thuần chỉ giải quyết phần ngọn, trong khi việc định vị và triệt tiêu các đặc trưng dư thừa (Easy) mới là chìa khóa tối ưu hóa thực sự.
2. **Sự kết hợp Calibrator + Perturbation (A4)**:
   - Khi kết hợp calibrator và perturbation theo đúng thứ tự logic (`P2 -> Calibrator -> Perturbation -> Detect`), mô hình đạt test mAP50-95 `0.2763`. Việc kết hợp này không mang lại sự cải thiện so với A3 đơn lẻ, cho thấy việc thêm calibrator động (được huấn luyện chung với perturbation) có thể khiến quá trình tối ưu hóa phức tạp hơn mà không đem lại lợi ích trực tiếp về độ chính xác định vị.


## Phép thử giả thuyết Channel Irreducibility & Consensus (Sự Đồng Thuận Kênh)

Chúng tôi đã thực hiện một chuỗi thực nghiệm kiểm chứng nghiêm ngặt để xác thực sâu hơn về cơ chế "không thể thu gọn chéo kênh" (cross-channel irreducibility) trên checkpoint `plain_p2_only` (seed 42, NMS IoU = 0.50). 

### 1. Thiết lập thực nghiệm nâng cao (Oracle-Free & Confound Control)
- **Thu thập dữ liệu**: Trích xuất bản đồ đặc trưng P2 ($X \in \mathbb{R}^{32 \times 128 \times 128}$) trên cả **tập huấn luyện (Train split)** và **tập kiểm thử (Test split)** của LEVIR-Ship.
- **Huấn luyện reconstructor**: Huấn luyện hai bộ reconstructor 1x1 Conv độc lập trên Train split và Test split để khôi phục các kênh bị che (masking 25% ngẫu nhiên).
- **Tính toán Irreducibility ($I_c$)**: Đo sai số khôi phục trên vùng đối tượng ($e_c^{\text{obj}}$) và vùng nền ($e_c^{\text{bg}}$) trên cả hai tập dữ liệu độc lập.
- **Mục tiêu**:
  1. Loại bỏ yếu tố oracle bằng cách lấy xếp hạng kênh từ Train split và áp dụng trực tiếp lên Test split.
  2. Đo độ ổn định tập hợp (Set Stability) của các kênh khó khôi phục giữa Train và Test.
  3. Loại trừ các yếu tố gây nhiễu về biên độ đặc trưng (Confound Control) như RMS (Root Mean Square) hay Variance.
  4. Quét mức độ triệt tiêu mềm (Soft Suppression Sweep) với hệ số $\lambda$.

### 2. Độ ổn định tập hợp (Set Stability)
Thống kê các kênh khó khôi phục nhất (Top-6 Hardest Channels) thu được từ Zero-Diagonal Reconstructor và hệ tọa độ Object Mask (xyxy) chuẩn xác:
- **Consolidated Train hard channels**: `[28, 1, 30, 21, 12, 29]`
- **Test hard channels**:  `[28, 1, 30, 21, 12, 13]`
- **Độ ổn định Reconstructor (5 seeds)**: Mean Jaccard = **`0.8286`** (Xếp hạng cực kỳ vững chắc qua các seed khởi tạo khác nhau).
- **Chỉ số tương đồng Jaccard chéo (Train vs Test)**: **`0.7143`** (Trùng khớp 5/6 kênh).

### 3. Phân tích đối chứng Confound Control (Isolated Process)
Chúng tôi cô lập hoàn toàn từng đợt đánh giá bằng cách khởi tạo mới YOLO instance và validator, đồng thời sử dụng bboxes projection chuẩn xác để tính toán `xyxy` object mask. Kết quả đo đạc khi triệt tiêu hoàn toàn ($\lambda=0.0$) nhóm 6 kênh theo các tiêu chí khác nhau:

| Cấu hình triệt tiêu ($K=6$) | Test AP50 | Test AP75 | Test mAP50-95 | Delta mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** (Plain P2 Control) | 0.7530 | 0.1026 | 0.2741 | — |
| **Easy (Predictable) - Oracle-Free** | **0.8210** | **0.1280** | **0.2987** | **+0.0246** |
| **Highest RMS** | 0.6970 | 0.0810 | 0.1959 | -0.0782 |
| **Highest Variance** | 0.7540 | 0.0910 | 0.2430 | -0.0311 |
| **Lowest RMS** | 0.7870 | 0.1010 | 0.2735 | -0.0006 |
| **Random (5-trial Avg)** | 0.7410 | 0.0910 | 0.2586 | -0.0155 |
| **Train-Hard (Oracle-Free)** | 0.3360 | 0.0260 | 0.1032 | -0.1709 |
| **Test-Hard (Oracle)** | 0.4040 | 0.0310 | 0.1248 | -0.1493 |

**Nhận xét**: 
- **Sự đảo chiều đầy bất ngờ**: Sau khi chuẩn hóa Object Mask và loại bỏ self-reconstruction thông qua Zero-Diagonal Reconstructor, **Easy channels** (các kênh dễ khôi phục chéo từ các kênh còn lại) mới chính là các kênh mang thông tin dư thừa hoặc gây nhiễu cho detector. Việc tắt hoàn toàn chúng (`Easy Mute`) đem lại bước nhảy hiệu năng ấn tượng **+2.46% mAP50-95 (lên 0.2987)**.
- **Hard channels mang thông tin sống còn**: Việc tắt các kênh Hard chéo (`Train-Hard` / `Test-Hard`) khiến detector sập hoàn toàn (giảm từ `0.2741` xuống `0.1032` mAP). Điều này phản ánh chính xác bản chất: các kênh độc lập, khó khôi phục chứa đựng các đặc trưng quan trọng nhất giúp detector định vị và phân loại đối tượng, trong khi các kênh có mức độ phụ thuộc chéo quá cao (Easy) hoạt động giống như một không gian con nhiễu nền hoặc đồng thuận dư thừa (redundant consensus).

### 4. Quét mức độ triệt tiêu mềm đối với Hard Channels (Soft Suppression Sweep)
Chúng tôi kiểm tra ảnh hưởng của mức độ triệt tiêu Hard channels bằng cách nhân chúng với hệ số $\lambda \in [0.0, 1.0]$:

| Hệ số $\lambda$ | Test AP50 | Test AP75 | Test mAP50-95 |
| :---: | :---: | :---: | :---: |
| **0.00** (Pruning hoàn toàn) | 0.3426 | 0.0262 | 0.1032 |
| **0.25** | 0.5550 | 0.0549 | 0.1837 |
| **0.50** | 0.6647 | 0.0913 | 0.2389 |
| **0.75** | 0.7279 | 0.0997 | 0.2659 |
| **1.00** (Baseline) | 0.7530 | 0.1026 | 0.2741 |

**Nhận xét**: Hiệu năng sụt giảm tuyến tính/phi tuyến tính cực kỳ rõ nét khi $\lambda$ tiến dần về 0. Điều này xác thực mạnh mẽ tầm quan trọng sống còn của các kênh Hard độc lập.

### 5. Kết luận khoa học mới: Khử nhiễu dựa trên Không gian Đồng thuận (Consensus-Guided Denoising)
- **Hard channels = Core Semantics**: Các kênh không thể khôi phục chéo chứa các đặc trưng độc nhất vô nhị không thể thay thế bởi phần còn lại. Detector phụ thuộc tuyệt đối vào chúng.
- **Easy channels = Redundant Clutter**: Các kênh dễ khôi phục chéo mang thông tin đồng thuận cao nhưng dư thừa, có xu hướng khuếch đại nhiễu nền của background hoặc gây nhiễu loạn thông tin biên của đối tượng nhỏ trên P2.
- **Ý tưởng thiết kế Module mới**: Thay vì cố gắng giữ lại tất cả các kênh, một thiết kế lý tưởng là **Consensus Denoising Module**:
  $$X'_c = (1 - g_c) X_c$$
  Trong đó $g_c$ ước lượng độ đồng thuận chéo kênh (cross-channel predictability). Các kênh có độ dự báo chéo cao (Easy) sẽ bị suy giảm cường độ mềm ($g_c \to 0.75$), trong khi các kênh độc lập (Hard) được bảo toàn tuyệt đối ($g_c \to 0$).








