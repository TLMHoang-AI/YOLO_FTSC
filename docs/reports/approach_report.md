# Báo cáo các hướng tiếp cận YOLO cho phát hiện vật thể nhỏ

## 1. Phạm vi

Tài liệu này mô tả các thay đổi kiến trúc và loss đang được thử nghiệm trên họ YOLO cho bài toán tàu nhỏ. Trọng tâm là cơ chế của từng method, vị trí can thiệp, đường gradient và ảnh hưởng tại inference; không dùng kết quả của một run đơn lẻ để kết luận method tốt hơn.

Các hướng chính gồm:

- YOLO với detection level P2.
- P2 offset regression.
- P2 kết hợp DBSS.
- HIT (Dual-Irreducibility Hardness-Induced Transport).
- GCTS v1 và GCTS v2.
- Non-uniform DFL và pair-competitive DFL.
- Conflict-guided fine reconstruction (CFR).

## 2. YOLO-P2: baseline cho tiny object

YOLO chuẩn thường dự đoán trên P3, P4 và P5, tương ứng stride 8, 16 và 32. Với vật thể rất nhỏ, việc bắt đầu ở P3 làm một đối tượng chỉ còn vài activation, nên sai lệch một cell có thể làm giảm IoU mạnh.

YOLO-P2 mở rộng neck thêm một nhánh top-down tới P2 và dùng head bốn mức:

```text
backbone P2 (stride 4)
        ↑
P3 upsample + fusion
        ↓
Detect [P2, P3, P4, P5]
```

Nhánh bottom-up P3 vẫn được tạo lại từ fused P2 để duy trì truyền thông tin đa tỉ lệ. P2 cung cấp lưới dày hơn bốn lần P3 theo số location, nên tăng khả năng gán positive và định vị tàu nhỏ.

Đổi lại, P2 làm tăng bộ nhớ, FLOPs và số candidate của head. Đây là baseline kiến trúc chung để đánh giá các method P2-specific; bản thân nó không bao gồm offset regression, DBSS, GCTS, NUDFL hay CFR.

## 3. P2 offset regression

### Mục tiêu

P2 có lưới dày nhưng mỗi cạnh bounding box vẫn được dự đoán từ cùng một sampling location. P2 offset regression cho phép bốn cạnh trái, trên, phải và dưới lấy feature tại bốn vị trí lệch nhau ở mức sub-cell.

### Cơ chế

Từ feature P2, module dự đoán tám giá trị:

\[
\Delta=(\Delta x_l,\Delta y_l,\ldots,\Delta x_b,\Delta y_b).
\]

Offset được giới hạn bằng `tanh` và hệ số bán kính \(\rho\). Với mỗi cạnh \(s\), feature được lấy mẫu bằng bilinear `grid_sample`:

\[
F_2^s=\operatorname{GridSample}(F_2, G+\rho\tanh(\Delta_s)).
\]

Bốn nhánh sau đó sinh riêng phân phối DFL cho \(l,t,r,b\). Offset network được zero-initialize và side logits được khởi tạo từ regression predictor gốc, nên điểm bắt đầu gần với head mặc định.

### Phạm vi

- Chỉ áp dụng cho P2; P3-P5 giữ regression path chuẩn.
- Có thêm phép sampling và predictor khi inference.
- Là option độc lập, mặc định phải là `false` để tránh làm nhiễm ablation của method khác.
- Không nên suy luận rằng checkpoint “P2” có offset nếu config không bật rõ `p2_offset_regression: true`.

## 4. DBSS: Dynamic Background Subspace Suppression

### Trực giác

Trong ảnh biển, phần lớn feature có thể được giải thích bởi các pattern nền lặp lại như mặt nước, sóng và đường bờ. DBSS xây dựng một background subspace động từ chính feature map, tách phần có thể tái tạo bởi nền và dùng residual khó tái tạo để tạo correction có biên độ giới hạn.

### Quy trình

Với token feature \(X\):

1. Chiếu feature bằng convolution \(1\times1\) và chuẩn hóa.
2. Lấy các basis candidate trên một adaptive grid.
3. Chấm điểm candidate theo cosine similarity trung bình với toàn bộ token.
4. Chọn tập basis vừa đại diện nền tốt vừa không quá trùng nhau.
5. Giải ridge projection để tái tạo thành phần nền:

\[
\widehat X_{bg}=B(B^TB+\lambda I)^{-1}B^TX.
\]

6. Lấy residual:

\[
R=X-\widehat X_{bg}.
\]

7. Từ residual, dự đoán hướng và độ lớn của correction. Correction được chặn mềm:

\[
\Delta X=alpha\gamma_{max}\gamma
\frac{d(R)}{1+\lVert d(R)\rVert},
\qquad
Y=X+\Delta X.
\]

`gamma` điều khiển mức can thiệp, tránh để module phá hỏng feature pretrained ngay từ đầu.

### P2 + DBSS

Biến thể trực tiếp đặt DBSS trên fused P2 trước Detect:

```text
fused P2 → DBSS → Detect P2
        └────────→ bottom-up P3 (tùy topology)
```

Cần ghi rõ topology của từng YAML:

- **DBSS full:** feature đã sửa có thể đi tiếp tới nhiều nhánh downstream.
- **DBSS trước P2 Detect:** chỉ regression/classification tại P2 nhận feature đã sửa; bottom-up P3 có thể vẫn dùng P2 gốc.
- **DBSS P2-aware/routed:** auxiliary objective dùng positive và target score của TAL tại P2 để đánh giá residual đúng vùng detection quan tâm.
- **Feature-only ablation:** giữ phép biến đổi feature nhưng tắt hoặc tách auxiliary supervision.

Các topology này không tương đương và không được gộp thành cùng một method khi so sánh.

### Auxiliary signal

Ở biến thể TAL-aware, residual ratio tại positive P2 được weighted bằng target score. Loss này khuyến khích đối tượng khó ít bị background subspace giải thích hơn. DBSS vẫn thay đổi feature trên inference path, khác với CFR training-only.

## 5. HIT: Dual-Irreducibility Hardness-Induced Transport

### Trực giác

HIT xem một location là khó khi feature tại đó không thể được tái tạo tốt theo cả hai hướng:

- từ hàng xóm không gian;
- từ quan hệ giữa các channel.

Nếu chỉ một trong hai residual lớn, đó có thể là texture hoặc nhiễu. Harmonic mean buộc cả hai loại irreducibility cùng cao mới tạo hardness lớn.

### Cơ chế

Spatial reconstruction loại bỏ center pixel để tránh nghiệm sao chép identity. Channel reconstruction học cách dự đoán channel từ các channel còn lại. Hai residual là:

\[
R_s=X-\widehat X_s,
\qquad
R_c=X-\widehat X_c.
\]

Hardness:

\[
H=\frac{2|R_s||R_c|}{|R_s|+|R_c|+\epsilon}.
\]

Chỉ top-\(q\) location theo hardness được giữ lại bằng sparse gate. Module dự đoán offset và dùng Gaussian splatting để vận chuyển residual hữu ích tới vùng đích:

```text
dual residual → hardness → sparse gate
              → offsets → Gaussian transport → residual fusion
```

Offset và transport projection được zero-initialize, nên module bắt đầu gần identity.

### Supervision và ablation

- Reconstruction loss giám sát hai predictor không gian/channel.
- Offset target có thể hướng transport về location đối tượng dựa trên GT.
- Biến thể `no_transport` giữ hardness/reconstruction nhưng loại bỏ bước vận chuyển, giúp kiểm tra gain đến từ tín hiệu irreducibility hay từ transport thực sự.

HIT chạy trên inference path nếu transport được bật; vì vậy chi phí và latency phải được báo cáo cùng accuracy.

## 6. GCTS v1: Grid-Cell Target Selection

### Mục tiêu

Downsample P2 sang P3 có thể trộn bốn sub-cell vào cùng một cell và làm mất vị trí nội bộ của tiny object. GCTS v1 thay convolution P2→P3 bằng một downsampling có chọn lọc.

### Cơ chế

`pixel_unshuffle` tách mỗi vùng \(2\times2\) của P2 thành bốn candidate tương ứng top-left, top-right, bottom-left và bottom-right. Selector sinh phân phối:

\[
\pi=\operatorname{softmax}(z),
\]

và routed detail là:

\[
D=\sum_{q=1}^{4}\pi_q D_q.
\]

Output:

\[
P3=P3_{base}+\gamma\,\operatorname{Proj}(D).
\]

`gamma` được khởi tạo bằng 0 để giữ đường Conv pretrained tại thời điểm bắt đầu. Target selector được tạo từ vị trí GT trong cell, dùng one-hot hoặc bilinear weight. Auxiliary cross-entropy buộc selector chọn đúng sub-cell chứa thông tin mục tiêu.

GCTS v1 thay đổi feature P3 trong cả training và inference.

## 7. GCTS v2: head-local routing

GCTS v2 không nhất thiết thay P2→P3 trong neck. Nó nhận `[P2, P3, P4, P5]` tại head và dùng P2 để bổ sung riêng cho nhánh P3.

Hai đường chính:

- **Classification:** chọn content từ bốn P2 sub-cell rồi chiếu vào feature classification P3 bằng projection zero-init.
- **Regression:** dùng xác suất selector để tính tọa độ kỳ vọng \((\hat x,\hat y)\), entropy và confidence gate; các đại lượng này tạo positional correction cho regression P3.

Loss gồm KL divergence giữa selector và target phân bố không gian, cộng SmoothL1 cho vị trí kỳ vọng. Có thể dùng tiny-object gate để auxiliary loss chỉ tập trung vào GT đủ nhỏ.

GCTS v1 và v2 phải được xem là hai kiến trúc khác nhau: v1 sửa neck transition, còn v2 route thông tin P2 cục bộ vào head P3.

## 8. Non-uniform DFL

### Vấn đề của DFL đều

DFL chuẩn dùng support points:

\[
[0,1,2,\ldots,15].
\]

Tại P2 stride 4, sai số nhỏ ở các distance gần zero ảnh hưởng IoU của tiny box mạnh hơn sai số tương tự ở vùng distance lớn. Non-uniform DFL giữ nguyên 16 logits nhưng phân bố nhiều support point hơn gần zero, nên không tăng số channel của regression head.

Codebook P2 hiện tại:

```text
[0.00, 0.35, 0.70, 1.05, 1.40, 1.80, 2.30, 2.90,
 3.60, 4.50, 5.60, 6.90, 8.40, 10.20, 12.40, 15.00]
```

P3 NUDFL legacy dùng codebook khác:

```text
[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00,
 3.00, 4.00, 5.00, 7.00, 9.00, 11.00, 13.00, 15.00]
```

Hai biến thể không được gọi lẫn nhau.

### Target encoding

Với target distance \(t\), dùng `searchsorted` để tìm:

\[
b_l\le t<b_r.
\]

Weight nội suy theo khoảng cách vật lý giữa hai support point:

\[
w_r=\frac{t-b_l}{b_r-b_l},
\qquad
w_l=1-w_r.
\]

Loss:

\[
\mathcal L_{NUDFL}=-w_l\log p_l-w_r\log p_r.
\]

Decode cũng phải dùng codebook:

\[
\widehat t=\sum_k p_kb_k,
\]

không được dùng kỳ vọng theo bin index \(k\). Trong cấu hình P2 NUDFL, chỉ prediction thuộc level có stride nhỏ nhất dùng custom centers; P3-P5 tiếp tục dùng bins đều.

## 9. Pair-competitive DFL

NUDFL tăng resolution nhưng chưa buộc phân phối target pair thắng rõ các interval kế bên. Với target pair \((l,r)\), target score là:

\[
s_{tar}=\operatorname{LSE}
\left(z_l+\log(w_l+\epsilon),
z_r+\log(w_r+\epsilon)\right).
\]

Competitor lấy hai bin ngay ngoài target pair:

\[
\mathcal C=\{l-1,r+1\},
\]

sau khi bỏ index vượt biên. Pair-competitive loss:

\[
\mathcal L_{PC}=\operatorname{softplus}
\left(m+s_{cmp}-s_{tar}\right).
\]

Loss P2 đầy đủ:

\[
\mathcal L_{P2-DFL}=\mathcal L_{NUDFL}
+\lambda_{PC}\mathcal L_{PC}.
\]

Target pair của loss cạnh tranh và conflict score phải được tìm bằng cùng custom codebook. Dùng `floor/ceil` theo giá trị target sẽ quay lại semantics của uniform bins và làm conflict map sai.

## 10. CFR: Conflict-Guided Fine Reconstruction

### Cơ chế hiện tại

CFR nhận fused P2 và backbone P1. Trong training, decoder thực hiện:

```text
P2 → Conv 1×1 → upsample ×2 → DWConv 3×3 → Conv 1×1
   → reconstructed P1 feature + reconstructed P1 detail
```

Targets được detach:

\[
T_1=\operatorname{sg}(F_1),
\qquad
D_1=\operatorname{sg}\left(F_1-\operatorname{AvgPool}_{3\times3}(F_1)\right).
\]

Loss gồm weighted SmoothL1 cho full feature, weighted SmoothL1 cho detail và cosine loss.

Conflict score được lấy từ pair-competitive NUDFL trên bốn cạnh, scatter thành P2 map, kết hợp với TAL positive mask, mở rộng context và upsample lên P1:

\[
W_1=M_1(1+\eta C_1).
\]

Conflict được detach, nên reconstruction không tối ưu DFL bằng cách thao túng weight; gradient đi từ decoder về fused P2, neck và backbone.

### Giới hạn cần diễn đạt đúng

`ConflictFineReconstruction.forward()` luôn trả lại P2 gốc và decoder bị tắt khi inference. Vì vậy implementation hiện tại là **training-only auxiliary regularization**, không phải feature enhancer trực tiếp trên inference path. Nó khuyến khích P2 giữ thông tin có thể tái tạo P1, nhưng không đưa reconstructed detail trở lại Detect.

Việc tái tạo P1 cũng có nguy cơ giữ texture nền, sóng hoặc bờ biển. Conflict weighting giảm vấn đề này bằng cách tập trung loss quanh positive khó, nhưng không loại bỏ hoàn toàn rủi ro.

Ngoài ra, các cột `p2_positive_count` và `p2_positive_fraction` từng ghi 0 dù `loss_pc_dfl` và conflict khác 0. Đây là lỗi plumbing/logging counter của auxiliary loss, không phải bằng chứng TAL không có positive P2. Không dùng hai cột này để kết luận assignment trước khi sửa và kiểm chứng logger.

## 11. Method kết hợp P2 NUDFL-PC-CFR

Luồng đầy đủ:

```text
P1 ───────────────────────────────→ detached fine-detail target
                                      ↑
fused P2 → Detect P2 with NUDFL-PC    │
        └→ training-only CFR decoder ─┘

P3-P5 → Detect với uniform DFL
```

Ba vai trò tách biệt:

1. **P2 head** tăng mật độ spatial location.
2. **Non-uniform DFL** tăng resolution của distance gần zero.
3. **Pair competition** ép target interval thắng adjacent intervals và sinh conflict để weight CFR.
4. **CFR** tạo auxiliary gradient buộc fused P2 giữ thêm thông tin P1 quanh positive khó.

Objective:

\[
\mathcal L=
\mathcal L_{box}+
\mathcal L_{cls}+
\mathcal L_{NUDFL}+
\lambda_{PC}\mathcal L_{PC}+
\lambda_{CFR}\mathcal L_{CFR}.
\]

Thiết lập run đầu:

```yaml
pc_dfl_gain: 1.0
pc_dfl_margin: 1.5
cfr_gain: 2.0
cfr_detail_gain: 1.0
cfr_cos_gain: 1.0
cfr_conflict_weight: 3.0
p2_offset_regression: false
```

## 12. So sánh phạm vi can thiệp

| Method | Vị trí chính | Thay đổi inference path | Auxiliary supervision | Mục tiêu chính | Rủi ro chính |
|---|---|---:|---:|---|---|
| YOLO-P2 | Neck + Detect P2-P5 | Có | Không bắt buộc | Giữ spatial resolution | Tăng FLOPs/memory/candidates |
| P2 offset regression | Regression head P2 | Có | Không bắt buộc | Sampling riêng cho bốn cạnh | Offset bất ổn, tăng latency |
| DBSS | Feature map/transition đã chọn | Có | Tùy biến thể | Tách nền và nhấn residual | Background basis có thể chứa object |
| HIT | Feature transition | Có | Có | Tìm location dual-irreducible và transport detail | Transport nhiễu, chi phí splatting |
| GCTS v1 | P2→P3 neck transition | Có | Có | Chọn đúng sub-cell khi downsample | Selector collapse/sai target |
| GCTS v2 | P3 head, dùng candidate P2 | Có | Có | Content routing + positional correction | Head phức tạp, coupling cls/reg |
| NUDFL | P2 hoặc P3 regression semantics | Có, nhưng không thêm channel | DFL chính | Tăng distance resolution gần zero | Codebook lệch phân bố dữ liệu |
| Pair competition | DFL loss | Không thêm inference operator | Có | Làm target pair decisive | Margin quá mạnh gây overconfidence |
| CFR hiện tại | Auxiliary decoder P2→P1 | Không | Có | Regularize P2 giữ fine detail | Có thể học texture nền |

## 13. Quy tắc ablation và default

Để mỗi thí nghiệm chỉ đo đúng method được đặt tên:

- Tất cả custom module phải opt-in; default là `false` hoặc gain bằng 0.
- YOLO-P2 baseline không được tự bật P2 offset regression.
- NUDFL-PC-CFR mặc định phải tắt offset regression trừ khi tên run ghi rõ tổ hợp đó.
- Không so sánh DBSS full với DBSS routed như cùng một topology.
- GCTS v1, GCTS v2, P2 NUDFL và P3 NUDFL phải có slug/config riêng.
- Một run chỉ được xác nhận có module sau khi kiểm tra YAML đã resolve và model graph/log lúc khởi tạo, không chỉ dựa vào tên thư mục.

Các default custom đã được chuyển sang opt-in trong commit `93f7293`; checkpoint tạo trước thay đổi này phải được audit lại config trước khi dùng.

## 14. Bản đồ implementation

- `models_related/ultralytics/ultralytics/nn/modules/head.py`
  - `P2OffsetRegression`
  - `P2NUDFLDetect`
  - legacy P3 NUDFL detect
  - GCTS head-local detect
- `models_related/ultralytics/ultralytics/nn/modules/block.py`
  - `GCTS`
  - `DBSS`
  - `DualIrreducibilityHIT`
  - `ConflictFineReconstruction`
- `models_related/ultralytics/ultralytics/utils/loss.py`
  - non-uniform target encoding
  - pair-competitive loss và conflict score
  - P2 routing theo stride nhỏ nhất
- `models_related/models_config/yolov8/levir/`
  - YAML baseline P2 và các biến thể DBSS, HIT, GCTS, NUDFL-PC-CFR.

Khi cập nhật method, report này cần được đối chiếu trực tiếp với bốn nhóm file trên để tránh mô tả một ý tưởng chưa đúng với code thực thi.
