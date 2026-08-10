# Báo Cáo Chi Tiết: Kết Quả Phát Hiện & Sai Số Định Vị (Wise-IoU Model)

Báo cáo này phân tích chi tiết các kết quả phát hiện vật thể, các đối tượng bị bỏ sót (misses), và sai lệch biên hộp giới hạn (alignment errors) của mô hình **YOLOv8n P2 + Wise-IoU (WIoU)** trên tập dữ liệu thử nghiệm LEVIR-Ship (Test Split, 788 ảnh, 696 Ground Truth).

---

## 1. Thống Kê Tổng Quan (Tại ngưỡng Confidence = 0.25)

* **Tổng số đối tượng Ground Truth (GT):** **696**
* **Số đối tượng phát hiện được (IoU $\ge$ 0.5):** **547** (tỉ lệ **78.6%**)
* **Số đối tượng bị bỏ sót (Missed):** **149** (tỉ lệ **21.4%**)

---

## 2. Phân Tích Tỉ Lệ Bỏ Sót Theo Nhóm Kích Thước (Dataset-Wide Breakdown)

Thống kê số lượng vật thể và tỉ lệ bỏ sót thực tế trên toàn bộ tập dữ liệu (test split) chia theo các nhóm diện tích:

| Nhóm Kích Thước | Kích thước tương đương | Tổng số lượng trong Dataset | Số lượng phát hiện được | Số lượng bị bỏ sót | Tỉ lệ bỏ sót (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Siêu Nhỏ (Tiny <100 px²)** | $< 10 \times 10$ px | 14 | 0 | 14 | **100.00%** |
| **Rất Nhỏ (Very Small 100-400 px²)** | $10 \times 10 \rightarrow 20 \times 20$ px | 387 | 293 | 94 | **24.29%** |
| **Nhỏ-Trung Bình (400-1000 px²)** | $20 \times 20 \rightarrow 31 \times 31$ px | 274 | 236 | 38 | **13.87%** |
| **Lớn ($\ge$ 1000 px²)** | $\ge 31 \times 31$ px | 21 | 18 | 3 | **14.29%** |

---

## 3. Đặc Điểm Chi Tiết Của Nhóm Bị Bỏ Sót (Missed Objects)

* **Diện tích (Area):**
  * Diện tích trung vị (Median Area): **`272.0 px²`** (khoảng $16.5 \times 16.5$ pixel).
  * Diện tích trung bình (Mean Area): **`348.4 px²`**.
  * Phạm vi diện tích: từ **`25.5 px²`** ($5 \times 5$ pixel) đến **`2520.0 px²`** ($50 \times 50$ pixel).
* **Tỷ lệ hình học (Aspect Ratio W/H):**
  * Trung vị: **`0.95`** (vật thể có hình dạng gần vuông).
* **Phân bố các trường hợp bị bỏ sót:**
  * Chiếm tỉ lệ lớn nhất là nhóm **Rất Nhỏ (100-400 px²)** với **94/149 đối tượng bị sót (63.1%)**.
  * Nhóm **Siêu Nhỏ (<100 px²)** dù bị sót 100% nhưng số lượng trong dataset rất ít (chỉ 14 đối tượng, tương đương **2.0%** dữ liệu).

---

## 4. Phân Tích Phân Bố Không Gian & Mật Độ Trên Ảnh Của Vật Thể Bị Sót

Phân tích bối cảnh ảnh (image-level context) và vị trí địa lý của các GT bị bỏ sót so với các GT phát hiện được:

### Mật độ vật thể trong cùng một ảnh (Object Density)
* **Số lượng GT trung bình trong ảnh chứa vật thể bị sót:** **`7.67 vật thể/ảnh`** (Trung vị: `2.0`).
* **Số lượng GT trung bình trong ảnh chứa vật thể phát hiện được:** **`4.32 vật thể/ảnh`** (Trung vị: `2.0`).
* *Nhận xét:* Vật thể bị sót có xu hướng nằm ở các bức ảnh có **mật độ tàu thuyền dày đặc hơn gần gấp đôi** (ví dụ: các khu neo đậu, cảng biển bận rộn) so với các ảnh đơn giản ngoài khơi.

### Mức độ đứng gần nhau (Clustered vs Isolated)
* **Khoảng cách trung vị tới tàu thuyền gần nhất (Proximity to nearest neighbor):**
  * Đối với các vật thể bị sót: **`85.5 pixel`** (Trung bình: `111.1px`).
  * Đối với các vật thể phát hiện được: **`131.3 pixel`** (Trung bình: `167.3px`).
* **Tỉ lệ vật thể đứng cô lập (chỉ có 1 tàu duy nhất trong ảnh):**
  * Tỉ lệ cô lập ở nhóm bị sót: **36.2%** (54/149 đối tượng). Có tới **63.8%** nằm ở khu vực có cụm tàu.
  * Tỉ lệ cô lập ở nhóm phát hiện được: **43.3%** (237/547 đối tượng).
* *Nhận xét:* Vật thể bị bỏ sót có khoảng cách tới tàu gần nhất ngắn hơn đáng kể (85.5px so với 131.3px), khẳng định mô hình WIoU gặp khó khăn trong việc tách và phát hiện các tàu nằm sát nhau trong các cụm (clusters).

### Ảnh hưởng của biên ảnh (Border Effect)
* **Vị trí trung tâm trung bình:** $X_c = 0.480, Y_c = 0.503$ (Không bị lệch lệch hướng cụ thể nào).
* **Tỉ lệ vật thể nằm quá sát biên ảnh (cách biên dưới 10% chiều dài/rộng ảnh):** **`32.2%`** (48/149 đối tượng bị sót nằm ở biên).
* *Nhận xét:* Hơn 1/3 số lượng tàu bị bỏ sót nằm ở rìa bức ảnh. Việc vật thể bị cắt một phần do biên ảnh làm mất đi các đặc trưng đầy đủ của CNN, cộng với ảnh hưởng của phần đệm (padding) trong quá trình Letterbox khiến tỉ lệ bỏ sót ở rìa tăng cao.


---

## 5. Phân Bố Chất Lượng Hộp Phát Hiện (IoU Distribution)

Đối với **547 đối tượng** phát hiện thành công:

* **Trung vị IoU (Median IoU):** **`0.711`** (71.1% trùng khớp diện tích).
* **Trung bình IoU (Mean IoU):** **`0.705`** (độ lệch chuẩn $\sigma = 0.097$).
* **Ranh giới phân chia IoU:**
  * Tỉ lệ đạt IoU $\ge$ 0.75 (ngưỡng AP75): **34.4%** (188 đối tượng).
  * Tỉ lệ đạt IoU $\ge$ 0.85 (định vị siêu chính xác): **7.7%** (42 đối tượng).

---

## 6. Phân Tích Sai Số Định Vị Hộp (Alignment & Box Errors)

Thống kê sai lệch về vị trí tâm và kích thước hộp giới hạn của mô hình WIoU so với GT thực tế:

### Sai lệch vị trí tâm (Center Shifts)
* **Độ lệch tâm trung vị (Median Center Shift):** **`1.59 pixel`**.
* **Độ lệch tâm trung bình (Mean Center Shift):** **`1.85 pixel`**.
* **Độ lệch trung bình theo trục X (Mean X Shift):** **`+0.58 pixel`** (lệch nhẹ về bên phải).
* **Độ lệch trung bình theo trục Y (Mean Y Shift):** **`+0.04 pixel`** (hầu như không lệch theo chiều dọc).

### Sai lệch kích thước hộp (Dimension Errors)
* **Sai số tuyệt đối chiều rộng trung bình (Mean Absolute Width Error):** **`3.25 pixel`**.
* **Sai số tuyệt đối chiều cao trung bình (Mean Absolute Height Error):** **`2.72 pixel`**.
* **Sai số có dấu trung bình (Mean Signed Error):**
  * Chiều rộng: **`-0.14 pixel`** (hộp dự đoán có xu hướng hẹp hơn một chút so với GT).
  * Chiều cao: **`-0.32 pixel`** (hộp dự đoán có xu hướng ngắn hơn một chút so với GT).

---


---

## 8. Phân Tích Thực Nghiệm Độ Recall Sát Biên Ảnh & Chất Lượng TAL

Để làm rõ nguyên nhân của lỗi bỏ sót ở sát biên ảnh, chúng tôi đã tiến hành thêm 3 nghiên cứu thực nghiệm trực tiếp trên Marimo server sử dụng mô hình Baseline P2.

### 8.1. Chất Lượng Gán Nhãn TAL & Phân Tách Biên Ảnh (Touching vs Fully-Visible)

Chúng tôi chia các đối tượng Ground Truth trong tập kiểm thử thành 3 nhóm vị trí:
1. **CENTER**: Đối tượng nằm cách biên ảnh $> 16$ pixel.
2. **BORDER FULLY VISIBLE**: Đối tượng nằm cách biên $\le 16$ pixel nhưng không chạm rìa ảnh.
3. **BORDER TOUCHING**: Đối tượng chạm rìa ảnh (có tọa độ biên $\le 1.5$ pixel hoặc $\ge W - 1.5$).

Kết quả đo đạc chất lượng gán nhãn TAL (Task-Aligned Assigner) và Recall thực tế:

| Vị trí / Trạng thái | Nhóm kích thước | Số lượng GT | Recall (%) | $S_{\text{TAL}}$ (Sum Score) | Max assigned score | Mean IoU với GT |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CENTER** | Small | 158 | **68.35%** | 4.4212 | 0.7738 | 0.6861 |
| | Medium | 297 | **84.51%** | 4.9729 | 0.8011 | 0.7299 |
| | Large | 111 | **90.09%** | 5.0318 | 0.8275 | 0.7513 |
| **BORDER FULLY VISIBLE** | Small | 25 | **72.00%** | 4.1988 | 0.7533 | 0.6730 |
| | Medium | 32 | **75.00%** | 4.6925 | 0.8034 | 0.7217 |
| | Large | 14 | **92.86%** | 4.9336 | 0.8275 | 0.7649 |

| **BORDER TOUCHING** | Small | 14 | **50.00%** | 3.9866 | 0.7782 | 0.6785 |
| | Medium | 31 | **83.87%** | 4.8415 | 0.7969 | 0.7029 |
| | Large | 5 | **100.00%** | 4.7988 | 0.7801 | 0.6634 |

#### Phân Tích Chi Tiết 14 Vật Thể Small Touching-Border Bị Bỏ Sót:
Để hiểu rõ nguyên nhân sụt giảm recall ở nhóm chạm rìa (touching), chúng tôi đã trích xuất chi tiết kết quả dự đoán của từng đối tượng trong số 14 đối tượng này trước NMS:

- **Nhóm Phát hiện thành công (7/14 - 50.0%)**: Đạt Conf từ `0.30` đến `0.58` với IoU cao.
- **Nhóm Bỏ sót do lệch phân phối phân loại (5/14 - 35.7%)**:
  * Các candidate tương ứng có **IoU cực kỳ tốt với GT** (`0.700`, `0.709`, `0.718`, `0.790`, `0.847`) nhưng mức độ tự tin (Confidence score) bị tụt nhẹ xuống dưới ngưỡng 0.25 (`0.206`, `0.183`, `0.135`, `0.094`, `0.055`).
- **Nhóm Bỏ sót hoàn toàn (2/14 - 14.3%)**: Không tìm thấy candidate nào có Conf > 0.05.

**Kết luận quan trọng:**
1. **Regression không phải điểm nghẽn**: Không có bất kỳ trường hợp nào bị bỏ sót do lệch hộp giới hạn (IoU luôn rất cao đối với các candidate tốt nhất). Đầu regression hoạt động cực kỳ ổn định bất chấp việc vật thể chạm biên.
2. **Điểm nghẽn nằm ở Classification**: Việc bị che khuất/cắt cụt một phía (partial clipping) làm thay đổi đặc trưng ngữ cảnh/pha cục bộ đi vào đầu phân loại, làm giảm nhẹ điểm số Confidence xuống dưới ngưỡng 0.25, biến các detection chất lượng cao thành các trường hợp bỏ sót (miss).


### 8.2. Kiểm Chứng Mất Ổn Định Pha & Độ Bền Định Vị (Standardized Controlled Translation Test)

Để cô lập hoàn toàn các yếu tố gây lệch pha và biến động đặc trưng, chúng tôi thiết lập phép thử dịch chuyển có kiểm soát (standardized control) trên 123 ảnh đơn vật thể phát hiện đúng ở trung tâm, tạo ra 3 biến thể dịch chuyển nhân tạo trên cùng một ảnh nền sea-background cố định:

1. **CENTER_SHIFT**: Dịch chuyển tịnh tiến vật thể trong vùng trung tâm (làm mẫu đối chứng dịch chuyển).
2. **BORDER_VISIBLE**: Dịch chuyển vật thể sát rìa ảnh nhưng giữ nguyên vẹn (Fully-Visible).
3. **BORDER_TOUCHING**: Dịch chuyển vật thể vượt rìa ảnh 16 pixel (Clipped / Touching).

Kết quả đo đạc trung bình trên P2 feature map:

* **CENTER_SHIFT**:
  * Cosine Similarity: **`0.9999`**
  * Confidence Drop: **`-0.0001`**
  * IoU Drop: **`-0.0017`**
* **BORDER_VISIBLE**:
  * Cosine Similarity: **`0.9932`**
  * Confidence Drop: **`0.0015`**
  * IoU Drop: **`0.0078`**
* **BORDER_TOUCHING**:
  * Cosine Similarity: **`0.2779`**
  * Confidence Drop: **`0.0000`**
  * IoU Drop: **`-0.0092`**

**Nhận xét:**
- Phép dịch chuyển trong trung tâm và sát biên (Fully-Visible) giữ được độ tương đồng đặc trưng hoàn hảo ($>0.99$ cosine similarity) và hầu như không làm sụt giảm chất lượng hộp hay độ tự tin (Confidence / IoU drop $\approx 0.00$). Điều này chứng minh CNN có tính bất biến dịch chuyển (translation equivariance) rất cao và khoảng cách tới biên đơn thuần không gây mất ổn định pha (phase instability).
- Khi vật thể chạm rìa và bị cắt cụt (BORDER_TOUCHING), vector đặc trưng bị biến dạng mạnh (cosine similarity giảm sâu xuống **`0.2779`**). Tuy nhiên, độ sụt giảm confidence và IoU của các vật thể này vẫn bằng **0.00**.
- **Kết luận**: Hiện tượng xoay/lệch pha đặc trưng khi chạm biên *không phải* nguyên nhân gây bỏ sót (miss). Đầu Detect của mô hình cực kỳ bền bỉ trước sự xoay đặc trưng này. Điểm nghẽn bỏ sót thực tế nằm ở **vấn đề quan sát một phần (partial-observation difficulty)**: khi các vật thể nhỏ chạm biên bị cắt bớt quá nhiều thông tin chi tiết, lượng điểm ảnh còn lại không đủ cấu trúc ngữ cảnh để classifier nhận diện.

### 8.3. Thống kê Chi tiết Bỏ sót (False Negatives) trên nhóm Small & Large của YOLOv8n P2 Baseline vs P1-DRR

Để làm rõ nguyên nhân bỏ sót ngoài nhóm Siêu Nhỏ (Tiny < 100 px²), chúng tôi chạy kiểm chứng định lượng trên tập kiểm thử (Test split, 788 ảnh) đối với mô hình YOLOv8n P2 Baseline và mô hình sử dụng P1-DRR (Cơ chế cổng giải cứu chi tiết và restraint loss):

* **Nhóm SMALL (100–400 px²)**: 
  * Baseline bỏ sót **68 vật thể** (Recall đạt **65.48%**).
  * P1-DRR giảm số lượng bỏ sót xuống **61 vật thể** (Recall tăng lên **69.04%**, tức cứu thêm 7 vật thể nhỏ).
  * Dải diện tích bị bỏ sót của nhóm này tập trung từ **106.5 px² đến 385.9 px²** (trung bình **262.4 px²**, tương đương kích thước khoảng $16\times16$ pixels).
* **Nhóm LARGE (> 400 px²)**:
  * Baseline bỏ sót **67 vật thể** (Recall đạt **86.33%**).
  * P1-DRR giảm số lượng bỏ sót xuống **63 vật thể** (Recall tăng lên **87.14%**).
  * Dải diện tích bị bỏ sót của nhóm này tập trung từ **400.0 px² đến 3937.5 px²** (trung bình **788.9 px²**, tương đương kích thước khoảng $28\times28$ pixels).

#### Phân bố ảnh lỗi nhiều nhất (Top FN Images):
1. **`GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png`** (Bỏ sót **19 vật thể**): Đây là khu cảng neo đậu tàu bận rộn với mật độ tàu cực dày đặc. Do các tàu neo sát cạnh nhau, NMS hoặc Assigner gộp nhãn dẫn đến bỏ sót hàng loạt.
2. **`GF1_WFV1_E120.0_N36.3_20200423_L2A0004760887_10240_9216.png`** (Bỏ sót **5 vật thể**).
3. **`GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png`** (Bỏ sót **5 vật thể**).
4. **`GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png`** (Bỏ sót **4 vật thể**).
5. **`GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png`** (Bỏ sót **3 vật thể**).

#### Khoảng cách mật độ (Proximity Check):
* Phân tích cho thấy **15.97%** số FNs ở P2 Baseline (23/144 FN) và **17.29%** số FNs ở P1-DRR (23/133 FN) nằm ở khoảng cách rất gần (**< 40 pixels**) so với một vật thể khác trên ảnh, khẳng định sự chồng lấn trong cụm (dense port) vẫn là một trong các nguyên nhân chính gây bỏ sót.

#### Phân Tích Cơ Chế Lỗi (Classification vs Localization Failures):
Chúng tôi tiến hành chẩn đoán sâu 135 đối tượng bị bỏ sót (nhóm SMALL + LARGE) của mô hình Baseline nhằm xác định lỗi thuộc về đầu định vị (Regression) hay đầu phân loại (Classification):
* **Lỗi Phân loại (Classification Failures): 85.19%** (115 / 135 vật thể).
  * *Chi tiết*: Đây là các vật thể **được định vị chính xác** (có đề xuất bounding box đạt IoU $\ge$ 0.3 với GT) nhưng bị bỏ sót vì **điểm Confidence thấp hơn ngưỡng 0.25** (Conf trung bình của các candidate này chỉ đạt **0.1042**, cao nhất là **0.2430**).
* **Lỗi Định vị (Localization Failures): 14.81%** (20 / 135 vật thể).
  * *Chi tiết*: Mô hình thực sự không tìm thấy hoặc không đưa ra được bất kỳ bounding box nào chồng lấn tốt với GT (IoU < 0.3).

**Nguyên nhân cụ thể gây ra lỗi Classification ở các vật thể bị sót:**
1. **Sự thiếu hụt ngữ cảnh ngữ nghĩa (Semantic Context)**: Các vật thể nhỏ hoặc rất nhỏ không có đủ cấu trúc chi tiết bên trong (như cabin, boong tàu) và thông tin bối cảnh. Dù đầu hồi quy (Regression head) định vị dựa trên biên độ đặc trưng biên rất tốt, đầu phân loại (Classification head) vẫn đánh giá độ tự tin thấp vì bối cảnh xung quanh quá đơn giản (chỉ có nước biển).
2. **Sự tương đồng về tần số với nhiễu nền (Sea Clutter Similarity)**: Tần số không gian của sóng biển/bọt nước lân cận có tính chất tương tự một phần với các tàu cá nhỏ ngoài khơi xa. Để kiểm soát False Positives (FP) không bùng nổ, classifier bắt buộc phải siết chặt phân phối xác suất (đưa conf về mức rất thấp ~0.10) ở những khu vực không có ngữ nghĩa rõ ràng, vô tình triệt tiêu luôn cả các vật thể thực sự.
3. **Bài toán gán nhãn TAL trong quá trình huấn luyện**: Vì vật thể nhỏ, chỉ cần bounding box dự đoán lệch nhẹ 1-2 pixel là IoU rụng về 0. Task-Aligned Assigner (TAL) của YOLOv8n chọn positive anchor dựa trên tích số `Score * IoU`. Khi IoU sụt giảm mạnh do kích thước nhỏ, TAL sẽ gán các anchor này là background, khiến classification head của mô hình bị huấn luyện để triệt tiêu độ tự tin tại các vị trí này.

## 9. Kết Luận & Định Hướng Cải Tiến

### 9.1. Thống kê Phân bổ Mẫu Gán (Target Assignment) theo Tầng Đặc trưng

Để hiểu rõ mức độ đóng góp của từng tầng Scale (P2/P3/P4/P5) đối với các vật thể nhỏ trong LEVIR-Ship, chúng tôi đã trích xuất kết quả gán nhãn thực tế từ **Task-Aligned Assigner (TAL)** trên tập test (tổng số **6,913 positive anchors** được gán):

* **Stride 4 (Tầng P2 - $1/4$ độ phân giải ảnh gốc): 6,878 anchors (chiếm 99.49%)**
* **Stride 8 (Tầng P3 - $1/8$ độ phân giải ảnh gốc): 35 anchors (chiếm 0.51%)**
* **Stride 16 (Tầng P4): 0 anchors (0.00%)**
* **Stride 32 (Tầng P5): 0 anchors (0.00%)**

> [!WARNING]
> Kết quả định lượng này cho thấy **99.5%** vật thể được gán hoàn toàn vào tầng độ phân giải cao nhất **P2**. Tầng P3 chỉ xử lý 0.51% lượng anchor (35 anchors trên toàn bộ tập test), trong khi P4 và P5 hoàn toàn trống rỗng.

---

### 9.2. Định Hướng Cải Tiến

1. **Chuyển dịch sang cấu trúc P2-Only (Bỏ qua P3/P4/P5)**:
   * Vì P3 chỉ đóng góp 0.51% lượng mẫu gán, toàn bộ các nhánh top-down/bottom-up PAN liên quan tới P3, P4, P5 đang gây lãng phí tham số và FLOPs nghiêm trọng.
   * **Khuyến nghị**: Thiết kế mô hình cực đoan **YOLOv8n-P2-Only** (chỉ giữ lại 1 đầu Detect tại P2 Stride 4). Điều này giúp thu gọn mạng, đẩy nhanh tốc độ suy luận và tối ưu hóa 100% tài nguyên biểu diễn của backbone vào tầng mịn.
2. **Tập trung cơ chế Attention vào duy nhất P2**:
   * Khi tích hợp các module Attention (như KVCA, CBAM, Coordinate Attention) để giải quyết lỗi phân loại (Classification), ta **chỉ nên áp dụng trên nhánh đặc trưng P2**. Việc chia sẻ Attention lên P3 là không cần thiết.
3. **Cơ hội từ Box Consensus**:
   * Box Consensus giúp co cụm và giảm phương sai của các anchors quanh vùng biên. Khi kết hợp với WIoU, Consensus có thể giúp mô hình giữ vững khả năng tìm kiếm diện rộng của WIoU, đồng thời tinh chỉnh (refine) sai lệch kích thước tuyệt đối $2.7 - 3.2$ pixel này về dưới $1.0$ pixel, cải thiện đáng kể mAP75.

---

## 10. Nghiên cứu Cơ chế Channel Attention bằng Phép Can thiệp (Channel Gate Interventions)

Để làm rõ nguồn gốc cải thiện nhất quán của cơ chế `ChannelAttention` (bản GAP - Average pooling) đối với độ chính xác định vị và phân loại, chúng tôi đã tiến hành chẩn đoán sâu thông qua phép can thiệp ma trận (Inference Intervention Matrix) trực tiếp trên checkpoint đã huấn luyện (seed 42, test split 788 ảnh, NMS IoU = 0.50). 

Bằng cách thay thế trọng số attention của từng kênh kích hoạt $g_{i,c}$ bằng các cấu hình can thiệp tĩnh/vô hướng trong quá trình suy luận, chúng tôi thu được kết quả định lượng như sau:

| Phương pháp can thiệp | Mô tả toán học | Test AP50 | Test AP75 | Test mAP50-95 | Test Precision | Test Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** | Không can thiệp ($g_{i,c} = \text{act}(gate)$) | 0.8162 | 0.1305 | 0.3106 | 0.8266 | 0.7701 |
| **Static-channel** | Lấy trung bình ảnh: $g'_{i,c} = \text{mean}_i(g_{i,c})$ | 0.8229 | 0.1357 | 0.3198 | 0.8148 | **0.7839** |
| **Dynamic-scalar** | Chỉ giữ scale của ảnh: $g'_{i,c} = \text{mean}_c(g_{i,c})$ | **0.8329** | **0.1421** | **0.3254** | 0.8409 | 0.7802 |
| **Global-scalar** | Trọng số hằng số: $g' = \text{mean}_{i,c}(g_{i,c})$ | 0.8319 | 0.1382 | 0.3241 | **0.8431** | 0.7718 |
| **Cross-image shuffle**| Phá sự tương ứng ảnh: $g'_{i,c} = g_{j,c}$ | 0.8095 | 0.1244 | 0.3084 | 0.8242 | 0.7748 |
| **Channel shuffle** | Phá tương quan kênh: permute $g_{i,c}$ theo $c$ | 0.8123 | 0.1259 | 0.3057 | 0.8352 | 0.7572 |

### Kết luận cơ chế quan trọng:

1. **Sự sụt giảm do Channel-specific Gating (Channel Distortion)**:
   * Mô hình nguyên bản (**Original**) có kết quả **tệ hơn đáng kể** ở mọi chỉ số so với việc ép tất cả các kênh chia sẻ chung một tỷ lệ scale (**`Dynamic-scalar`** và **`Global-scalar`**). 
   * Cụ thể, việc đồng nhất hóa trọng số kênh theo từng ảnh (`Dynamic-scalar`) giúp **AP50 vọt lên 0.8329 (+1.67%) và AP75 lên 0.1421 (+1.16%)**.
2. **Bản chất cơ chế là Adaptive Feature Amplitude Calibration**:
   * Việc `Dynamic-scalar` và `Global-scalar` đạt hiệu suất cao nhất chứng minh rằng giá trị thực tế của `ChannelAttention` không nằm ở việc "chọn lọc đặc trưng ngữ nghĩa kênh" (semantic channel selection), mà chỉ đóng vai trò như một **bộ hiệu chuẩn biên độ kích hoạt toàn cục** (global activation amplitude calibrator) cho feature map P2.
   * Việc áp đặt các trọng số attention khác nhau lên từng kênh (Original) vô tình làm méo cấu trúc đồng thích ứng kênh (channel co-adaptation) đã được tối ưu hóa ở các lớp tích chập phía trước, gây tác động tiêu cực đến downstream Detect head.
3. **Khuyến nghị kiến trúc tiếp theo**:
   * Các nghiên cứu attention tiếp theo nên tập trung vào các cơ chế hiệu chuẩn biên độ vô hướng (scalar-amplitude calibration) đơn giản như **`DynamicScalarScale`** thay vì các khối attention phức tạp biến đổi kênh riêng biệt.


