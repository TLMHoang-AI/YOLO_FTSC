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

## 7. Kết Luận & Định Hướng Cải Tiến


1. **WIoU chịu ảnh hưởng nặng từ kích thước:**
   * Tỉ lệ bỏ sót nhóm **Siêu Nhỏ (<10x10 px)** là **100%**. Đây là giới hạn vật lý của feature map khi không được tối ưu đặc trưng cục bộ mịn.
   * Nhóm **Rất Nhỏ (10x10 đến 20x20 px)** là nhóm phổ biến nhất trong dataset (chiếm 55.6% tổng số GT) nhưng tỉ lệ bỏ sót vẫn còn cao ở mức **24.29%**.
2. **Sai lệch hộp giới hạn thấp:**
   * Sai số lệch tâm trung bình chỉ **`1.85 pixel`** và sai số kích thước tuyệt đối khoảng **`2.7 - 3.2 pixel`** cho thấy mô hình Wise-IoU định vị tương đối ổn định sau khi đã tìm ra vật thể.
3. **Cơ hội từ Box Consensus:**
   * Box Consensus giúp co cụm và giảm phương sai của các anchors quanh vùng biên. Khi kết hợp với WIoU, Consensus có thể giúp mô hình giữ vững khả năng tìm kiếm diện rộng của WIoU, đồng thời tinh chỉnh (refine) sai lệch kích thước tuyệt đối $2.7 - 3.2$ pixel này về dưới $1.0$ pixel, cải thiện đáng kể mAP75.
