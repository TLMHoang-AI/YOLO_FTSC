# LEVIR-Ship Bounding Box Bottleneck Diagnostics Report

Bản báo cáo này tổng hợp kết quả của 4 probe chẩn đoán (diagnostics) chạy trực tiếp trên checkpoint mô hình tốt nhất của baseline (best.pt - seed 42) chạy validation trên tập kiểm thử (test split) của LEVIR-Ship.

---

## 1. Probe 1: One-Pixel Translation Equivariance Test
Mục tiêu là kiểm chứng mô hình có đạt tính bất biến tịnh tiến sub-pixel hay không khi dịch chuyển ảnh đúng 1 pixel theo các hướng.

- **Độ lệch tâm trung bình (\Delta c):** `1.0525` pixel
- **Độ lệch chiều rộng trung bình (\Delta w):** `1.8971` pixel
- **Độ lệch chiều cao trung bình (\Delta h):** `1.4103` pixel
- **Độ tương quan IoU giữa nguyên bản và dịch chuyển (\text{IoU}_{eq}):** `82.12%`

*Nhận xét:* Độ lệch tâm $\Delta c \approx 1.0$ cho thấy việc dịch tâm di chuyển tịnh tiến tương đối đồng đều theo dịch chuyển của ảnh. Tuy nhiên, kích thước chiều rộng/cao biến thiên mạnh từ 1.4 đến 1.9 pixel khi chỉ dịch chuyển 1 pixel. Điều này cho thấy bounding box shape khá nhạy cảm với pha/subpixel shift.

---

## 2. Probe 2: Raw-Candidate Oracle Gap
Mục tiêu là kiểm tra xem classifier chọn score đúng hay sai candidate có bounding box tốt nhất quanh GT.

- **Oracle Gap trung bình (\Delta_{\text{oracle}}):** `0.1539` (Chênh lệch IoU tối đa của candidate tốt nhất so với candidate có score cao nhất)
- **Oracle Gap lớn nhất:** `0.7756`

*Nhận xét:* Oracle Gap trung bình đạt `0.1539` (chênh lệch rất lớn). Điều này khẳng định regression head thực tế đã sinh ra hộp bao quanh rất tốt (IoU cao), nhưng classification score đang chọn sai candidate tối ưu, gây suy giảm mAP.

---

## 3. Probe 3: DFL Uncertainty–Error Correlation
Mục tiêu là đo lường xem Entropy/Variance của phân phối DFL có đồng biến với sai số định vị thực tế của các cạnh hay không.

Hệ số tương quan Spearman giữa Độ bất định và Sai số định vị (pixels):

| Cạnh | Tương quan với Entropy | Tương quan với Variance |
| :--- | :---: | :---: |
| **Trái (l)** | `0.2060` | `0.2108` |
| **Trên (t)** | `0.2687` | `0.2624` |
| **Phải (r)** | `0.0956` | `0.0960` |
| **Dưới (b)** | `0.1414` | `0.1405` |

*Nhận xét:* Hệ số tương quan Spearman dương nhưng ở mức thấp đến trung bình thấp (hầu hết < 0.25). Độ bất định (uncertainty) của phân phối DFL phản ánh một phần sai số nhưng chưa đủ mạnh để làm tín hiệu dẫn đường chính xác cho việc hiệu chỉnh biên trực tiếp.

---

## 4. Probe 4: Edge-Error Decomposition
Phân tích sai số định vị thành phần để xác định mô hình bị lệch tâm (translation) hay lệch kích cỡ (scale).

| Thành phần sai số | Giá trị trung bình (Signed Mean) | Sai lệch chuẩn (Std) | Sai số tuyệt đối trung bình (MAE) |
| :--- | :---: | :---: | :---: |
| **Lệch tâm X (\epsilon_{cx})** | `0.2285` px | `1.0460` px | `0.8034` px |
| **Lệch tâm Y (\epsilon_{cy})** | `-0.0820` px | `1.1594` px | `0.8462` px |
| **Lệch Chiều rộng (\epsilon_w)** | `0.4994` px | `2.2910` px | `1.7014` px |
| **Lệch Chiều cao (\epsilon_h)** | `0.1781` px | `2.5409` px | `1.8461` px |

*Nhận xét:* 
- Sai số tuyệt đối trung bình (MAE) của việc lệch tâm X/Y (~0.8 px) nhỏ hơn nhiều so với sai số MAE của chiều rộng/chiều cao (~1.7-1.8 px). Điều này cho thấy mô hình xác định tâm vật thể tương đối ổn định, nhưng dự đoán kích cỡ (width/height) lại có phương sai rất lớn.
- Signed Mean của chiều rộng (+0.49 px) và chiều cao (+0.17 px) mang giá trị dương, cho thấy mô hình có xu hướng dự đoán kích thước box hơi to hơn so với Ground Truth một cách hệ thống.

---

## 5. Phân tích bổ sung: Đặc trưng Tần số không gian (Spatial Frequency Analysis)
Để kiểm chứng mối liên hệ giữa vật thể (ships) và sự bất ổn định kích thước dưới dịch chuyển pixel (nhận xét từ Probe 1), chúng tôi thực hiện phân tích đặc trưng tần số không gian (Laplacian Variance và Gradient Magnitude) trên **3,219 nhãn đối tượng** trên toàn bộ tập dữ liệu (Train + Val + Test).

### Kết quả so sánh định lượng:

| Vùng Ảnh | Laplacian Variance (Độ sắc nét / Tần số cao) | Gradient Magnitude (Cường độ biên) |
| :--- | :---: | :---: |
| **Ground Truth (Tàu)** | **25.6681** | **15.3428** |
| **Random Background** | **9.6547** (~2.66x lower) | **6.4755** (~2.37x lower) |
| **Adjacent Background** (Nền sát cạnh) | **5.9865** (~4.29x lower) | **4.8861** (~3.14x lower) |

### Biểu đồ phân phối tần số (GT vs. Adjacent BG):
![Phân phối tần số không gian](/home/duylearch/.gemini/antigravity-ide/brain/6c51aafe-02a0-491a-b18e-1ec9a07d1bec/gt_vs_adjacent_bg_frequency.png)

*Nhận xét*: 
- Vùng chứa tàu (GT) có các thành phần tần số cao vượt trội hoàn toàn (gấp hơn 4 lần so với nước biển xung quanh).
- Điều này khẳng định tàu là các "điểm dị biệt tần số cao" trên nền đại dương tần số thấp. Khi mô hình downsampling qua các lớp tích chập, các chi tiết tần số cao này bị răng cưa/mất pha trầm trọng, trực tiếp gây ra sự thiếu bất biến tịnh tiến kích thước ở Probe 1.
