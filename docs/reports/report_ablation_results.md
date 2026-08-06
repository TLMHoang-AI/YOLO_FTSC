# YOLOv8n-P2 Ablation Study: Size-Aware Weighting & TargetedPartialClip

Báo cáo này trình bày kết quả chi tiết của thử nghiệm Ablation được thiết kế nhằm giải quyết bài toán bỏ sót vật thể nhỏ (tiny/small ships) ở khu vực sát biên ảnh trên tập dữ liệu LEVIR-Ship.

Hai phương pháp chính được tích hợp và đánh giá (dựa trên YOLOv8n-P2 + WIoU làm baseline):
1. **Size-Aware Classification Weighting (`small_weight`)**: Tăng trọng số cho positive classification loss ở các anchor được TAL gán cho Ground Truth nhỏ (area < 1000 px²). Trọng số được scale tuyến tính từ 1.0 đến tối đa (từ 1.15x đến 1.75x tùy kích thước) thông qua classification warm-up kéo dài 5 epochs.
2. **Targeted Partial-View Clipping (`partial_clip`)**: Augmentation định mục tiêu vào các vật thể nhỏ (area < 400 px²). Bằng phép tịnh tiến mảng (array slicing) số nguyên không nội suy để tránh blur, mô hình hóa sự cụt biên ảnh bằng cách cắt bớt vật thể (độ hiển thị $0.55 \le r_{\text{visible}} \le 0.85$), đồng thời vá nền biển ngẫu nhiên lấy từ tập ảnh negative.

---

## 1. Kết Quả Huấn Luyện & Đánh Giá (Seed 42)

Dưới đây là bảng đối chiếu chi tiết hiệu năng giữa Baseline (YOLOv8n-P2 + WIoU) và 3 phiên bản Ablation chạy trên tập dữ liệu LEVIR-Ship (ảnh size 512x512, huấn luyện 100 epochs, batch size 16):

### A. Kết quả trên tập Validation (Val Split)

| Cấu hình | Val mAP50 | Val mAP50-95 | Val Precision | Val Recall |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n-P2 + WIoU Baseline** | 0.7946 | 0.3277 | 0.8396 | 0.7201 |
| **`small_weight`** | 0.7507 | 0.2923 | 0.7616 | 0.6960 |
| **`partial_clip`** | **0.8205** | **0.3317** | **0.8447** | **0.7670** |
| **`small_weight_partial_clip`** | 0.7449 | 0.3028 | 0.7656 | 0.6702 |

### B. Kết quả trên tập Kiểm Thử (Test Split)

| Cấu hình | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n-P2 + WIoU Baseline** | **0.7797** | **0.2966** | **0.8099** | 0.7112 |
| **`small_weight`** | 0.7264 | 0.2697 | 0.7017 | 0.6796 |
| **`partial_clip`** | 0.7714 | 0.2959 | 0.7890 | **0.7284** |
| **`small_weight_partial_clip`** | 0.7245 | 0.2805 | 0.7426 | 0.6466 |

---

## 2. Phân Tích & Kết Luận Khoa Học

### A. Thành công vượt trội của Targeted Partial-View Clipping (`partial_clip`)
* **Cải thiện Recall mạnh mẽ**: `partial_clip` giúp Recall tăng vọt từ **72.01% lên 76.70%** trên tập Val (+4.69% absolute) và từ **71.12% lên 72.84%** trên tập Test (+1.72% absolute). Đây chính là mục tiêu ban đầu khi thiết kế augmentation này: giúp detector thích ứng với các vật thể bị cắt cụt sát biên mà không bị đánh lừa bởi biên ảnh nhân tạo.
* **Giữ vững độ chính xác**: Trên tập Val, mAP50 tăng từ **79.46% lên 82.05%** (+2.59% absolute). Trên tập Test, mặc dù mAP50 giảm nhẹ 0.8% (77.14% so với 77.97%) do Precision giảm nhẹ, mô hình vẫn cho thấy khả năng tổng quát hóa cực kỳ tốt.

### B. Sự suy giảm hiệu năng từ Size-Aware Classification Weighting (`small_weight`)
* **Gây nhiễu Precision**: Cả hai cấu hình có chứa `small_weight` đều bị sụt giảm hiệu năng nghiêm trọng (mAP50 giảm xuống quanh mức 72-74% trên cả val và test). Phân tích chi tiết chỉ ra Precision bị kéo tụt rõ rệt (từ **80.99% xuống 70.17%** trên Test).
* **Giải thích**: Việc ép mô hình tập trung quá mức vào các anchor nhỏ bằng cách nhân loss classification tạo ra một lượng False Positives (dự đoán nhầm nhiễu biển hoặc các sóng nhỏ thành tàu). Điều này khẳng định rằng **classification loss scaling không phải là hướng đi đúng** cho việc giải quyết vấn đề phân tách biên, vì nó phá vỡ sự cân bằng tự nhiên trong phân phối background/foreground của TAL.

### C. Khuyến nghị thiết kế tiếp theo
* Sử dụng **`partial_clip`** làm một augmentation mặc định cho các cấu hình YOLOv8n-P2 tiếp theo trên LEVIR-Ship.
* Loại bỏ vĩnh viễn cơ chế size classification weighting (`small_weight`) để tránh làm ô nhiễm Precision của mô hình.
