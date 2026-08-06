# Codex Status: Repository Goals & Structure

Tài liệu này lưu trữ thông tin tóm tắt về mục tiêu, cấu trúc mã nguồn và các hướng tiếp cận của repository để các AI agent có thể đọc trực tiếp khi khởi động session mới, tránh mất token phân tích lại từ đầu.

---

## 1. Mục Tiêu Của Repository (Goals)
Nghiên cứu, tùy chỉnh cấu trúc mô hình và các hàm Loss của họ **YOLO (YOLOv8/YOLOv10)** để tối ưu hóa khả năng phát hiện vật thể cực nhỏ (tiny object detection) — cụ thể là phát hiện tàu biển trên tập dữ liệu **LEVIR-Ship** và **Varroa**.

---

## 2. Cấu Trúc Repository (Repository Structure)

- **`models_related/ultralytics/`**: Mã nguồn core YOLO (tùy biến từ thư viện Ultralytics).
  - [block.py](file:///mnt/data/varroa/yolo_related/models_related/ultralytics/ultralytics/nn/modules/block.py): Nơi cài đặt các module bổ sung (như CFR).
  - [head.py](file:///mnt/data/varroa/yolo_related/models_related/ultralytics/ultralytics/nn/modules/head.py): Tùy biến nhánh regression/detection head (nhánh P2, offset regression).
  - [tasks.py](file:///mnt/data/varroa/yolo_related/models_related/ultralytics/ultralytics/nn/tasks.py): Định nghĩa pipeline khởi tạo mạng neural.
  - [loss.py](file:///mnt/data/varroa/yolo_related/models_related/ultralytics/ultralytics/utils/loss.py): Nơi cài đặt tất cả các hàm loss tùy chỉnh (NUDFL, PC-loss, Box Consensus loss).
- **`models_related/models_config/yolov8/levir/`**: Chứa các file cấu hình mạng (`.yaml`) cho từng thực nghiệm:
  - `yolov8n_p2_levir_nudfl_pc_cfr.yaml`: Cấu hình chạy kết hợp NUDFL + PC + CFR.
  - `yolov8n_p2_levir_consensus.yaml`: Cấu hình chạy Box Consensus.
- **Các script huấn luyện (`train_all_*.py`)**:
  - Chạy thực nghiệm quét trên nhiều hạt giống (seeds), tự động tính toán metrics trên tập Validation/Test, ghi log và upload kết quả lên Hugging Face (`duyle2408`).
- **Các script kiểm tra (`test_*.py`)**: Smoke test nhanh tính đúng đắn của mô hình/loss trước khi chạy lớn.
- **`diagnostics/`**: Thư mục chứa log phân tích chi tiết (ví dụ: đo phương sai dự đoán hộp giới hạn cục bộ).
- **`approach_report.md`**: Tài liệu báo cáo chính thức phân tích lý thuyết, cơ chế hoạt động, đường truyền gradient và kết quả của từng phương pháp.

---

## 3. Các Hướng Tiếp Cận & Cơ Chế Đang Thực Hiện (Implemented Approaches)

1. **YOLO-P2 Baseline:** Thêm nhánh dự đoán P2 (stride 4) để tăng mật độ định vị đối với tiny object.
2. **P2 Offset Regression:** Cho phép các cạnh bounding box lấy mẫu feature lệch nhau ở mức sub-cell bằng bilinear `grid_sample`.
3. **DBSS (Dynamic Background Subspace Suppression):** Triệt tiêu nhiễu nền (mặt nước, sóng) bằng phương pháp ridge projection qua một background subspace động.
4. **HIT (Dual-Irreducibility Hardness-Induced Transport):** Xác định các pixel khó dựa trên độ bất khả quy không gian & channel để vận chuyển residual hữu ích.
5. **GCTS v1 & v2 (Grid-Cell Target Selection):** Định tuyến thông tin P2 cục bộ vào head hoặc neck P3 để tránh mất dấu vị trí đối tượng khi downsample.
6. **Non-uniform DFL (NUDFL) + Pair-competitive DFL:** Phân bố bin DFL dày hơn gần vị trí 0 (khoảng cách nhỏ) và áp dụng loss PC để tăng độ sắc nét của phân phối cạnh.
7. **CFR (Conflict-Guided Fine Reconstruction):** Hướng dẫn tái cấu trúc feature tại các vùng xảy ra conflict giữa phân loại và định vị.
8. **Box Consensus Loss:** Phạt phương sai không gian giữa các dự đoán xung quanh Ground Truth nhằm giảm thiểu lỗi bất định vị trí của P2 head.
