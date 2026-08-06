# LEVIR Ship: Tiny Object Detection with Customized YOLO

Tài liệu này tổng hợp toàn bộ thông tin về mục tiêu, kiến trúc, các hướng tiếp cận và hướng dẫn chạy thực nghiệm cho bài toán phát hiện tàu biển siêu nhỏ (tiny ships) sử dụng họ mô hình YOLOv8/YOLOv10 tùy chỉnh trên tập dữ liệu **LEVIR-Ship**.

---

## 1. Tổng Quan Dự Án & Mục Tiêu
Phát hiện vật thể cực nhỏ (tiny objects) luôn là thách thức đối với các phiên bản YOLO chuẩn do quá trình downsampling (stride lớn) làm mất mát đặc trưng không gian. Dự án này nghiên cứu và cải tiến các thành phần:
- **Kiến trúc mạng (Architecture):** Thêm nhánh P2 (stride 4), GCTS local routing, CFR.
- **Hàm Loss (Loss Functions):** Non-uniform DFL, Pair-competitive loss, Box Consensus loss.
- **Xử lý đặc trưng nền (Feature Filtering):** DBSS (triệt nhiễu nền biển), HIT (vận chuyển đặc trưng bất khả quy).

---

## 2. Cấu Trúc Mã Nguồn (Organized Structure)

Sau khi sắp xếp lại, cấu trúc thư mục chính như sau:

```text
.
├── train_levir_scripts/      # Thư mục chứa toàn bộ script chạy train, test và chẩn đoán
│   ├── train_all_*.py        # Huấn luyện tự động qua các hạt giống (seeds)
│   ├── test_*.py             # Các file smoke test kiểm tra nhanh tính đúng đắn
│   └── analyze_*.py          # Các script đo đạc phương sai (variance) và chẩn đoán lỗi
├── docs/
│   └── reports/              # Thư mục chứa các tài liệu phân tích và báo cáo
│       ├── approach_report.md  # Báo cáo lý thuyết chi tiết cơ chế của từng method
│       ├── report_yolo.md      # Kết quả thực nghiệm và so sánh tổng quan
│       └── report_levir_dbss_generalization.md # Đánh giá khả năng tổng quát hóa của DBSS
├── models_related/
│   ├── ultralytics/          # Mã nguồn core YOLO được tùy biến (nn/tasks.py, utils/loss.py,...)
│   └── models_config/        # Các file YAML cấu hình kiến trúc model cho từng thực nghiệm
└── .ai-bridge/               # Cầu nối thông tin kế hoạch và trạng thái giữa các phiên làm việc
    └── codex-status.md       # Lưu vết nhanh trạng thái Codex cho các session sau
```

---

## 3. Các Phương Pháp Thực Nghiệm Chính

Chi tiết kỹ thuật sâu hơn được trình bày cụ thể trong [approach_report.md](file:///mnt/data/varroa/yolo_related/docs/reports/approach_report.md):

| Phương Pháp | Mô Tả Cơ Chế | File Can Thiệp Chính |
| :--- | :--- | :--- |
| **YOLO-P2 Baseline** | Thêm nhánh P2 (stride 4) tăng mật độ anchor dự đoán vật thể nhỏ | `nn/tasks.py`, `models_config/` |
| **Non-uniform DFL** | Thay đổi phân phối bin DFL, tập trung bin dày ở khoảng cách nhỏ | `utils/loss.py` |
| **Pair-competitive DFL** | Thêm loss cạnh tranh giữa các bin lân cận để tăng độ sắc nét | `utils/loss.py` |
| **CFR (Conflict-Guided)** | Tái cấu trúc đặc trưng tại vị trí xung đột giữa phân loại và định vị | `nn/modules/block.py` |
| **Box Consensus Loss** | Phạt sự lệch pha (variance) của các dự đoán quanh Ground Truth | `utils/loss.py` |
| **DBSS** | Triệt tiêu các tín hiệu nền lặp lại (sóng, bờ biển) bằng Ridge projection | `nn/modules/block.py` |
| **HIT** | Đo độ khó pixel bằng quan hệ không gian/channel rồi vận chuyển residual | `nn/modules/block.py` |

---

## 4. Hướng Dẫn Chạy Thực Nghiệm (Quick Start)

> [!IMPORTANT]
> Trước khi chạy bất kỳ script nào, hãy kích hoạt môi trường conda thích hợp:
> ```bash
> conda activate ml2
> # Hoặc dùng source kích hoạt trực tiếp:
> source /home/duylearch/miniconda3/bin/activate ml2
> ```

### Chạy Huấn Luyện (Matrix Sweep)
Các script huấn luyện nằm trong thư mục `train_levir_scripts/`. Hãy chạy chúng từ thư mục gốc của repository:

```bash
# Huấn luyện tổ hợp P2 + NUDFL + PC + CFR
python train_levir_scripts/train_all_levir_yolov8n_p2_nudfl_pc_cfr.py --data-root <LEVIR_DATA_ROOT> --device cuda

# Huấn luyện mô hình P2 Box Consensus
python train_levir_scripts/train_all_levir_yolov8n_p2_consensus.py --data-root <LEVIR_DATA_ROOT> --device cuda
```

### Chạy Smoke Test Kiểm Tra
Luôn khuyến khích chạy smoke test trước khi khởi động tiến trình train dài hạn:
```bash
python train_levir_scripts/test_levir_yolov8n_p2_nudfl_pc_cfr.py
```

### Phân Tích Phương Sai (Variance Diagnostics)
Để đo đạc độ ổn định của các dự đoán hộp giới hạn:
```bash
python train_levir_scripts/analyze_p2_box_field_variance.py --checkpoint <PATH_TO_PT>
```

---

## 5. Hướng Dẫn Phát Triển (Developer Guide)

### Bản Clone Ultralytics Nội Bộ (Local Ultralytics Clone)
Các khối tùy chỉnh (custom blocks), loss tùy chỉnh và tác vụ mạng (tasks) được nhận diện trực tiếp bằng cơ chế nội bộ của Ultralytics thông qua bản clone nội bộ nằm tại:
```text
models_related/ultralytics
```
Các script trong `train_levir_scripts/` tự động thiết lập import ưu tiên bản local clone này. Để kiểm tra thủ công xem bạn có đang sử dụng đúng bản clone nội bộ này không, hãy chạy:
```bash
PYTHONPATH=models_related/ultralytics python -c "import ultralytics; print(ultralytics.__file__)"
```
Kết quả kỳ vọng phải là đường dẫn nội bộ: `<repo>/models_related/ultralytics/ultralytics/__init__.py`. Nếu kết quả chứa `site-packages/`, các class tùy chỉnh sẽ lỗi `KeyError`.

### Các Khối Tùy Chỉnh Hiện Tại (Current Custom Blocks)
Các block sau đã được đăng ký trong bản local clone:
- `CFR` (Conflict-Guided Fine Reconstruction)
- `DBSSBlock` (Dynamic Background Subspace Suppression)
- `HITBlock` (Hardness-Induced Transport)

Nơi đăng ký cụ thể:
- `models_related/ultralytics/ultralytics/nn/modules/block.py` (Khai báo class và thêm vào `__all__`)
- `models_related/ultralytics/ultralytics/nn/modules/__init__.py` (Import và thêm vào `__all__`)
- `models_related/ultralytics/ultralytics/nn/tasks.py` (Import và thêm vào `base_modules` trong `parse_model()`)

*Chú ý:* File `models_related/custom_blocks/custom_blocks.py` chỉ là file tham chiếu có thể đọc trực tiếp; bộ giải mã YAML thực tế của Ultralytics sẽ gọi các class được đăng ký trong thư mục clone nói trên.

---

### Hướng Dẫn Từng Bước (Step-by-Step Guides)

#### 1. Đổi một Block trong YAML
Nên chọn phương pháp đổi tên block trước (thay vì chèn thêm) để giữ nguyên chỉ số index của các layer phía sau.
Ví dụ trong file config YAML:
- Dòng gốc: `[-1, 3, C2f, [128, True]]`
- Dòng thay thế: `[-1, 3, CFR, [128, True]]`

*Định dạng layer:* `[from, repeats, module, args]`
Đối với các block nằm trong `base_modules` của `parse_model()`, Ultralytics sẽ tự động truyền kích thước kênh của layer trước làm tham số `c1` đầu tiên của constructor. Bạn chỉ cần viết các tham số còn lại (ví dụ `c2`, `shortcut`) vào phần `args` của YAML.

#### 2. Thêm một Class Block Mới
Để thêm một block mới tên là `MyBlock`:
1. Thêm định nghĩa class vào `models_related/ultralytics/ultralytics/nn/modules/block.py`.
2. Thêm `"MyBlock"` vào danh sách `__all__` của `block.py`.
3. Import `MyBlock` và thêm vào `__all__` trong `models_related/ultralytics/ultralytics/nn/modules/__init__.py`.
4. Import `MyBlock` trong `models_related/ultralytics/ultralytics/nn/tasks.py`.
5. Đăng ký trong `parse_model()` của `tasks.py`:
   - Thêm vào danh sách `base_modules` nếu hàm `__init__` nhận đầu vào dạng `(c1, c2, ...)`.
   - Thêm vào danh sách `repeat_modules` nếu block cần tham số lặp lại `n`, ví dụ `C2f(c1, c2, n, ...)`.
6. Gọi tên `MyBlock` trong file YAML cấu hình thực nghiệm của bạn.

#### 3. Chèn một Layer Mới
*Lưu ý:* Việc chèn thêm một dòng module mới vào YAML sẽ làm dịch chuyển toàn bộ index của các layer phía sau. Hãy nhớ cập nhật lại các tham số tham chiếu `from` ở hạ nguồn (ví dụ: `[-1, 9]` hoặc `[15, 18, 21]`).

---

### Kiểm Tra Sau Khi Chỉnh Sửa (Testing & Verification)

1. **Kiểm tra import:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "from ultralytics.nn.tasks import parse_model; print('Import OK')"
   ```
2. **Khởi tạo thử mô hình từ YAML:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "from ultralytics import YOLO; m=YOLO('models_related/models_config/yolov8/levir/yolov8n_p2_levir_nudfl_pc_cfr.yaml'); print(m.model.model[2])"
   ```
3. **Chạy dummy forward pass:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "import torch; from ultralytics import YOLO; m=YOLO('models_related/models_config/yolov8/levir/yolov8n_p2_levir_nudfl_pc_cfr.yaml'); y=m.model(torch.randn(1, 3, 640, 640)); print(type(y))"
   ```
