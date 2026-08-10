---
description: Hướng dẫn phát triển code tại local và chạy huấn luyện YOLO trên Marimo
---
# Local to Marimo Training Workflow

Sử dụng checklist này khi bạn muốn phát triển/cải tiến mô hình ở local và đẩy lên chạy huấn luyện trên server Marimo.

## 1. Đồng bộ mã nguồn lên GitHub (Local)

Kích hoạt môi trường và kiểm tra tính đúng đắn trước khi commit:
```bash
conda activate ml2
python -m py_compile train_levir_scripts/train_all_levir.py
git diff --check
```

Commit và đẩy code lên nhánh chính (`main`):
```bash
git add .
git commit -m "feat(levir): description of changes"
git push origin main
```

## 2. Kết nối và đồng bộ trên Marimo Server

Kết nối với Marimo kernel thông qua `marimo-pair` helper:
```bash
bash .agents/skills/marimo-pair/scripts/execute-code.sh \
  --url "$MARIMO_URL" --session "$MARIMO_SESSION" <<'PY'
import marimo as mo
mo.status.toast("🚀 Connected — ready to pair on LEVIR training!")
PY
```

Trên terminal của Marimo server (hoặc thông qua marimo-pair shell), di chuyển vào `/marimo/yolo_code` và kéo code mới nhất về:
```bash
cd /marimo/yolo_code
git pull --ff-only origin main
```

## 3. Khởi chạy huấn luyện (Detached Process)

`HF_TOKEN` là biến trong live marimo kernel, không mặc định nằm trong `os.environ`. Khi launch qua `marimo-pair`, lấy biến này từ kernel globals và truyền riêng vào `env` của detached subprocess; không in token ra output, log hay notebook cell. Đồng thời đảm bảo không có PID nào đang chạy trùng lặp.

Upload Hugging Face là bước **bắt buộc** của mọi training runner. Runner phải fail fast trước khi train nếu thiếu `HF_TOKEN` hoặc `hf_repo_id`; không dùng `--no-upload`, không hard-code `no_upload=True`, và không chờ toàn bộ matrix xong mới upload.

## Protocol inference/evaluation bắt buộc

- Mọi validation, test evaluation và inference YOLO phải truyền explicit `iou=0.5`; không dùng NMS IoU mặc định của Ultralytics.
- Runner phải gọi `model.val(..., iou=0.5)` và `model.predict(..., iou=0.5)`.
- Mọi metrics JSON, manifest, summary và report phải ghi `nms_iou: 0.5`.
- Không so sánh kết quả khác NMS IoU. Artifact không ghi threshold phải được xác minh từ runner hoặc re-evaluate trước khi dùng.
- AP50/AP75 là matching threshold của metric, không phải NMS threshold và không thay thế yêu cầu `iou=0.5`.

```python
_env = os.environ.copy()
_env["HF_TOKEN"] = HF_TOKEN
_process = subprocess.Popen(command, cwd="/marimo/yolo_code", env=_env, ...)
```
Khởi chạy script huấn luyện trong thư mục `train_levir_scripts/` và lưu PID:

```bash
# Thí nghiệm P2 Baseline
python train_levir_scripts/train_all_levir.py --data-root "$LEVIR_DATA_ROOT" --device cuda \
  >> runs/levir_ship_baselines/train_all.log 2>&1 &
echo $! > runs/levir_ship_baselines/train_all.pid

# Thí nghiệm P2 NUDFL-PC-CFR
python train_levir_scripts/train_all_levir_yolov8n_p2_nudfl_pc_cfr.py --data-root "$LEVIR_DATA_ROOT" --device cuda \
  >> runs/levir_yolov8n_p2_nudfl_pc_cfr/train_all.log 2>&1 &
echo $! > runs/levir_yolov8n_p2_nudfl_pc_cfr/train_all.pid
```

Mỗi runner phải xử lý tuần tự theo đơn vị `variant/seed`:

```text
train clean exit
  → kiểm tra best.pt, last.pt, results.csv
  → evaluate best.pt trên val và test
  → ghi evaluation_metrics.json + config/manifest
  → upload toàn bộ run lên HF
  → gọi list_repo_files để xác minh các remote path bắt buộc
  → ghi upload_complete.json
  → mới chuyển sang variant/seed tiếp theo
```

Các remote path tối thiểu phải có sau mỗi run:

```text
runs/<variant>/seed_<seed>/weights/best.pt
runs/<variant>/seed_<seed>/weights/last.pt
runs/<variant>/seed_<seed>/results.csv
runs/<variant>/seed_<seed>/evaluation_metrics.json
runs/<variant>/seed_<seed>/args.yaml
```

Nếu thí nghiệm đánh giá nhiều checkpoint, thay `evaluation_metrics.json` bằng toàn bộ file được yêu cầu, ví dụ `evaluation_metrics_best.json` và `evaluation_metrics_last.json`. Upload thêm YAML model, runner, fixed-split manifest và summary hiện có sau mỗi run để repo luôn khôi phục được trạng thái mới nhất.

Upload phải retry lỗi mạng ít nhất 3 lần. Khi restart, runner được phép reuse local training/evaluation hoàn chỉnh, nhưng chỉ skip upload sau khi đã xác minh đủ remote paths; marker local, PID hoặc lời gọi upload thành công chưa phải bằng chứng artifact đã có trên HF. Nếu upload/verification lỗi, dừng matrix thay vì âm thầm chuyển sang run kế tiếp.

## 4. Giám sát tiến trình

Theo dõi log thời gian thực:
```bash
tail -f /marimo/yolo_code/runs/levir_ship_baselines/train_all.log
```

Chỉ báo một run hoàn tất khi đồng thời có clean exit, train artifacts, val/test evaluation, config/manifest và remote HF verification. Không kết luận hoàn tất từ PID, toast, `best.pt`, metric một split hoặc marker upload đơn lẻ.
