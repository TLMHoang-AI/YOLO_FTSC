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

## 4. Giám sát tiến trình

Theo dõi log thời gian thực:
```bash
tail -f /marimo/yolo_code/runs/levir_ship_baselines/train_all.log
```
