import sys
from pathlib import Path
ROOT = Path("/mnt/data/varroa/yolo_related")
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO

try:
    model = YOLO(str(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_channel_kvca.yaml"))
    model.load(str(ROOT / "yolov8n.pt"), smart_transfer=True)
    print("SUCCESS: ChannelKVCompressedAttention model built and weights transferred successfully!")
    params = sum(p.numel() for p in model.model.parameters())
    print(f"Parameters: {params:,}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", e)
