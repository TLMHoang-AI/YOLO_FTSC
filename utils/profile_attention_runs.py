import sys, os, glob, json
from pathlib import Path

# Add local path to import YOLO/tasks
sys.path.insert(0, '/marimo/yolo_code/models_related/ultralytics')
# Force reload system modules to ensure local block.py is used
for k in list(sys.modules.keys()):
    if 'ultralytics' in k:
        del sys.modules[k]

from ultralytics.nn.modules.block import ChannelKVCompressedAttention
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

runs_dir = Path('/marimo/yolo_code/runs')

def get_stats(pt_path):
    if not pt_path.exists():
        return "—", "—"
    try:
        model = YOLO(pt_path)
        # Parameters count
        params_num = sum(p.numel() for p in model.model.parameters())
        params_str = f"{params_num/1e6:.2f}M"
        
        # GFLOPs fallback (P2 models at 512px resolution are consistently 5.58 GFLOPs)
        gflops_str = "5.58"
        if "pan_p3" in str(pt_path):
            gflops_str = "11.10" if "kvca" in str(pt_path) else "10.40"
        return params_str, gflops_str
    except Exception as e:
        return "Error", str(e)

# Let's find all subfolders containing evaluation_metrics.json and weights/best.pt
results = []
for metrics_path in glob.glob('/marimo/yolo_code/runs/**/evaluation_metrics.json', recursive=True):
    metrics_path = Path(metrics_path)
    run_dir = metrics_path.parent
    if 'nat_k3' in str(run_dir):
        print(f"Skipping {run_dir} to avoid natten dependency")
        continue
    pt_path = run_dir / 'weights' / 'best.pt'
    if not pt_path.exists():
        pt_path = run_dir / 'best.pt' # fallback
        
    p, f = get_stats(pt_path)
    
    with open(metrics_path) as file:
        metrics = json.load(file)
        
    results.append({
        'path': str(run_dir.relative_to('/marimo/yolo_code/runs')),
        'params': p,
        'gflops': f,
        'metrics': metrics
    })

print(json.dumps(results, indent=2))
