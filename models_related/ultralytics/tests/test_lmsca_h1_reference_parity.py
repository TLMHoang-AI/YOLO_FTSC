"""Subprocess parity test against the historical LMSCA Ultralytics fork."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ACTIVE_ROOT = ROOT / "models_related" / "ultralytics"
SOURCE_ROOT = ROOT.parent / "models_related_LMSCA_YOLO" / "ultralytics"
ACTIVE_YAML = ROOT / "models_related" / "models_config" / "yolov8" / "yolov8_lmsca_h1_2head_p3p4_kvca_heads.yaml"
SOURCE_YAML = ROOT.parent / "models_related_LMSCA_YOLO" / "models_config" / "yolov8_varroa_h1_2head_p3p4_kvca_heads.yaml"


SNAPSHOT_SCRIPT = r'''
import json
import sys
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel

yaml_path, output_path = sys.argv[1:]
torch.manual_seed(123)
model = DetectionModel(yaml_path, ch=3, nc=1, verbose=False).eval()
torch.manual_seed(456)
x = torch.randn(1, 3, 64, 64)
captured = {}
def hook(name):
    def _hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            captured[name] = output.detach().cpu()
    return _hook
for index in (18, 21, 25, 26, 27, 28, 29):
    model.model[index].register_forward_hook(hook(str(index)))
with torch.no_grad():
    prediction = model(x)
if isinstance(prediction, tuple):
    prediction = prediction[0]
captured["prediction"] = prediction.detach().cpu()
detect = model.model[-1]
snapshot = {
    "layer_count": len(model.model),
    "layer_types": [layer.__class__.__name__ for layer in model.model],
    "from": [layer.f for layer in model.model],
    "detect_from": detect.f,
    "stride": detect.stride.tolist(),
    "params": sum(parameter.numel() for parameter in model.parameters()),
    "state_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
    "attrs": {
        "p3_sr": model.model[28].sr_ratio,
        "p4_sr": model.model[29].sr_ratio,
        "p3_mode": model.model[28].mode,
        "p4_mode": model.model[29].mode,
        "detect_loc_cv": len(getattr(detect, "loc_cv", [])),
        "detect_cvq": len(getattr(detect, "cvq", [])),
    },
}
torch.save({"meta": snapshot, "tensors": captured}, output_path)
'''


def _run_snapshot(pythonpath: str, yaml_path: Path, output_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-c", SNAPSHOT_SCRIPT, str(yaml_path), str(output_path)],
        cwd=pythonpath.split(":", 1)[0],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"snapshot subprocess failed: {result.stderr}")
    return torch_load(output_path)


def torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def test_lmsca_h1_matches_historical_reference_numerically():
    import torch

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        source_namespace = temporary_root / "models_related"
        source_namespace.mkdir()
        (source_namespace / "ultralytics").symlink_to(SOURCE_ROOT, target_is_directory=True)

        source_snapshot = _run_snapshot(
            f"{temporary_root}:{SOURCE_ROOT}", SOURCE_YAML, temporary_root / "source.pt"
        )
        active_snapshot = _run_snapshot(
            str(ACTIVE_ROOT), ACTIVE_YAML, temporary_root / "active.pt"
        )

    source_meta, active_meta = source_snapshot["meta"], active_snapshot["meta"]
    assert active_meta["layer_count"] == source_meta["layer_count"] == 31
    assert active_meta["layer_types"] == source_meta["layer_types"]
    assert active_meta["from"] == source_meta["from"]
    assert active_meta["detect_from"] == source_meta["detect_from"] == [28, 29]
    assert active_meta["stride"] == source_meta["stride"] == [8.0, 16.0]
    assert active_meta["params"] == source_meta["params"] == 2_226_825
    assert active_meta["state_shapes"] == source_meta["state_shapes"]
    assert active_meta["attrs"] == source_meta["attrs"]

    for name in ("18", "21", "25", "26", "27", "28", "29", "prediction"):
        torch.testing.assert_close(
            active_snapshot["tensors"][name], source_snapshot["tensors"][name], rtol=1e-5, atol=1e-6
        )
