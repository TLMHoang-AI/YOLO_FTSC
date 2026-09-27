"""Lightweight architecture preflight for the ported LMSCA H1 two-head model."""

from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel


MODEL_YAML = (
    Path(__file__).resolve().parents[3]
    / "models_related"
    / "models_config"
    / "yolov8"
    / "yolov8_lmsca_h1_2head_p3p4_kvca_heads.yaml"
)


def test_lmsca_h1_graph_and_forward():
    model = DetectionModel(str(MODEL_YAML), ch=3, nc=1, verbose=False).eval()
    layers = model.model
    detect = layers[-1]

    assert len(detect.cv2) == 2
    assert detect.stride.tolist() == [8.0, 16.0]
    assert sum(p.numel() for p in model.parameters()) == 2_226_825
    assert layers[25].__class__.__name__ == "P2LocalToP3"
    assert layers[26].__class__.__name__ == "DeformableHeadConv"
    assert layers[27].__class__.__name__ == "DeformableHeadConv"
    assert layers[28].__class__.__name__ == "KVCompressedAttention"
    assert layers[29].__class__.__name__ == "KVCompressedAttention"
    assert layers[28].sr_ratio == 4
    assert layers[29].sr_ratio == 2
    assert layers[28].mode == layers[29].mode == "avg_dwk"
    assert layers[28].f == 26 and layers[29].f == 27
    assert detect.f == [28, 29]
    assert len(detect.loc_cv) == 0 and len(detect.cvq) == 0

    with torch.no_grad():
        prediction = model(torch.zeros(1, 3, 128, 128))
    assert isinstance(prediction, tuple)
    assert prediction[0].shape == (1, 5, 320)
