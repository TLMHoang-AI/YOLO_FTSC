import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import RandomHFAttenuation  # noqa: E402
from ultralytics.nn.modules import Detect, MaskedP2DetailReconstruction  # noqa: E402


def test_hf_attenuation_keeps_labels_and_shape():
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    labels = {"img": img.copy(), "instances": object()}
    out = RandomHFAttenuation(p=1.0, min_alpha=0.0, max_alpha=0.0, blur_kernel=5, mask_grid=4)(labels)
    assert out["img"].shape == img.shape
    assert out["img"].dtype == img.dtype
    assert out["instances"] is labels["instances"]
    assert not np.array_equal(out["img"], img)


def test_masked_p2_detail_train_eval_behavior():
    module = MaskedP2DetailReconstruction(16, mask_prob=1.0)
    x = torch.randn(2, 16, 8, 8)
    module.train()
    y = module(x)
    assert y.shape == x.shape
    assert module.last_aux is not None
    assert module.last_aux["detail_pred"].shape == x.shape
    assert module.last_aux["detail_target"].shape == x.shape
    assert module.last_aux["detail_mask"].shape == (2, 1, 8, 8)
    module.eval()
    z = module(x)
    assert torch.equal(z, x)
    assert module.last_aux is None


def test_variant_yamls_build():
    config_root = ROOT / "models_related/models_config/yolov8/levir"
    hf = YOLO(config_root / "yolov8n_p2_levir_hf_atten_aug.yaml").model
    detail = YOLO(config_root / "yolov8n_p2_levir_masked_detail_recon.yaml").model
    assert isinstance(hf.model[-1], Detect)
    assert hf.model[-1].f == [19, 22, 25, 28]
    assert hf.model[-1].stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert isinstance(detail.model[20], MaskedP2DetailReconstruction)
    assert detail.model[-1].f == [20, 23, 26, 29]
    assert detail.model[-1].stride.tolist() == [4.0, 8.0, 16.0, 32.0]


if __name__ == "__main__":
    test_hf_attenuation_keeps_labels_and_shape()
    test_masked_p2_detail_train_eval_behavior()
    test_variant_yamls_build()
    print("p2_detail_hf checks passed")
