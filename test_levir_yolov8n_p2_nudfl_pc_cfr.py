from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ConflictFineReconstruction, P2NUDFLDetect
from ultralytics.utils.loss import BboxLoss, DFLoss


CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_nudfl_pc_cfr.yaml"
EXPECTED_BINS = torch.tensor(
    [0.0, 0.35, 0.70, 1.05, 1.40, 1.80, 2.30, 2.90, 3.60, 4.50, 5.60, 6.90, 8.40, 10.20, 12.40, 15.0]
)


def test_non_uniform_target_encoding_and_pair_competition():
    logits = torch.zeros(1, 16, requires_grad=True)
    target = torch.tensor([[0.50]])
    loss = DFLoss(16)(logits, target, EXPECTED_BINS)
    expected = -(0.5714286 * torch.log_softmax(logits, 1)[0, 1] + 0.4285714 * torch.log_softmax(logits, 1)[0, 2])
    assert torch.allclose(loss.squeeze(), expected, atol=1e-6)

    bbox_loss = BboxLoss(dfl_bins=EXPECTED_BINS, pc_dfl_gain=1.0, pc_dfl_margin=1.5)
    pc, conflict = bbox_loss.pair_competitive_loss(logits.view(1, 1, 16).expand(1, 4, 16), target.expand(1, 4))
    assert pc.shape == (1, 1) and conflict.shape == (1,)
    pc.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_model_routes_original_p2_and_disables_decoder_in_eval():
    model = YOLO(CONFIG).model
    cfr = next(module for module in model.modules() if isinstance(module, ConflictFineReconstruction))
    detect = next(module for module in model.modules() if isinstance(module, P2NUDFLDetect))
    assert torch.equal(detect.p2_dfl_bins.cpu(), EXPECTED_BINS)
    assert model.model[19].f == [18, 0]
    assert model.model[20].f == 18
    assert model.model[-1].f == [19, 22, 25, 28]

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert cfr.last_aux is None
    assert output[0].shape[-1] == sum((128 // stride) ** 2 for stride in (4, 8, 16, 32))


def test_cfr_weighted_training_loss_is_finite():
    module = ConflictFineReconstruction([32, 16], hidden=8).train()
    p2, p1 = torch.randn(2, 32, 8, 8, requires_grad=True), torch.randn(2, 16, 16, 16)
    assert module([p2, p1]) is p2
    context = {
        "p2_fg_mask": torch.zeros(2, 8, 8, dtype=torch.bool),
        "p2_dfl_conflict": torch.zeros(2, 8, 8),
    }
    context["p2_fg_mask"][:, 3, 4] = True
    context["p2_dfl_conflict"][:, 3, 4] = 2.0
    hyp = type("Hyp", (), {"cfr_gain": 2.0, "cfr_detail_gain": 1.0, "cfr_cos_gain": 1.0, "cfr_conflict_weight": 3.0})
    loss, metrics = module.auxiliary_loss(context, hyp)
    assert torch.isfinite(loss)
    assert {"loss_cfr", "loss_cfr_full", "loss_cfr_detail", "loss_cfr_cos", "loss_pc_dfl"} <= set(metrics)
    loss.backward()
    assert p2.grad is not None and p2.grad.abs().sum() > 0
