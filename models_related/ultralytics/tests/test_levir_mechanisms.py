import pytest
import torch

from ultralytics.nn.modules import DBSS, DualIrreducibilityHIT


def batch(with_box=True):
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]) if with_box else torch.empty(0),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]) if with_box else torch.empty(0, 4),
    }


@pytest.mark.parametrize("module", [DBSS(16, embed_channels=8), DualIrreducibilityHIT(16)])
def test_levir_module_is_initially_identity_and_finite(module):
    module.train()
    x = torch.randn(1, 16, 16, 16, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, x)
    output.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_dbss_full_auxiliary_loss_has_gradient():
    module = DBSS(16, embed_channels=8, loss_weight=0.5)
    module.train()
    module(torch.randn(1, 16, 16, 16, requires_grad=True))
    loss, metrics = module.auxiliary_loss(batch())
    assert torch.isfinite(loss)
    assert "loss_dbss_sep" in metrics
    loss.backward()


def test_hit_sparse_gate_and_full_auxiliary_loss():
    module = DualIrreducibilityHIT(16, stride=4, source_topq=0.01, loss_recon_weight=0.1, loss_offset_weight=0.1)
    module.train()
    output = module(torch.randn(1, 16, 16, 16, requires_grad=True))
    assert module.last_aux["gate"].sum() == 3
    loss, metrics = module.auxiliary_loss(batch())
    assert torch.isfinite(loss)
    assert {"loss_hit_recon", "loss_hit_offset"} <= metrics.keys()
    (output.mean() + loss).backward()


def test_hit_no_transport_and_empty_gt():
    module = DualIrreducibilityHIT(8, transport_enabled=False, loss_recon_weight=0.1)
    module.train()
    x = torch.randn(1, 8, 8, 8)
    assert torch.equal(module(x), x)
    loss, _ = module.auxiliary_loss(batch(False))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("constructor", [lambda: DBSS(8, num_bases=25), lambda: DualIrreducibilityHIT(8, stride=0)])
def test_invalid_mechanism_configuration(constructor):
    with pytest.raises(ValueError):
        constructor()
