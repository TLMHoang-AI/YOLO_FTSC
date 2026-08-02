from copy import deepcopy

import pytest
import torch

from ultralytics.nn.modules import DBSS, GCTS, Conv, DualIrreducibilityHIT
from ultralytics.utils.torch_utils import fuse_conv_and_bn


def batch(with_box=True):
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]) if with_box else torch.empty(0),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]) if with_box else torch.empty(0, 4),
    }


def test_gcts_is_initially_the_pretrained_conv_path():
    module = GCTS(8, 16).eval()
    baseline = Conv(8, 16, 3, 2).eval()
    baseline.conv.load_state_dict(module.conv.state_dict())
    baseline.bn.load_state_dict(module.bn.state_dict())
    x = torch.randn(2, 8, 12, 12)
    output = module(x)
    assert output.shape == (2, 16, 6, 6)
    assert torch.isfinite(output).all()
    assert torch.equal(output, baseline(x))

    module.gamma.data.fill_(0.25)
    expected = module(x)
    fused = deepcopy(module)
    fused.conv = fuse_conv_and_bn(fused.conv, fused.bn)
    delattr(fused, "bn")
    assert torch.allclose(fused.forward_fuse(x), expected, atol=1e-5, rtol=1e-5)


def test_gcts_candidate_order_and_targets():
    packed = torch.nn.functional.pixel_unshuffle(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]), 2)
    assert packed.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]  # TL, TR, BL, BR

    onehot = GCTS(4, 8, target_mode="onehot")
    _, _, target = onehot._targets(torch.tensor([[0.125, 0.125], [0.375, 0.375]]), 2, 2)
    assert torch.equal(target, torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]))

    bilinear = GCTS(4, 8, target_mode="bilinear")
    _, _, target = bilinear._targets(torch.tensor([[0.25, 0.25], [1.0, 1.0]]), 2, 2)
    assert torch.allclose(target[0], torch.full((4,), 0.25))
    assert torch.allclose(target.sum(1), torch.ones(2))


def test_gcts_auxiliary_loss_has_selector_gradient_and_handles_empty_gt():
    module = GCTS(8, 16, loss_weight=0.2).train()
    module(torch.randn(1, 8, 8, 8, requires_grad=True))
    loss, metrics = module.auxiliary_loss(batch())
    assert torch.isfinite(loss) and "loss_gcts_select" in metrics
    loss.backward()
    assert module.selector.weight.grad is not None and torch.isfinite(module.selector.weight.grad).all()

    module(torch.randn(1, 8, 8, 8))
    empty_loss, _ = module.auxiliary_loss(batch(False))
    assert empty_loss == 0 and empty_loss.requires_grad


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


def test_dbss_matches_reference_displacement():
    torch.manual_seed(7)
    module = DBSS(8, embed_channels=4, candidate_grid=(2, 2), shortlist_size=4, num_bases=2, gamma_max=0.6)
    module.train()
    module.direction[-1].weight.data.normal_(std=0.02)
    x = torch.randn(2, 8, 6, 6)
    output = module(x)
    reference = []
    embedding = module._embed(x)
    for index in range(x.shape[0]):
        emb = embedding[index : index + 1]
        tokens = emb.flatten(2).squeeze(0).T
        candidates = torch.nn.functional.adaptive_avg_pool2d(emb, (2, 2)).flatten(2).squeeze(0).T
        normalized_tokens = torch.nn.functional.normalize(tokens, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidates, dim=-1)
        scores = (normalized_candidates @ normalized_tokens.T).mean(1)
        indices = module._select_bases(scores, normalized_candidates)
        residual = (module._project(tokens, candidates[indices]) - tokens).neg().T.reshape_as(emb)
        direction = module.direction(torch.cat((x[index : index + 1], residual), 1))
        gamma = module.magnitude(residual).sigmoid()
        scale = (x[index : index + 1].square().mean(1, keepdim=True) + 1e-6).sqrt()
        displacement = module.gamma_max * scale * gamma * direction / (1 + direction.norm(dim=1, keepdim=True))
        reference.append(x[index : index + 1] + displacement)
    assert torch.allclose(output, torch.cat(reference), atol=1e-6, rtol=1e-5)
    assert module.last_aux["displacement_ratio"] < module.gamma_max


def test_dbss_ridge_falls_back_to_lstsq(monkeypatch):
    module = DBSS(8, embed_channels=4, candidate_grid=(2, 2), shortlist_size=4, num_bases=2)
    original = torch.linalg.solve_ex

    def failed_solve(matrix, rhs, check_errors=False):
        return torch.full_like(rhs, torch.nan), torch.ones((), device=matrix.device, dtype=torch.int32)

    monkeypatch.setattr(torch.linalg, "solve_ex", failed_solve)
    projected = module._project(torch.randn(6, 4), torch.randn(2, 4))
    monkeypatch.setattr(torch.linalg, "solve_ex", original)
    assert torch.isfinite(projected).all()
    assert module._ridge_retry_count == 1
    assert module._ridge_lstsq_count == 1


def test_dbss_mixed_precision_is_finite():
    module = DBSS(8, embed_channels=4, candidate_grid=(2, 2), shortlist_size=4, num_bases=2)
    x = torch.randn(1, 8, 8, 8, dtype=torch.bfloat16)
    module = module.to(dtype=torch.bfloat16)
    assert torch.isfinite(module(x)).all()


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
