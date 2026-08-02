from copy import deepcopy

import pytest
import torch

from ultralytics.nn.modules import (
    DBSS,
    GCTS,
    Conv,
    DualIrreducibilityHIT,
    v10GCTSDetect,
    v10GCTSP3NUDFLDetect,
    v10P3NUDFLDetect,
)
from ultralytics.utils.loss import DFLoss
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


def test_gcts_v2_is_initially_identity_and_preserves_quadrant_coordinates():
    head = v10GCTSDetect(nc=1, ch=(8, 16, 32, 64)).train()
    features = [torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4), torch.randn(2, 64, 2, 2)]
    box_features, cls_features = head._route(features)
    assert deepcopy(head).last_gcts is None
    assert torch.equal(box_features[0], features[1])
    assert torch.equal(cls_features[0], features[1])
    assert all(torch.equal(a, b) for a, b in zip(box_features[1:], features[2:]))

    centers = torch.tensor([[0.1375, 0.2125], [0.925, 0.8875]])
    head.last_gcts = {"alpha": torch.empty(1, 4, 8, 8)}
    batch_data = {"batch_idx": torch.zeros(2), "bboxes": torch.cat((centers, torch.ones(2, 2)), 1)}
    _, _, _, fractions, target = head._targets(batch_data)
    expected = torch.stack((target[:, 1] + target[:, 3], target[:, 2] + target[:, 3]), 1)
    assert torch.allclose(expected, fractions)


def test_gcts_v2_routes_separate_box_and_class_features_and_backpropagates():
    head = v10GCTSDetect(nc=1, epsilon=0.05, tiny_gate=True, ch=(8, 16, 32, 64)).train()
    head.cls_projection.bias.data.fill_(1)
    head.pos_projection.bias.data.fill_(1)
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4), torch.randn(1, 64, 2, 2)]
    box_features, cls_features = head._route(features)
    assert torch.allclose(cls_features[0], features[1] + 1)
    assert not torch.equal(box_features[0], features[1])
    assert (box_features[0] - features[1]).abs().max() <= head.epsilon

    batch_data = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0, 0]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]]),
    }
    loss, metrics = head.auxiliary_loss(batch_data)
    assert torch.isfinite(loss)
    assert {"loss_gcts_v2_pos", "loss_gcts_v2_gate", "gcts_v2_coord_mae"} <= metrics.keys()
    loss.backward()
    assert head.selector.weight.grad is not None

    no_gate = v10GCTSDetect(nc=1, tiny_gate=False, ch=(8, 16, 32, 64)).train()
    no_gate._route(features)
    _, no_gate_metrics = no_gate.auxiliary_loss(batch_data)
    assert no_gate_metrics["loss_gcts_v2_gate"] == 0


def test_gcts_v2_gate_thresholds_collisions_and_background_sampling():
    head = v10GCTSDetect(nc=1, tiny_gate=True, ch=(8, 16, 32, 64)).train()
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4), torch.randn(1, 64, 2, 2)]
    head._route(features)
    # All boxes share a cell: the <20 px target must win over ignore (20-24 px) and large (>24 px).
    boxes = torch.tensor([[0.3, 0.3, 10 / 64, 0.0], [0.3, 0.3, 22 / 64, 0.0], [0.3, 0.3, 30 / 64, 0.0]])
    batch_data = {"img": torch.rand(1, 3, 64, 64), "batch_idx": torch.zeros(3), "bboxes": boxes}
    bi, ys, xs, _, _ = head._targets(batch_data)
    indices, targets = head._gate_targets(batch_data, bi, ys, xs)
    assert len(indices) == 2  # one labeled cell plus one deterministic background cell
    assert sorted(targets.tolist()) == [0.0, 1.0]

    # Without the tiny object, a 20-24 px collision remains ignored and only the large cell is labeled.
    boxes = torch.tensor([[0.3, 0.3, 22 / 64, 0.0], [0.7, 0.7, 30 / 64, 0.0]])
    batch_data["batch_idx"] = torch.zeros(2)
    batch_data["bboxes"] = boxes
    bi, ys, xs, _, _ = head._targets(batch_data)
    indices, targets = head._gate_targets(batch_data, bi, ys, xs)
    assert len(indices) == 2 and not targets.any()  # one large negative plus one background negative


def test_gcts_v2_gate_loss_is_autocast_safe():
    head = v10GCTSDetect(nc=1, tiny_gate=True, ch=(8, 16, 32, 64)).train()
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4), torch.randn(1, 64, 2, 2)]
    batch_data = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        head._route(features)
        loss, _ = head.auxiliary_loss(batch_data)
    assert torch.isfinite(loss)


def test_p3_nonuniform_dfl_targets_and_expectation():
    bins = torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 7, 9, 11, 13, 15.0])
    targets = torch.tensor([[0.0, 0.3, 1.25, 14.5]])
    right = torch.searchsorted(bins, targets, right=False).clamp(1, len(bins) - 1)
    left = right - 1
    right_weight = (targets - bins[left]) / (bins[right] - bins[left])
    distribution = torch.zeros(4, len(bins))
    distribution.scatter_(1, left.view(-1, 1), (1 - right_weight).view(-1, 1))
    distribution.scatter_add_(1, right.view(-1, 1), right_weight.view(-1, 1))
    assert torch.allclose(distribution.matmul(bins), targets.flatten())
    logits = distribution.clamp_min(1e-6).log().requires_grad_()
    loss = DFLoss(len(bins))(logits, targets, bins)
    assert loss.shape == (1, 1) and torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_p3_nonuniform_heads_keep_other_levels_uniform():
    for head in (v10P3NUDFLDetect(nc=1, ch=(16, 32, 64)), v10GCTSP3NUDFLDetect(nc=1, ch=(8, 16, 32, 64))):
        assert torch.equal(head.p3_dfl_bins[:8], torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2.0]))
        assert torch.equal(head.dfl.conv.weight.flatten(), torch.arange(16, dtype=torch.float))


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
