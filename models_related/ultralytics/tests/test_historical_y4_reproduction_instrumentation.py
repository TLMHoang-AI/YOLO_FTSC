"""Focused safety checks for the historical-Y4 observational instrumentation."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOSS_SOURCE = ROOT / "models_related/ultralytics/ultralytics/utils/loss.py"
RUNNER_SOURCE = ROOT / "train_levir_scripts/train_historical_y4_reproduction.py"


def _instrumentation_block() -> str:
    source = LOSS_SOURCE.read_text(encoding="utf-8")
    start = source.index("        self.ftsc_metrics = {}\n        if self.ftsc_calibrator is not None:")
    end = source.index("        cls_target_scores = target_scores", start)
    return source[start:end]


def test_diagnostics_are_detached_and_do_not_call_training_or_rng_apis():
    block = _instrumentation_block()
    assert ".detach()" in block
    forbidden = (".backward(", ".step(", "torch.rand", "torch.randn", "optimizer", "scheduler")
    assert not any(token in block for token in forbidden)
    tree = ast.parse(textwrap.dedent(block))
    assigned = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert "self" not in assigned


def test_diagnostics_do_not_reassign_assignment_inputs():
    block = _instrumentation_block()
    tree = ast.parse(textwrap.dedent(block))
    assignment_inputs = {"fg_mask", "target_scores", "target_bboxes", "target_gt_idx", "pred_distri"}
    writes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    writes.add(target.id)
    assert not assignment_inputs.intersection(writes)


def test_diagnostic_summary_does_not_change_gradient_or_rng():
    import pytest

    torch = pytest.importorskip("torch")
    parameter = torch.tensor([0.2, -0.4, 0.8], requires_grad=True)
    before_rng = torch.get_rng_state().clone()
    detached = parameter.detach().float()
    summary = {
        "mean": float(detached.mean()),
        "std": float(detached.std(unbiased=False)),
        "min": float(detached.min()),
        "max": float(detached.max()),
    }
    (parameter.square().sum()).backward()
    assert summary["mean"] == float(parameter.detach().mean())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.allclose(parameter.grad, 2 * parameter.detach())


def test_historical_freeze_and_fitness_contract_are_source_locked():
    trainer_source = (ROOT / "models_related/ultralytics/ultralytics/engine/trainer.py").read_text(encoding="utf-8")
    assert 'always_freeze_names = [".dfl"]' in trainer_source
    assert 'metrics.pop("fitness", -self.loss.detach().cpu().numpy())' in trainer_source
    assert "optimizer=auto" in trainer_source


def test_runner_is_locked_to_seed42_and_historical_snapshot():
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert 'HISTORICAL_COMMIT = "ca48644cd1abe08315de08008d138f84565bf6f7"' in source
    assert 'if args.seeds != [42]:' in source
    assert 'EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_historical_y4_reproduction"' in source
    assert "model.train(**kwargs)" in source
    assert "source_preflight_only" in source


def test_runner_has_no_current_worktree_or_marimo_dependency():
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "marimo" not in source.lower()
    assert "git reset" not in source
    assert "git checkout" not in source
