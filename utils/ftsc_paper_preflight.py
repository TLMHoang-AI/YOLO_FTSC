"""Small shared validators for the Phase-1 FTSC paper case registries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


FTSC_KEYS = (
    "enabled", "policy", "evidence", "position_alpha", "dfl_entropy_tau",
    "dfl_variance_tau", "dfl_detach", "dfl_apply_cls", "dfl_apply_box",
    "dfl_apply_dfl", "apply_cls", "apply_box", "apply_dfl", "log_clip",
    "per_gt_norm", "warmup_epochs", "ramp_epochs", "strength_init",
    "strength_max", "strength_reg_weight", "fixed_strengths",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Invalid YAML mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layers(payload: dict[str, Any]) -> list[list[Any]]:
    return [*payload["backbone"], *payload["head"]]


def resolved_inputs(graph: list[list[Any]], index: int) -> tuple[int, ...]:
    source = graph[index][0]
    values = source if isinstance(source, list) else [source]
    return tuple(index - 1 if value == -1 else int(value) for value in values)


def ancestors(graph: list[list[Any]], index: int) -> set[int]:
    found: set[int] = set()
    pending = [index]
    while pending:
        current = pending.pop()
        for source in resolved_inputs(graph, current):
            if source >= 0 and source not in found:
                found.add(source)
                pending.append(source)
    return found


def canonical_ftsc(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload["ftsc"][key] for key in FTSC_KEYS}


def validate_ftsc(actual: dict[str, Any], expected: dict[str, Any], case: str) -> None:
    require(canonical_ftsc(actual) == canonical_ftsc(expected), f"{case}: canonical FTSC contract changed")


def validate_active_graph(payload: dict[str, Any], case: str) -> None:
    graph = layers(payload)
    require(graph[-1][2] == "Detect", f"{case}: final layer is not Detect")
    active = ancestors(graph, len(graph) - 1)
    missing = sorted(set(range(len(graph) - 1)) - active)
    require(not missing, f"{case}: layers not connected to Detect: {missing}")


def static_case_report(path: Path, expected_ftsc: dict[str, Any], detect_from: tuple[int, ...],
                       family: str, case: str) -> dict[str, Any]:
    payload = load_yaml(path)
    validate_ftsc(payload, expected_ftsc, case)
    validate_active_graph(payload, case)
    graph = layers(payload)
    require(tuple(graph[-1][0]) == detect_from, f"{case}: Detect taps changed: {graph[-1][0]}")
    module_names = [str(layer[2]) for layer in graph]
    require(family in module_names, f"{case}: expected {family} model family")
    forbidden_family = "RepC2f" if family == "C2f" else "C2f"
    require(forbidden_family not in module_names, f"{case}: mixed {family}/{forbidden_family} model family")
    return {
        "status": "STATIC_PASS",
        "yaml": str(path),
        "yaml_sha256": sha256(path),
        "detect_from": list(detect_from),
        "layer_count": len(graph),
        "active_layer_count": len(ancestors(graph, len(graph) - 1)),
        "model_family": family,
    }


def _shape_summary(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    return type(value).__name__


def runtime_case_report(path: Path, detect_from: tuple[int, ...], strides: tuple[float, ...],
                        pretrained: str | None = None, imgsz: int = 640) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    torch.manual_seed(0)
    model = DetectionModel(path, verbose=False).cpu()
    head = model.model[-1]
    require(isinstance(head, Detect), f"{path.name}: runtime head is not Detect")
    require(tuple(head.f) == detect_from, f"{path.name}: runtime Detect taps changed")
    actual_strides = tuple(float(value) for value in head.stride.tolist())
    require(actual_strides == strides, f"{path.name}: strides {actual_strides} != {strides}")
    calibrator = head.ftsc_calibrator
    require(calibrator is not None and calibrator.policy == "f5", f"{path.name}: FTSC F5 inactive")
    require(tuple(calibrator.evidence_names) == ("position_gaussian", "dfl_distribution"),
            f"{path.name}: FTSC evidence changed")
    require("position_gaussian" in calibrator.strength_logits, f"{path.name}: position strength is not learnable")
    require("dfl_distribution" not in calibrator.strength_logits, f"{path.name}: DFL strength is not fixed")
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    transfer: dict[str, Any]
    if pretrained and Path(pretrained).is_file():
        from utils.ftsc_paper_run_integrity import verify_pretrained_transfer
        loaded = YOLO(path)
        transfer = verify_pretrained_transfer(loaded, Path(pretrained))
        transfer["checkpoint"] = str(Path(pretrained).resolve())
    else:
        transfer = {"status": "UNVERIFIED_CHECKPOINT_NOT_LOCAL", "checkpoint": pretrained}
    return {
        "status": "RUNTIME_PASS",
        "detect_from": list(head.f),
        "detect_strides": list(actual_strides),
        "params": int(get_num_params(model)),
        "gflops": float(get_flops(model, imgsz=imgsz)),
        "gflops_imgsz": imgsz,
        "forward_output": _shape_summary(output),
        "ftsc": {
            "policy": calibrator.policy,
            "evidence": list(calibrator.evidence_names),
            "position_strength_trainable": True,
            "dfl_strength_fixed": 1.0,
        },
        "pretrained_transfer": transfer,
    }


def synthetic_loss_backward(path: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    torch.manual_seed(7)
    model = YOLO(path).model.cpu().train()
    model.args = get_cfg(DEFAULT_CFG)
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    loss, items = model(batch)
    total = loss.sum()
    require(bool(torch.isfinite(total)), f"{path.name}: non-finite synthetic loss")
    total.backward()
    position = model.model[-1].ftsc_calibrator.strength_logits["position_gaussian"]
    require(position.grad is not None and bool(torch.isfinite(position.grad)),
            f"{path.name}: missing/non-finite position-strength gradient")
    color_diagnostic = None
    edge_diagnostic = None
    from ultralytics.nn.modules import EdgeCueFusion, P2ColorCueFusion
    color_modules = [part for part in model.modules() if isinstance(part, P2ColorCueFusion)]
    require(len(color_modules) <= 1, f"{path.name}: unexpected multiple Color Cue modules")
    if color_modules:
        cue = color_modules[0]
        weight_grad = cue.projection.weight.grad
        require(weight_grad is not None and bool(torch.isfinite(weight_grad).all()),
                f"{path.name}: Color projection receives no finite gradient")
        initial_projection_grad = float(weight_grad.detach().norm())
        require(initial_projection_grad > 0,
                f"{path.name}: Color projection is disconnected from the detection loss")
        encoder_grads = [p.grad for p in cue.encoder.parameters() if p.grad is not None]
        require(all(bool(torch.isfinite(g).all()) for g in encoder_grads),
                f"{path.name}: non-finite Color encoder gradient")
        initial_encoder_grad = sum(float(g.detach().abs().sum()) for g in encoder_grads)
        require(initial_encoder_grad == 0.0,
                f"{path.name}: zero projection unexpectedly passes encoder gradients")
        # A zero-initialized final projection blocks initial encoder gradients.
        # Probe the encoder after perturbing the projection away from zero.
        with torch.no_grad():
            cue.projection.weight.add_(0.001)
        model.zero_grad(set_to_none=True)
        warm_loss, _ = model(batch)
        require(bool(torch.isfinite(warm_loss).all()),
                f"{path.name}: Color warm-probe loss is non-finite")
        warm_loss.sum().backward()
        later_encoder_grad = sum(
            float(p.grad.detach().abs().sum()) for p in cue.encoder.parameters()
            if p.grad is not None and bool(torch.isfinite(p.grad).all())
        )
        require(later_encoder_grad > 0,
                f"{path.name}: Color encoder still disconnected after projection unfreezes")
        color_diagnostic = {
            "projection_grad_norm_at_zero_init": initial_projection_grad,
            "encoder_grad_sum_at_zero_init": initial_encoder_grad,
            "encoder_grad_sum_after_probe": later_encoder_grad,
            "zero_init_encoder_grad_expected": True,
            "probe_only_no_optimizer_step": True,
        }
    edge_modules = [part for part in model.modules() if isinstance(part, EdgeCueFusion)]
    require(len(edge_modules) <= 1, f"{path.name}: unexpected multiple Edge Cue modules")
    if edge_modules:
        cue = edge_modules[0]
        weight_grad = cue.projection.weight.grad
        require(weight_grad is not None and bool(torch.isfinite(weight_grad).all()),
                f"{path.name}: Edge projection receives no finite gradient")
        initial_projection_grad = float(weight_grad.detach().norm())
        require(initial_projection_grad > 0,
                f"{path.name}: Edge projection is disconnected from detection loss")
        upstream = [*cue.encoder.parameters(), *(cue.gate.parameters() if cue.gate is not None else ())]
        initial_upstream = sum(float(p.grad.detach().abs().sum()) for p in upstream if p.grad is not None)
        require(initial_upstream == 0.0,
                f"{path.name}: zero Edge projection unexpectedly passes upstream gradients")
        with torch.no_grad():
            cue.projection.weight.add_(0.001)
        model.zero_grad(set_to_none=True)
        warm_loss, _ = model(batch)
        require(bool(torch.isfinite(warm_loss).all()), f"{path.name}: Edge warm-probe loss is non-finite")
        warm_loss.sum().backward()
        later_upstream = sum(
            float(p.grad.detach().abs().sum()) for p in upstream
            if p.grad is not None and bool(torch.isfinite(p.grad).all())
        )
        require(later_upstream > 0, f"{path.name}: Edge branch remains disconnected after projection probe")
        edge_diagnostic = {
            "projection_grad_norm_at_zero_init": initial_projection_grad,
            "upstream_grad_sum_at_zero_init": initial_upstream,
            "upstream_grad_sum_after_probe": later_upstream,
            "zero_init_upstream_grad_expected": True,
            "probe_only_no_optimizer_step": True,
        }
    return {
        "status": "LOSS_BACKWARD_PASS",
        "loss": float(total.detach()),
        "loss_items": [float(value) for value in items.detach().flatten()],
        "position_strength_grad": float(position.grad.detach()),
        "color_cue": color_diagnostic,
        "edge_cue": edge_diagnostic,
    }
