"""Phase-1 contracts for the new TinyPerson FTSC paper cases."""

from argparse import Namespace
from pathlib import Path

import pytest

from train_tinyperson_scripts import train_tinyperson_ftsc_paper_new_cases_v1 as runner


EXPECTED = {
    "TP_PRUNE_H0_P2345", "TP_PRUNE_H1_P234", "TP_PRUNE_H3_P2", "TP_CUE_COLOR_H2_MOSAIC",
}


def _args(tmp_path: Path) -> Namespace:
    return Namespace(epochs=100, imgsz=640, batch_size=8, device="cpu", workers=0,
                     patience=20, project=tmp_path, pretrained="yolov8n.pt")


def test_registry_is_exact_unique_and_does_not_alias_historical_cases():
    assert set(runner.CASES) == EXPECTED
    assert runner.DEFAULT_SEEDS == (42, 43, 44)
    assert len({spec.namespace for spec in runner.CASES.values()}) == 4
    assert all(spec.yaml.is_file() and not spec.historical_candidates for spec in runner.CASES.values())
    assert set(runner.CASES).isdisjoint(runner.historical.VARIANTS)
    assert set(runner.HISTORICAL_REFERENCES) == {"H2_MOSAIC", "ES1_MOSAIC"}


def test_source_graphs_ftsc_color_routing_and_structural_pruning():
    report = runner.source_preflight()
    assert report["status"] == "PREPARED_NOT_RUN"
    counts = [report["cases"][case]["layer_count"] for case in
              ("TP_PRUNE_H0_P2345", "TP_PRUNE_H1_P234", "TP_PRUNE_H3_P2")]
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 3
    assert all(item["active_layer_count"] == item["layer_count"] - 1 for item in report["cases"].values())
    color = report["cases"]["TP_CUE_COLOR_H2_MOSAIC"]
    assert color["detect_from"] == [20, 23] and color["color_cue_count"] == 1
    compatibility = report["historical_h2_compatibility"]
    assert compatibility["structural_prefix_match"] and compatibility["ftsc_match"]
    assert compatibility["status"].endswith("PENDING_PHASE2")
    assert "not equivalent" in report["historical_edge_reuse"]["semantic_warning"]


def test_all_new_cases_use_exact_historical_h2_mosaic_training_recipe(tmp_path):
    args = _args(tmp_path)
    for case in runner.CASES:
        kwargs = runner.train_kwargs(args, case, 42, Path("tinyperson.yaml"))
        assert (kwargs["mosaic"], kwargs["close_mosaic"]) == (1.0, 10)
        assert (kwargs["optimizer"], kwargs["lr0"], kwargs["weight_decay"]) == ("AdamW", .002, .0005)
        assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"]) == (100, 640, 8)
        assert kwargs["seed"] == 42 and kwargs["deterministic"] is True
        assert Path(kwargs["project"]).name == runner.CASES[case].namespace


def test_default_cli_is_static_and_training_requires_explicit_identity(monkeypatch, capsys):
    monkeypatch.setattr(runner, "model_preflight", lambda *args: pytest.fail("runtime preflight called"))
    monkeypatch.setattr(runner, "run_one", lambda *args: pytest.fail("training called"))
    runner.main([])
    assert "PREPARED_NOT_RUN" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="requires --case and --seed"):
        runner.main(["--confirm-run"])


def test_runtime_forward_profile_and_ftsc_contract():
    reports = runner.model_preflight()
    assert all(item["status"] == "RUNTIME_PASS" for item in reports.values())
    assert [len(reports[name]["detect_strides"]) for name in
            ("TP_PRUNE_H0_P2345", "TP_PRUNE_H1_P234", "TP_PRUNE_H3_P2")] == [4, 3, 1]
    assert reports["HISTORICAL_H2_MOSAIC_REFERENCE"]["detect_strides"] == [4., 8.]
    color = reports["TP_CUE_COLOR_H2_MOSAIC"]
    assert color["color_cue"] == {"count": 1, "hidden": 32, "zero_initialized_projection": True}
    assert color["edge_cue_count"] == 0
    runner.historical.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(runner.CASES["TP_CUE_COLOR_H2_MOSAIC"].yaml, verbose=False)
    cue = next(module for module in model.modules() if type(module).__name__ == "P2ColorCueFusion")
    baseline = DetectionModel(runner.CANONICAL_H2, verbose=False)
    assert color["params"] - sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in cue.parameters())


def test_color_is_functionally_h2_at_zero_residual_with_mapped_shared_weights():
    import torch
    runner.historical.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    baseline = DetectionModel(runner.CANONICAL_H2, verbose=False).eval()
    color = DetectionModel(runner.CASES["TP_CUE_COLOR_H2_MOSAIC"].yaml, verbose=False).eval()
    source = baseline.state_dict()
    remapped = color.state_dict()
    for key, tensor in source.items():
        sections = key.split(".")
        if len(sections) < 3 or sections[0] != "model" or not sections[1].isdigit():
            continue
        idx = int(sections[1])
        sections[1] = str(idx if idx <= 19 else idx + 1)
        target_key = ".".join(sections)
        if target_key in remapped and remapped[target_key].shape == tensor.shape:
            remapped[target_key] = tensor.detach().clone()
    color.load_state_dict(remapped, strict=True)
    torch.manual_seed(101)
    image = torch.rand(1, 3, 64, 64)
    captured = {}
    handles = [
        color.model[20].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p2", out.detach().clone())),
        color.model[23].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p3", out.detach().clone())),
    ]
    with torch.no_grad():
        base_output = baseline(image)
        color_output = color(image)
        initial_p2, initial_p3 = captured["p2"], captured["p3"]
    base_predictions = base_output[0] if isinstance(base_output, (tuple, list)) else base_output
    color_predictions = color_output[0] if isinstance(color_output, (tuple, list)) else color_output
    torch.testing.assert_close(color_predictions, base_predictions, rtol=1e-4, atol=1e-5)
    cue = color.model[20]
    with torch.no_grad():
        cue.projection.weight.add_(0.1)
        changed_output = color(image)
        changed_p2, changed_p3 = captured["p2"], captured["p3"]
    for handle in handles:
        handle.remove()
    changed_predictions = changed_output[0] if isinstance(changed_output, (tuple, list)) else changed_output
    assert not torch.allclose(changed_p2, initial_p2)
    assert not torch.allclose(changed_p3, initial_p3)
    assert float((changed_predictions - color_predictions).abs().max()) > 1e-5


def test_smart_transfer_from_local_yolov8n_architecture_for_every_case(tmp_path):
    runner.historical.local_ultralytics()
    from ultralytics import YOLO

    source = runner.ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
    checkpoint = tmp_path / "yolov8n.pt"
    YOLO(source).save(checkpoint)
    for spec in runner.CASES.values():
        report = runner.runtime_case_report(spec.yaml, spec.detect_from, spec.strides, str(checkpoint), 640)
        assert report["pretrained_transfer"]["status"] == "PASS"
        assert report["pretrained_transfer"]["changed_tensor_count"] > 0


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_synthetic_loss_backward_and_ftsc_gradient(case):
    report = runner.synthetic_loss_backward(runner.CASES[case].yaml)
    assert report["status"] == "LOSS_BACKWARD_PASS"
    if case == "TP_CUE_COLOR_H2_MOSAIC":
        color = report["color_cue"]
        assert color is not None
        assert color["projection_grad_norm_at_zero_init"] > 0
        assert color["encoder_grad_sum_after_probe"] > 0
        assert color["probe_only_no_optimizer_step"]
    else:
        assert report["color_cue"] is None
