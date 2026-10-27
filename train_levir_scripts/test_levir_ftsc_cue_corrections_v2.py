"""Architecture, dynamic cue and safety tests for corrected LEVIR H2 cues."""

from pathlib import Path

import pytest

from train_levir_scripts import train_levir_ftsc_cue_corrections_v2 as runner


EXPECTED = {"C_H2_POSTFUSION_V2", "E_H2_POSTFUSION_V2"}


def _mapped_models(case: str):
    import torch
    runner.workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    baseline = DetectionModel(runner.CANONICAL_H2, verbose=False).eval()
    cue_model = DetectionModel(runner.CASES[case].yaml, verbose=False).eval()
    target = cue_model.state_dict()
    for key, tensor in baseline.state_dict().items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            index = int(parts[1])
            parts[1] = str(index if index <= 19 else index + 1)
            destination = ".".join(parts)
            if destination in target and target[destination].shape == tensor.shape:
                target[destination] = tensor.detach().clone()
    cue_model.load_state_dict(target, strict=True)
    return baseline, cue_model


def _predictions(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def test_registry_is_exact_and_legacy_cases_are_not_trainable():
    assert set(runner.CASES) == EXPECTED
    assert runner.DEFAULT_SEEDS == (42, 43, 44)
    assert set(runner.HISTORICAL_REFERENCES) == {"H2_p2_p3_nomosaic", "C_H2", "E_H2", "ES1"}
    assert all(item["action"] == "HISTORICAL_ONLY_NOT_CORRECTED"
               for key, item in runner.HISTORICAL_REFERENCES.items() if key != "H2_p2_p3_nomosaic")


def test_static_semantic_contract_and_exact_legacy_cue_parameters():
    report = runner.source_preflight()
    assert report["status"] == "PREPARED_NOT_RUN"
    for case, item in report["cases"].items():
        assert item["detect_from"] == [20, 23]
        assert item["active_layer_count"] == item["layer_count"] - 1
        assert item["historical_result_reusable_as_corrected"] is False
        fixed = runner.layers(runner.load_yaml(runner.CASES[case].yaml))
        legacy = runner.layers(runner.load_yaml(runner.CASES[case].legacy_yaml))
        assert fixed[20][3] == legacy[20][3]
    edge = runner.layers(runner.load_yaml(runner.CASES["E_H2_POSTFUSION_V2"].yaml))
    assert edge[20][3] == [32, 0.25, "constant", 5, 15, "learned"]
    assert runner.CASES["E_H2_POSTFUSION_V2"].legacy_yaml == runner.LEGACY_ES1
    assert runner.layers(runner.load_yaml(runner.LEGACY_EDGE))[20][3] == [32]


def test_no_mosaic_training_protocol_is_exact():
    args = runner.parse_args([])
    kwargs = runner.train_kwargs(args, Path("levir.yaml"), 42)
    assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"]) == (100, 512, 8)
    assert (kwargs["optimizer"], kwargs["mosaic"], kwargs["close_mosaic"]) == ("auto", 0.0, 0)
    assert kwargs["seed"] == 42 and kwargs["deterministic"] is True


def test_default_cli_never_runs_model_or_training(monkeypatch, capsys):
    monkeypatch.setattr(runner, "model_preflight", lambda *args: pytest.fail("runtime preflight called"))
    monkeypatch.setattr(runner, "run_one", lambda *args: pytest.fail("training called"))
    runner.main([])
    assert "PREPARED_NOT_RUN" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="requires --case and --seed"):
        runner.main(["--confirm-run"])


def test_runtime_forward_profile_and_zero_initialization():
    reports = runner.model_preflight()
    assert set(reports) == EXPECTED
    assert all(item["status"] == "RUNTIME_PASS" and item["detect_strides"] == [4., 8.]
               for item in reports.values())
    assert reports["C_H2_POSTFUSION_V2"]["cue"]["module"] == "P2ColorCueFusion"
    assert reports["E_H2_POSTFUSION_V2"]["cue"]["module"] == "P2EdgeCueFusion"


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_zero_residual_matches_h2_then_changed_cue_affects_p2_and_downstream_p3(case):
    import torch

    baseline, cue_model = _mapped_models(case)
    image = torch.rand(1, 3, 64, 64)
    captured: dict[str, torch.Tensor] = {}
    handles = [
        cue_model.model[20].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p2", out.detach().clone())),
        cue_model.model[23].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p3", out.detach().clone())),
    ]
    with torch.no_grad():
        baseline_output = _predictions(baseline(image))
        initial_output = _predictions(cue_model(image))
        initial_p2, initial_p3 = captured["p2"], captured["p3"]
    torch.testing.assert_close(initial_output, baseline_output, rtol=1e-4, atol=1e-5)
    cue = cue_model.model[20]
    with torch.no_grad():
        # A finite diagnostic probe, not a training update. 0.1 yields a
        # prediction change above normal float/allclose noise for Edge as well.
        cue.projection.weight.add_(0.1)
        changed_output = _predictions(cue_model(image))
        changed_p2, changed_p3 = captured["p2"], captured["p3"]
    for handle in handles:
        handle.remove()
    assert not torch.allclose(changed_p2, initial_p2)
    assert not torch.allclose(changed_p3, initial_p3)
    assert float((changed_output - initial_output).abs().max()) > 1e-5


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_real_detection_loss_reaches_corrected_cue(case):
    report = runner.synthetic_loss_backward(runner.CASES[case].yaml)
    cue_key = "color_cue" if case.startswith("C_") else "edge_cue"
    cue = report[cue_key]
    assert cue["projection_grad_norm_at_zero_init"] > 0
    assert cue["probe_only_no_optimizer_step"]
    later = cue.get("encoder_grad_sum_after_probe", cue.get("upstream_grad_sum_after_probe"))
    assert later > 0


def test_smart_transfer_loader_with_local_supplemental_checkpoint(tmp_path):
    runner.workflow.local_ultralytics()
    from ultralytics import YOLO

    source = runner.ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
    checkpoint = tmp_path / "yolov8n.pt"
    YOLO(source).save(checkpoint)
    for spec in runner.CASES.values():
        report = runner.runtime_case_report(spec.yaml, spec.detect_from, spec.strides, str(checkpoint), 512)
        assert report["pretrained_transfer"]["status"] == "PASS"
        assert report["pretrained_transfer"]["verified_backbone_tensor_count"] > 0


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_shifted_h2_layer_transfer_maps_exact_source_tensors(case, tmp_path):
    runner.workflow.local_ultralytics()
    from ultralytics import YOLO
    from utils.ftsc_paper_run_integrity import verify_pretrained_transfer

    source_wrapper = YOLO(runner.CANONICAL_H2)
    checkpoint = tmp_path / "source_h2.pt"
    source_wrapper.save(checkpoint)
    target_wrapper = YOLO(runner.CASES[case].yaml)
    verify_pretrained_transfer(target_wrapper, checkpoint)
    # YOLO.save stores a half-precision checkpoint; compare against the actual
    # reloaded source bytes, not the pre-save FP32 object.
    source, target = YOLO(checkpoint).model.state_dict(), target_wrapper.model.state_dict()
    checked = 0
    for key, tensor in source.items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        index = int(parts[1])
        parts[1] = str(index if index <= 19 else index + 1)
        destination = ".".join(parts)
        if destination in target and target[destination].shape == tensor.shape:
            assert target[destination].equal(tensor)
            checked += 1
    assert checked > 100
