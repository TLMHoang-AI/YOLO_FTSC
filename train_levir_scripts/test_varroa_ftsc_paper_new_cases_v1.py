"""Phase-1 contracts for the new structural Varroa FTSC paper cases."""

from argparse import Namespace
from pathlib import Path

import pytest

from train_levir_scripts import train_varroa_ftsc_paper_new_cases_v1 as runner


EXPECTED = {
    "VR_STRUCT_H0_P2345", "VR_STRUCT_H1_P234", "VR_STRUCT_H2_P23", "VR_STRUCT_H3_P2",
}


def _args(tmp_path: Path) -> Namespace:
    return Namespace(epochs=100, imgsz=640, batch_size=4, device="cpu", workers=0,
                     patience=20, project=tmp_path, seed_scope="all")


def test_registry_is_exact_unique_and_separate_from_historical_cases():
    assert set(runner.CASES) == EXPECTED
    assert runner.DEFAULT_SEEDS == (42, 43, 44)
    assert len({spec.namespace for spec in runner.CASES.values()}) == 4
    assert all(spec.yaml.is_file() and not spec.historical_candidates for spec in runner.CASES.values())
    assert set(runner.CASES).isdisjoint(runner.historical.VARIANTS)
    assert set(runner.HISTORICAL_REFERENCES) == {
        "V1_FTSC", "HEAD_P234", "FTSC_P34", "V2_EDGE025", "V3_COLOR",
        "EDGE_P2_DETECT_ONLY", "EDGE_P3_DETECT_ONLY",
    }


def test_paper_comparison_cohorts_require_four_independent_structural_runs():
    structural = list(runner.CASES)
    historical_cues = ["V1_FTSC", "V2_EDGE025", "V3_COLOR",
                       "EDGE_P2_DETECT_ONLY", "EDGE_P3_DETECT_ONLY"]
    runner.assert_paper_comparison(structural, "head_pruning_table")
    runner.assert_paper_comparison(historical_cues, "edge_color_table")
    runner.assert_paper_comparison(["V1_FTSC", "HEAD_P234", "FTSC_P34"],
                                   "historical_head_scale_appendix")
    assert runner.PAPER_COMPARISON_POLICY["head_pruning_table"]["training_action"] == \
        "NEW_INDEPENDENT_TRAINING_ALL_FOUR_CASES"
    assert all("FULLNECK" not in name for name in runner.CASES)
    with pytest.raises(RuntimeError, match="Non-comparable"):
        runner.assert_paper_comparison(["VR_STRUCT_H2_P23", "V1_FTSC"],
                                       "head_pruning_table")
    with pytest.raises(RuntimeError, match="Non-comparable"):
        runner.assert_paper_comparison(["V1_FTSC", "VR_STRUCT_H2_P23"],
                                       "edge_color_table")


def test_amp_retry_only_follows_amp_related_failure():
    assert runner._amp_related_failure(RuntimeError("autocast float16 failure"))
    assert runner._amp_related_failure(RuntimeError("AMP validation failed"))
    assert not runner._amp_related_failure(FileNotFoundError("missing dataset example"))
    assert not runner._amp_related_failure(RuntimeError("CUDA out of memory"))
    assert not runner._amp_related_failure(RuntimeError("invalid model layer index"))


def test_source_graphs_are_repc2f_structural_and_not_historical_full_necks():
    report = runner.source_preflight()
    counts = [report["cases"][case]["layer_count"] for case in runner.CASES]
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 4
    assert all(item["active_layer_count"] == item["layer_count"] - 1 for item in report["cases"].values())
    assert report["cases"]["VR_STRUCT_H2_P23"]["layer_count"] < len(runner.layers(runner.load_yaml(runner.CANONICAL_V1)))
    assert report["cases"]["VR_STRUCT_H1_P234"]["layer_count"] < len(runner.layers(runner.load_yaml(runner.HISTORICAL_P234)))
    assert report["historical_v3_color_reuse"]["status"].endswith("PENDING_PHASE2")


def test_exact_varroa_v1_training_recipe_is_retained(tmp_path):
    kwargs = runner.train_kwargs(_args(tmp_path), Path("varroa.yaml"), 42)
    assert (kwargs["mosaic"], kwargs["close_mosaic"], kwargs["optimizer"]) == (1.0, 10, "auto")
    assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"]) == (100, 640, 4)
    assert kwargs["seed"] == 42 and kwargs["deterministic"] is True
    protocol = runner.protocol_metadata()
    assert (protocol["standard_nms_iou"], protocol["size_bucket_nms_iou"]) == (.7, .5)


def test_default_cli_is_static_and_training_requires_explicit_identity(monkeypatch, capsys):
    monkeypatch.setattr(runner, "model_preflight", lambda *args: pytest.fail("runtime preflight called"))
    monkeypatch.setattr(runner, "run_one", lambda *args: pytest.fail("training called"))
    runner.main([])
    assert "PREPARED_NOT_RUN" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="requires --case and --seed"):
        runner.main(["--confirm-run"])


def test_runtime_forward_profile_ftsc_and_monotonic_complexity():
    reports = runner.model_preflight()
    assert all(item["status"] == "RUNTIME_PASS" for item in reports.values())
    assert [len(reports[name]["detect_strides"]) for name in runner.CASES] == [4, 3, 2, 1]
    assert reports["HISTORICAL_V3_COLOR_REUSE_AUDIT"]["detect_from"] == [19, 22]


def test_historical_v3_color_has_live_detection_gradients_but_checkpoint_is_still_pending():
    report = runner.synthetic_loss_backward(runner.HISTORICAL_V3_COLOR)
    assert report["color_cue"]["projection_grad_norm_at_zero_init"] > 0
    assert report["color_cue"]["encoder_grad_sum_at_zero_init"] == 0
    assert report["color_cue"]["encoder_grad_sum_after_probe"] > 0


def test_historical_v3_zero_residual_matches_v1_and_color_reaches_p2_p3_predictions():
    import torch
    runner.shared.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    baseline = DetectionModel(runner.CANONICAL_V1, verbose=False).eval()
    color = DetectionModel(runner.HISTORICAL_V3_COLOR, verbose=False).eval()
    target = color.state_dict()
    for key, tensor in baseline.state_dict().items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            index = int(parts[1])
            parts[1] = str(index if index <= 18 else index + 1)
            destination = ".".join(parts)
            if destination in target and target[destination].shape == tensor.shape:
                target[destination] = tensor.detach().clone()
    color.load_state_dict(target, strict=True)
    captured = {}
    handles = [
        color.model[19].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p2", out.detach().clone())),
        color.model[22].register_forward_hook(lambda _m, _i, out: captured.__setitem__("p3", out.detach().clone())),
    ]
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        base = baseline(image)[0]
        initial = color(image)[0]
        p2_initial, p3_initial = captured["p2"], captured["p3"]
    torch.testing.assert_close(initial, base, rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        color.model[19].projection.weight.add_(0.1)
        changed = color(image)[0]
        p2_changed, p3_changed = captured["p2"], captured["p3"]
    for handle in handles:
        handle.remove()
    assert not torch.allclose(p2_changed, p2_initial)
    assert not torch.allclose(p3_changed, p3_initial)
    assert float((changed - initial).abs().max()) > 1e-5


def test_smart_transfer_from_local_yolov8n_architecture_for_every_case(tmp_path):
    runner.shared.local_ultralytics()
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
