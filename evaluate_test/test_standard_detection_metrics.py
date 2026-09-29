"""Pure no-training checks for shared standard detection evaluation semantics."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_test import standard_detection_metrics as metrics
from evaluate_test import size_bucket_evaluator as size_metrics


class _Box:
    all_ap = np.asarray([[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]])
    map75 = .6


class _Result:
    box = _Box()
    results_dict = {"metrics/precision(B)": .5, "metrics/recall(B)": .25, "metrics/mAP50(B)": .7, "metrics/mAP50-95(B)": .4}


def test_partner_map50_75_is_exactly_six_iou_points_and_ap75_is_distinct():
    assert metrics.MAP50_75_IOU_THRESHOLDS == (.5, .55, .6, .65, .7, .75)
    assert metrics.map50_75(_Result()) == np.asarray(_Box.all_ap[:, :6]).mean()
    values = metrics.standard_metrics(_Result(), "test")
    assert values["test/metrics/mAP75(B)"] == .6
    assert round(values["test/metrics/mAP50-75(B)"], 8) == .35


def test_f1_speed_and_fps_are_safe_and_distinct():
    assert metrics.f1_score(.5, .25) == 1 / 3
    assert metrics.f1_score(0, 0) == 0
    speed = metrics.speed_metrics({"preprocess": 1, "inference": 4, "postprocess": 5})
    assert speed["test_speed/total_ms_per_image"] == 10
    assert speed["test_speed/inference_only_fps"] == 250
    assert speed["test_speed/end_to_end_fps"] == 100


def test_coco_and_tinybenchmark_protocols_are_explicitly_separate():
    assert metrics.COCO_AREA_RANGES["small"] == (0.0, 32.0**2)
    assert metrics.COCO_AREA_RANGES["medium"] == (32.0**2, 96.0**2)
    assert metrics.COCO_AREA_RANGES["large"] == (96.0**2, 1e10)
    assert "Tiny1=1-8px" in metrics.TINYBENCHMARK_PROTOCOL
    assert len(metrics.COCO_IOU_THRESHOLDS) == 10


def test_mixed_hardware_is_not_speed_aggregate_comparable():
    common = {"benchmark/gpu_name": "A", "benchmark/backend": "PyTorch", "benchmark/device": "cuda:0", "benchmark/imgsz": 512, "benchmark/batch": 8, "benchmark/precision": "FP16"}
    assert metrics.same_benchmark_fingerprint([common, dict(common)])
    changed = dict(common); changed["benchmark/gpu_name"] = "B"
    assert not metrics.same_benchmark_fingerprint([common, changed])


def test_gflops_primary_positive_is_used_with_method_provenance():
    value, method = metrics._measure_gflops(object(), 512, lambda *_args, **_kwargs: 6.7, lambda *_args, **_kwargs: 9.9)
    assert (value, method) == (6.7, "thop")


def test_gflops_zero_or_primary_failure_uses_non_mutating_profiler_fallback():
    class Inner: pass
    original = Inner()
    received = []
    def profiler(clone, **_kwargs):
        received.append(clone)
        return 7.1
    value, method = metrics._measure_gflops(original, 512, lambda *_args, **_kwargs: 0.0, profiler)
    assert (value, method) == (7.1, "torch_profiler")
    assert received[0] is not original
    value, method = metrics._measure_gflops(original, 512, lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("thop")), profiler)
    assert (value, method) == (7.1, "torch_profiler")


def test_gflops_never_silently_records_zero_when_both_backends_fail():
    try:
        metrics._measure_gflops(object(), 512, lambda *_args, **_kwargs: 0.0, lambda *_args, **_kwargs: 0.0)
    except RuntimeError as error:
        assert "positive finite GFLOPs" in str(error)
    else:
        raise AssertionError("zero GFLOPs must fail closed")


def test_complexity_is_measured_before_validation_or_prediction(monkeypatch, tmp_path):
    events = []
    class YOLO:
        def __init__(self, _checkpoint): self.model = object()
        def val(self, **_kwargs): events.append("val"); return types.SimpleNamespace(speed={})
    monkeypatch.setattr(metrics, "ensure_local_ultralytics", lambda: "local")
    monkeypatch.setattr(metrics, "model_complexity", lambda *_args: events.append("complexity") or {"model/GFLOPs": 1})
    monkeypatch.setattr(metrics, "standard_metrics", lambda *_args: {})
    monkeypatch.setattr(metrics, "actual_precision", lambda *_args: "FP32")
    monkeypatch.setattr(metrics, "speed_metrics", lambda *_args: {})
    monkeypatch.setattr(metrics, "benchmark_provenance", lambda **_kwargs: {})
    monkeypatch.setattr(metrics, "coco_area_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))
    (tmp_path / "weights").mkdir(); (tmp_path / "weights/best.pt").touch()
    metrics.evaluate_run(tmp_path, tmp_path / "split.yaml", dataset="levir", imgsz=512, batch=8, device="cpu", workers=0, include_size=False)
    assert events[0] == "complexity"


def test_extended_artifact_does_not_overwrite_legacy_metrics(tmp_path):
    legacy = {"test/metrics/mAP50(B)": .7}; extra = {"test_coco/AP_small": .2}
    (tmp_path / "evaluation_metrics.json").write_text(__import__("json").dumps(legacy))
    (tmp_path / "evaluation_metrics_extended.json").write_text(__import__("json").dumps(extra))
    assert metrics.load_merged_metrics(tmp_path) == {**legacy, **extra}


def test_conflicting_extended_metric_fails_clearly(tmp_path):
    (tmp_path / "evaluation_metrics.json").write_text('{"test/metrics/mAP50(B)": 0.7}')
    (tmp_path / "evaluation_metrics_extended.json").write_text('{"test/metrics/mAP50(B)": 0.8}')
    try:
        metrics.load_merged_metrics(tmp_path)
    except ValueError as error:
        assert "conflicts with legacy metric" in str(error)
    else:
        raise AssertionError("conflicting metrics must not be overlaid")


def test_device_provenance_uses_selected_cuda_index(monkeypatch):
    class CUDA:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 2
        @staticmethod
        def get_device_name(index): return ["GPU-zero", "GPU-one"][index]
    fake_torch = types.SimpleNamespace(cuda=CUDA(), version=types.SimpleNamespace(cuda="12.1"), __version__="x")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert metrics.benchmark_provenance(device="cpu", imgsz=1, batch=1, workers=0, precision="FP32", split="test")["benchmark/gpu_name"] == "CPU"
    assert metrics.benchmark_provenance(device="0", imgsz=1, batch=1, workers=0, precision="FP16", split="test")["benchmark/device"] == "cuda:0"
    second = metrics.benchmark_provenance(device="cuda:1", imgsz=1, batch=1, workers=0, precision="FP16", split="test")
    assert second["benchmark/gpu_name"] == "GPU-one" and second["benchmark/half"] is True


def test_native_prediction_chunking_keeps_global_ids_and_arguments():
    calls = []
    class Values:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.value
    class Boxes:
        xyxy = Values([[1, 2, 3, 4]])
        conf = Values([.5])
    class Result: boxes = Boxes()
    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs); return iter([Result() for _ in kwargs["source"]])
    output = size_metrics._predictions(Model(), [Path(f"{index}.jpg") for index in range(250)], imgsz=1536, batch=8, device="0", workers=4)
    assert [len(call["source"]) for call in calls] == [100, 100, 50]
    assert [item["image_id"] for item in output] == list(range(1, 251))
    assert all((call["imgsz"], call["batch"], call["workers"], call["iou"]) == (1536, 8, 4, .5) for call in calls)


def test_actual_precision_prefers_validator_backend_then_dtype_fallback():
    class Backend: fp16 = False
    model = types.SimpleNamespace(validator=types.SimpleNamespace(model=Backend(), args=types.SimpleNamespace(half=None)), predictor=None)
    assert metrics.actual_precision(model) == "FP32"
    Backend.fp16 = True
    assert metrics.actual_precision(model) == "FP16"
    class Parameter: dtype = "torch.float16"
    model = types.SimpleNamespace(validator=None, predictor=None, model=types.SimpleNamespace(parameters=lambda: iter([Parameter()])))
    assert metrics.actual_precision(model) == "FP16"


def test_complexity_is_captured_before_validation_and_prediction(monkeypatch, tmp_path):
    events = []
    class Result:
        speed = {}
    class Model:
        def val(self, **kwargs): events.append(f"val:{kwargs['split']}"); return Result()
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda _: Model()
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(metrics, "ensure_local_ultralytics", lambda: "/local/ultralytics/__init__.py")
    monkeypatch.setattr(metrics, "model_complexity", lambda model, imgsz: events.append("complexity") or {"model/GFLOPs": 1})
    monkeypatch.setattr(metrics, "standard_metrics", lambda result, split: {})
    monkeypatch.setattr(metrics, "actual_precision", lambda model: "FP32")
    monkeypatch.setattr(metrics, "speed_metrics", lambda speed: {})
    monkeypatch.setattr(metrics, "benchmark_provenance", lambda **kwargs: {})
    monkeypatch.setattr(metrics, "coco_area_metrics", lambda *args, **kwargs: events.append("predict") or {})
    metrics.evaluate_run(tmp_path, tmp_path / "data.yaml", dataset="levir", imgsz=512, batch=8, device="cpu", workers=0, include_size=False)
    assert events[0] == "complexity"
    assert events.index("complexity") < events.index("val:val") < events.index("predict")


def test_local_ultralytics_path_priority_and_external_rejection(monkeypatch):
    monkeypatch.setattr(metrics.sys, "path", ["external"])
    valid = types.ModuleType("ultralytics")
    valid.__file__ = str(metrics.LOCAL_ULTRALYTICS / "ultralytics" / "__init__.py")
    monkeypatch.setitem(sys.modules, "ultralytics", valid)
    assert Path(metrics.ensure_local_ultralytics()) == Path(valid.__file__).resolve()
    assert metrics.sys.path[0] == str(metrics.LOCAL_ULTRALYTICS)
    invalid = types.ModuleType("ultralytics"); invalid.__file__ = "/pip/ultralytics/__init__.py"
    monkeypatch.setitem(sys.modules, "ultralytics", invalid)
    try:
        metrics.ensure_local_ultralytics()
    except RuntimeError as error:
        assert "external Ultralytics" in str(error)
    else:
        raise AssertionError("external package must fail")
