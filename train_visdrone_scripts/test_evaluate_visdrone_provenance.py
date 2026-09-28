"""Static regression guard for VisDrone TinyBenchmark dataset provenance."""
from pathlib import Path


def test_visdrone_passes_dataset_identity_to_shared_size_evaluator():
    source = (Path(__file__).resolve().parent / "evaluate_visdrone.py").read_text(encoding="utf-8")
    assert 'dataset="visdrone"' in source
