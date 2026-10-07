"""Static/unit contracts for the matched YOLOv8n report backfill."""
import json
import statistics

from train_levir_scripts import train_levir_v8_p23_portability_backfill_v1 as suite


def test_exact_case_and_seed_contract():
    assert suite.CASES == ("V8_B1_P23_NOFTSC", "V8_H2_P23_FTSC")
    assert suite.DEFAULT_SEEDS == (42, 43)
    assert suite.CANONICAL_SEEDS == (42, 43, 44)


def test_graph_parity_contract_is_ftsc_only():
    report = suite.static_graph_parity()
    assert report == {"detect_inputs": [19, 22], "b1_ftsc": False, "h2_ftsc": True}


def test_sample_standard_deviation_and_singleton_semantics():
    assert suite.sample_std([0.80, 0.82]) == statistics.stdev([0.80, 0.82])
    assert suite.sample_std([0.80]) is None


def test_h2_source_contract_is_evaluation_only():
    text = open(suite.__file__, encoding="utf-8").read()
    assert '"execution_mode": "evaluation_only_reuse"' in text
    assert "HISTORICAL_SUITE = \"levir_yolov8n_p2_ftsc_head_pruning\"" in text
    assert "def train_b1" in text and "def prepare_h2_reuse" in text
    assert "train_b1(args, row" in text
    assert "prepare_h2_reuse(row)" in text
    assert "original_source_sha256" in text and "copied_checkpoint_sha256" in text


def test_full_metric_contract_and_extended_evaluation():
    assert "test_size/AP50-Small" in suite.REQUIRED_METRICS
    assert "evaluation_metrics_extended.json" in suite.REQUIRED_ARTIFACTS
    assert "include_size=True" in open(suite.__file__, encoding="utf-8").read()


def test_status_and_aggregate_contract_are_fail_closed():
    text = open(suite.__file__, encoding="utf-8").read()
    assert 'return "EVAL_BACKFILL_REQUIRED"' in text
    assert 'statistics.stdev(values) if len(values) >= 2 else None' in text
    assert '"requested_seed_count": len(args.seeds)' in text
    assert '"canonical_seed_count": len(CANONICAL_SEEDS)' in text
    assert '"canonical_complete": set(CANONICAL_SEEDS).issubset(seeds)' in text


def test_runner_does_not_own_huggingface_networking():
    text = open(suite.__file__, encoding="utf-8").read().lower()
    assert "huggingface" not in text
    assert "hf_result_archive" not in text


def test_preflight_retains_deferred_seed_rows(tmp_path):
    args = suite.parse_args([])
    args.cases = list(suite.CASES)
    args.seeds = [42, 43]
    args.dataset_root = tmp_path
    args.pretrained = tmp_path / "missing.pt"
    rows = suite.build_preflight(args)
    assert {(row["case"], row["seed"]) for row in rows if row["requested"]} == {
        (case, seed) for case in suite.CASES for seed in (42, 43)
    }
    assert all(row["status"] == "NOT_REQUESTED" for row in rows if row["seed"] == 44)
