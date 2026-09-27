from pathlib import Path

import train_levir_lmsca_h1_2head_p3p4_kvca_heads as runner


def test_levir_lmsca_protocol_defaults_and_kwargs():
    args = runner.parse_args([])
    assert args.seeds == [42, 43, 44]
    assert args.epochs == 100
    assert args.imgsz == 512
    assert args.batch_size == 8
    assert args.workers == 4
    assert args.patience == 20
    kwargs = runner.train_kwargs(args, Path("data.yaml"), 42)
    assert kwargs["optimizer"] == "auto"
    assert kwargs["fitness_metric"] == "map50_95"
    assert kwargs["mosaic"] == 0.0 and kwargs["close_mosaic"] == 0


def test_levir_lmsca_model_preflight():
    report = runner.preflight_model()
    assert report["params"] == 2_226_825
    assert report["detect_from"] == [28, 29]
    assert report["detect_stride"] == [8.0, 16.0]
    assert report["ftsc_enabled"] is False
