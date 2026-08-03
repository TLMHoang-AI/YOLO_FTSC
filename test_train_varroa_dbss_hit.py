import argparse
import json
from pathlib import Path

import pytest

import train_varroa_dbss_hit as train


def test_matrix_and_default_seeds():
    assert set(train.MODELS) == {"yolov5n", "yolov8n", "yolov10n"}
    assert all({"dbss", "hit", "weights"} == set(spec) for spec in train.MODELS.values())
    assert train.parse_args([]).seeds == [42, 43, 44]


def test_validate_dataset_requires_fixed_split_counts(tmp_path: Path, monkeypatch):
    counts = {"train": 2, "val": 1, "test": 1}
    monkeypatch.setattr(train, "EXPECTED_SPLITS", counts)
    for split, count in counts.items():
        directory = tmp_path / "images" / split
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.png").touch()
    data_yaml = tmp_path / "varroa.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
        encoding="utf-8",
    )
    train.validate_dataset(data_yaml)
    (tmp_path / "images/train/1.png").unlink()
    with pytest.raises(ValueError, match="expected 2"):
        train.validate_dataset(data_yaml)


def test_amp_failure_is_archived_and_retried_without_amp(tmp_path: Path, monkeypatch):
    args = argparse.Namespace(
        project=tmp_path,
        data_yaml=tmp_path / "varroa.yaml",
        epochs=1,
        imgsz=64,
        batch_size=1,
        device="cpu",
        workers=0,
        patience=0,
    )
    calls = []

    class FakeModel:
        def train(self, **kwargs):
            calls.append(kwargs["amp"])
            run_dir = tmp_path / "yolov8n/dbss/seed_42"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "partial.txt").write_text("partial")
            if kwargs["amp"]:
                raise RuntimeError("AMP failed")
            for relative in train.REQUIRED_ARTIFACTS:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok")

    monkeypatch.setattr(train, "model_for", lambda *_: FakeModel())
    monkeypatch.setattr(train, "seed_everything", lambda _: None)
    run_dir = train.train_one("yolov8n", "dbss", 42, True, args)
    assert calls == [True, False]
    assert train.complete(run_dir)
    assert len(list((tmp_path / "yolov8n/dbss").glob("seed_42_amp_failed_*"))) == 1
