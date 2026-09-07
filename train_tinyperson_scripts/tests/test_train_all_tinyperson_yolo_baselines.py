from __future__ import annotations

from train_tinyperson_scripts import train_all_tinyperson_yolo_baselines as runner


def test_models_are_five_requested_baselines() -> None:
    assert runner.MODELS == {
        "yolov5": "yolov5nu.pt",
        "yolov8": "yolov8n.pt",
        "yolov9": "yolov9t.pt",
        "yolov10": "yolov10n.pt",
        "yolov11": "yolo11n.pt",
    }
    assert runner.SEEDS == (42, 43, 44)


def test_two_machine_sharding_covers_each_model_seed_once() -> None:
    all_jobs = set(runner.selected_jobs(list(runner.MODELS), [42, 43, 44], 0, 1))
    shards = [
        set(runner.selected_jobs(list(runner.MODELS), [42, 43, 44], index, 2))
        for index in (0, 1)
    ]
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == all_jobs
    assert len(all_jobs) == 15


def test_parse_defaults_are_three_seeds_and_upload_repo() -> None:
    args = runner.parse_args([])
    assert args.seeds == [42, 43, 44]
    assert args.models == list(runner.MODELS)
    assert args.machine_count == 1
    assert args.patience == 0
    assert args.hf_repo_id == "duyle2408/tinyperson-yolo-baselines"
