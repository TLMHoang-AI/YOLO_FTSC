from pathlib import Path

from PIL import Image

from misc.overfit_levir_positive import prepare_dataset
from misc.prepare_levir_ship import PUBLISHED_COUNTS, prepare, split_samples


def test_prepare_is_reproducible_and_disjoint(tmp_path: Path):
    source = tmp_path / "source"
    images, labels = source / "All Images", source / "All Annotations"
    images.mkdir(parents=True); labels.mkdir()
    for scene in range(8):
        for tile in range(2):
            stem = f"scene{scene}_{tile}_0"
            Image.new("RGB", (16, 16)).save(images / f"{stem}.png")
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    first = prepare(source, tmp_path / "first")
    second = prepare(source, tmp_path / "second")
    assert first.read_text().replace("first", "second") == second.read_text()
    split_images = []
    for split in ("train", "val", "test"):
        assert list((tmp_path / "first/images" / split).glob("*.png"))
        split_images.append({path.name for path in (tmp_path / "first/images" / split).iterdir()})
    assert not any(split_images[i] & split_images[j] for i in range(3) for j in range(i + 1, 3))


def test_published_split_counts_and_seeds():
    samples = [(Path(str(index)), Path(str(index)), "scene") for index in range(sum(PUBLISHED_COUNTS.values()))]
    first = split_samples(samples, 42)
    repeated = split_samples(samples, 42)
    different = split_samples(samples, 43)
    assert {name: len(records) for name, records in first.items()} == PUBLISHED_COUNTS
    assert first == repeated
    assert first != different


def test_prepare_positive_overfit_uses_only_labeled_images(tmp_path: Path):
    source = tmp_path / "source"
    images, labels = source / "All Images", source / "All Annotations"
    images.mkdir(parents=True); labels.mkdir()
    for index in range(5):
        Image.new("RGB", (16, 16)).save(images / f"sample_{index}.png")
        annotation = "0 0.5 0.5 0.25 0.25\n" if index % 2 == 0 else ""
        (labels / f"sample_{index}.txt").write_text(annotation, encoding="utf-8")
    yaml = prepare_dataset(source, tmp_path / "positive", count=3)
    assert len(list((yaml.parent / "images").iterdir())) == 3
    assert all(path.read_text().strip() for path in (yaml.parent / "labels").iterdir())
    assert "train: images\nval: images\ntest: images" in yaml.read_text()
