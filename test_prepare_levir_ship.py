from pathlib import Path

from PIL import Image

from prepare_levir_ship import prepare


def test_prepare_is_reproducible_and_scene_safe(tmp_path: Path):
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
    split_scenes = []
    for split in ("train", "val", "test"):
        assert list((tmp_path / "first/images" / split).glob("*.png"))
        split_scenes.append({path.stem.rsplit("_", 2)[0] for path in (tmp_path / "first/images" / split).iterdir()})
    assert not any(split_scenes[i] & split_scenes[j] for i in range(3) for j in range(i + 1, 3))
