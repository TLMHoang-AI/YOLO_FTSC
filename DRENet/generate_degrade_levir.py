"""Generate degraded images for LEVIR-Ship split (DRENet requirement).

DegradeGenerate.py is pixel-loop Python — too slow for 3896 images.
This script vectorises with numpy and runs multi-process.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SIZE = 512  # DRENet hardcodes 512

def degrade_one(args):
    img_path, lbl_path, dst_path = args
    if dst_path.exists():
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if h != SIZE or w != SIZE:
        img = cv2.resize(img, (SIZE, SIZE))

    if lbl_path.exists():
        label = np.loadtxt(str(lbl_path), ndmin=2)
        if label.size == 0:
            label = np.empty((0, 5))
    else:
        label = np.empty((0, 5))

    if label.shape[0] == 0:
        # No ships → simple blur
        dst = cv2.blur(img, (20, 20))
    else:
        # Object-aware blur  (vectorised, same formula as DegradeGenerate.py)
        centers_xy = label[:, 1:3] * SIZE  # cx, cy in pixels
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]  # (H,W)
        # dist to nearest ship center for each pixel
        # shape: (n_centers, H, W)
        dy = ys[None] - centers_xy[:, 1:2, None]  # broadcast
        dx = xs[None] - centers_xy[:, 0:1, None]
        dist2 = dy**2 + dx**2
        min_dist = np.sqrt(dist2.min(axis=0))  # (H,W)
        # box sizes per pixel (same formula as original)
        box = (1.03 ** min_dist).astype(int) // 2  # (H,W)
        box = np.clip(box, 0, 30)  # cap to reasonable kernel
        # For each unique box size, apply a blur and pick pixels
        # This is an approximation of per-pixel variable blur
        # via a set of fixed kernels (much faster, very close to original)
        dst = img.copy().astype(np.float32)
        unique_sizes = np.unique(box)
        for b in unique_sizes:
            if b == 0:
                continue
            k = 2 * b + 1
            blurred = cv2.blur(img, (k, k)).astype(np.float32)
            mask = (box == b)
            dst[mask] = blurred[mask]
        dst = dst.astype(np.uint8)

    cv2.imwrite(str(dst_path), dst)


def generate_for_split(split_dir: Path, split: str, workers: int = 8):
    img_dir = split_dir / "images" / split
    lbl_dir = split_dir / "labels" / split
    deg_dir = split_dir / "degrade" / split
    deg_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))
    tasks = [
        (p, lbl_dir / f"{p.stem}.txt", deg_dir / p.name)
        for p in img_paths
    ]
    already = sum(1 for *_, d in tasks if d.exists())
    print(f"  [{split}] {len(tasks)} images, {already} already done")
    if already == len(tasks):
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(degrade_one, tasks), total=len(tasks), desc=f"  {split}"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", type=Path,
                   default=Path("/mnt/data/varroa/yolo_related/datasets/levir_ship_yolo_seed42"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    for split in ("train", "val", "test"):
        print(f"\nGenerating degrade/{split}...")
        generate_for_split(args.split_dir, split, args.workers)
    print("\nDone.")
