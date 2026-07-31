#!/usr/bin/env python3
"""Plot model-family mAP50-95 vs GFLOPs trade-off from the SA-YOLO table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "visualize" / "model_family_map50_95_gflops.png"


FAMILIES = {
    "YOLOv8": {
        "x": [7.3, 151.8, 236.9],
        "y": [32.62, 32.87, 33.23],
        "labels": ["n", "l", "x"],
        "color": "tab:blue",
        "marker": "o",
        "linestyle": ":",
    },
    "YOLOv9": {
        "x": [6.4, 84.6, 161.5],
        "y": [32.93, 32.98, 32.85],
        "labels": ["t", "c", "e"],
        "color": "tab:green",
        "marker": "s",
        "linestyle": "--",
    },
    "YOLOv10": {
        "x": [9.1, 139.2, 189.8],
        "y": [33.33, 33.38, 33.66],
        "labels": ["n", "l", "x"],
        "color": "tab:orange",
        "marker": "^",
        "linestyle": "-.",
    },
    "YOLO11": {
        "x": [6.3, 85.7, 192.0],
        "y": [32.60, 33.40, 32.68],
        "labels": ["n", "l", "x"],
        "color": "tab:purple",
        "marker": "D",
        "linestyle": "-",
    },
}

SINGLETONS = [
    ("Faster R-CNN", 60.09, 33.83, "P", "tab:brown"),
    ("Cascade R-CNN", 87.90, 33.40, "X", "tab:pink"),
    ("SA-YOLO", 9.31, 35.66, "*", "red"),
]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5.6))

    for family, spec in FAMILIES.items():
        ax.plot(
            spec["x"],
            spec["y"],
            label=family,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=1.8,
            markersize=6,
        )
        for x, y, suffix in zip(spec["x"], spec["y"], spec["labels"]):
            ax.annotate(suffix, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)

    for name, x, y, marker, color in SINGLETONS:
        size = 140 if name == "SA-YOLO" else 75
        ax.scatter(x, y, label=name, marker=marker, s=size, color=color, edgecolor="black", linewidth=0.7, zorder=5)
        ax.annotate(name, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=10, weight="bold" if name == "SA-YOLO" else "normal")

    ax.set_xlabel("GFLOPs")
    ax.set_ylabel(r"mAP$_{50-95}$ (%)")
    ax.set_xlim(0, 250)
    ax.set_ylim(32, 36.5)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
