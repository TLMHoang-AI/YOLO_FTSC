# FTSC no-Mosaic generation audit

The first screen is explicitly matched to `N0_y4_nomosaic`; historical Y4 with
Mosaic remains a reference only. Every variant uses 100 epochs, patience 20,
`optimizer=auto`, linear LR (`lrf=0.01`), `map50_95` checkpoint fitness, and
`mosaic=0.0`, `close_mosaic=0` from epoch zero.

| Variant | One changed factor | Runtime boundary |
|---|---|---|
| N0 | no feature branch | Y4/F5, fixed DFL FTSC scalar 1, Position trainable |
| N1 | Adaptive Zoom | train-only GT-aware crop after geometric transform and before Format; validation/inference unchanged |
| N2 | support-aware assignment | standard TAL feasible pool/conflict resolution, then posterior top-five support per GT; DFL decoder unchanged |
| N3 | localization distillation | explicit frozen teacher, detached DFL KL on matched positives; student-only inference |

N1 is a project adaptation motivated by Wang et al. (ISPRS JPRS 2026), not a
copy of learned ZoomDet. It uses `T=16 px`, `p=0.5`, `z_max=2`, visible-area
threshold `0.5`, and seeded center jitter `0.05`; `16 px` is four P2 cells.
N2 is motivated by RFLA (ECCV 2022), DCFL (CVPR 2023), and NWD-RKA (ISPRS JPRS
2022), but keeps YOLO's geometric feasibility and DFL targets. N3 follows the
Localization Distillation idea of Zheng et al. (CVPR 2022), with an explicit
teacher path and no classification distillation.

CrossKD (CVPR 2024) is deferred because a clean head adapter would require an
invasive generic trainer change. No training, Marimo connection, upload, or
GitHub push is performed by this implementation task.
