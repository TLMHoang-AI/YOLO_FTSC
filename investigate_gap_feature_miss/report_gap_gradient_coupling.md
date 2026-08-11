# GAP Gradient Coupling Investigation

Checkpoint: `hf_gap/runs/gap/seed_42/weights/best.pt`
Layer: `model.model[19] ChannelAttention(avg), P2 pre-Detect`
Split: `test`; images: **788**; loss: **classification**; device: `cuda`

## Gradient Decomposition

| metric | p25 | median | p75 |
|---|---:|---:|---:|
| `indirect_over_full` | 0.024056 | 0.040136 | 0.058621 |
| `cos_full_detach` | 0.998282 | 0.999194 | 0.999711 |
| `indirect_object_bg_ratio` | 0.999999 | 1.000000 | 1.000001 |

Max forward absolute diff across normal vs detached passes: `0`.
Interpretation: **mixed_or_small_gradient_coupling**.

## Gate/Failure Correlations

| relation | Pearson | Spearman | n |
|---|---:|---:|---:|
| `gate_mean_vs_low_conf_fn` | -0.0635 | -0.1302 | 788 |
| `gate_mean_vs_median_positive_conf` | 0.1690 | 0.2494 | 340 |
| `gate_std_vs_low_conf_fn` | -0.0944 | -0.0648 | 788 |
| `gate_std_vs_median_positive_conf` | -0.0541 | -0.0290 | 340 |
| `p2_rms_vs_low_conf_fn` | 0.0485 | 0.0859 | 788 |
| `p2_rms_vs_median_positive_conf` | -0.0043 | -0.0936 | 340 |

Artifacts:
- `investigate_gap_feature_miss/tables/gap_gradient_coupling_per_image.csv`
- `investigate_gap_feature_miss/gap_gradient_coupling_summary.json`