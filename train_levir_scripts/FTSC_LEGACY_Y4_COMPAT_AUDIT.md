# Historical FTSC Y4 compatibility audit

This audit records the dependency closure used by `Y4_legacy_compat` in the main
codebase. The legacy tree is a read-only oracle and is not imported at runtime.

| Component | Main vs legacy | Decision | Reason |
|---|---|---|---|
| Canonical Y4 YAML | byte-identical (`e1a04a83…`) | Reuse values in a new named config | The historical architecture and FTSC values are unchanged. |
| `tal.py` | byte-identical (`6f0be1bf…`) | Reuse main | Assignment behavior is already identical. |
| `metrics.py` | byte-identical (`15e6e318…`) | Reuse main | Fitness calculation is shared; runner locks `map50_95`. |
| `ftsc.py` | differs | Reuse main with explicit fixed DFL scalar | Main contains the needed `fixed_strengths` API; broad legacy-only branches stay disabled. |
| `head.py` / `tasks.py` / `loss.py` | differ | Reuse main, validate Y4 branch closure | Y4 config disables the added heads/replacements and keeps standard TAL/loss. |
| `trainer.py` | differs | Reuse main narrow freeze | Actual DFL projection is frozen; no global `.dfl` name freeze is restored. |
| `default.yaml` | differs | Pin historical resolved values in runner | Prevent mutable defaults from changing the reproduction contract. |
| optimizer | differs in surrounding code | Pin `optimizer=auto`, resolve to AdamW for 100 epochs | Matches historical nc=1, short-run auto resolution (`lr=0.002`, beta1 `0.9`). |
| scheduler | differs in surrounding code | Pin `cos_lr=false`, `lrf=0.01` | Matches historical linear schedule. |
| augmentation | shared YOLO defaults | Pin `mosaic=1.0`, `close_mosaic=10` and defaults | Historical Y4 used default Mosaic/random perspective, not no-Mosaic. |

The compatibility config adds only `fixed_strengths: {dfl_distribution: 1.0}`
to the canonical Y4 YAML. It does not alter the corrected FTSC runner, historical
CSV files, assignment, inference head, or checkpoint selection.
