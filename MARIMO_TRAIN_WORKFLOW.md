# Local to marimo training workflow

Use this checklist for changes that are developed locally and executed in a
running marimo session.

## Experiment runner naming

Every multi-run experiment must have a dedicated root-level runner named
`train_all_<experiment_slug>.py`. Use the same stable slug for its project
directory, log, PID, and Hugging Face repository; do not repurpose a generic
runner or change the meaning of another experiment's runner.

## 1. Collect connection details

Ask for the marimo URL, session ID, and a fresh auth token. Keep the token out
of commands, files, commits, and logs by reading it into an environment
variable:

```bash
read -rs MARIMO_TOKEN
export MARIMO_TOKEN
```

Also confirm that `HF_TOKEN` is already configured in the marimo runtime. Never
print either token.

## 2. Develop and verify locally

If the workspace has no `.venv`, use the project conda environment:

```bash
source /home/duylearch/miniconda3/bin/activate ml2
python -m pytest -q test_prepare_levir_ship.py
python -m py_compile prepare_levir_ship.py train_all_levir.py
python train_all_levir.py --help >/dev/null
git diff --check
```

Review `git status` and stage only files belonging to the task. Commit with a
specific Conventional Commit message, then push over SSH; never force-push.
Keep the fetch URL on HTTPS so read-only environments can pull normally, and
configure a separate SSH push URL:

```bash
git remote set-url origin https://github.com/aduy2408/yolo_code.git
git remote set-url --push origin git@github.com:aduy2408/yolo_code.git
ssh -T git@github.com
git push origin main
```

Do not use an HTTPS token for pushes or embed one in a command, notebook cell,
or remote URL. Pulls continue to use the public HTTPS fetch URL.

## 3. Connect and synchronize marimo

Use the marimo-pair helper with the token in the environment:

```bash
bash .agents/skills/marimo-pair/scripts/execute-code.sh \
  --url "$MARIMO_URL" --session "$MARIMO_SESSION" <<'PY'
import marimo as mo
mo.status.toast("🚀 Connected — ready to pair on LEVIR training!")
PY
```

From the kernel, check `/marimo/yolo_code` for overlapping local changes and
stop if any exist. Otherwise run `git pull --ff-only origin main` and verify
that `HEAD` equals the pushed commit.

## 4. Preflight and launch

Before training, verify without exposing secrets:

- `HF_TOKEN` exists.
- CUDA is available.
- local Ultralytics imports successfully.
- either `/marimo/yolo_code/LevirShipData` or `/marimo/data/LevirShipData`
  contains 3,896 PNG images and 3,896 TXT annotations.
- the LEVIR tests pass and seeds 42, 43, and 44 each split to 2,320/788/788.

Launch only when no live PID is recorded. Run from `/marimo/yolo_code` with
stdout and stderr appended to the log:

```bash
python train_all_levir.py --data-root "$LEVIR_DATA_ROOT" --device cuda \
  >>runs/levir_ship_baselines/train_all.log 2>&1
```

Start it as a detached process, save its PID to
`runs/levir_ship_baselines/train_all.pid`, and send a toast containing the log
path. If it exits early, inspect the log and report the error; do not restart
blindly. Rerunning the same command later is safe because completed runs are
reused and partial runs resume from `last.pt`.

## 5. Monitor

Check the saved PID before launching another process. Follow progress with:

```bash
tail -f /marimo/yolo_code/runs/levir_ship_baselines/train_all.log
```

Each completed run is uploaded before the next model starts. Missing
`HF_TOKEN`, incomplete artifacts, or a failed upload stops the matrix.
