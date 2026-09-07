#!/usr/bin/env python3
"""Deterministic, dependency-free operations for Marimo training jobs.

This module is intentionally about orchestration and evidence, not model code:
- preflight checks the effective environment and exact Git revision;
- launch starts a detached process with durable PID/log/state files;
- status distinguishes process liveness from observable progress;
- artifacts verifies the minimum local completion contract.

It is safe to import from a runner. It never prints secret environment values.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_REQUIRED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
)


class MarimoOpsError(RuntimeError):
    """A preflight, launch, status, or artifact contract failure."""


def require_training_context(*, hf_repo_id: str | None = None) -> None:
    """Require the shared Marimo launch context for upload-required runners."""
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise MarimoOpsError(
            "Marimo training must be launched with `python -m utils.marimo_ops launch`"
        )
    if not os.environ.get("HF_TOKEN"):
        raise MarimoOpsError("HF_TOKEN is required for upload-required Marimo training")
    expected_repo = os.environ.get("MARIMO_HF_REPO_ID")
    if expected_repo and hf_repo_id and expected_repo != hf_repo_id:
        raise MarimoOpsError(
            f"HF repository mismatch: expected {expected_repo}, got {hf_repo_id}"
        )


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise MarimoOpsError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MarimoOpsError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MarimoOpsError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        output = (exc.output or "").strip()
        raise MarimoOpsError(
            f"Command failed ({exc.returncode}): {shlex.join(command)}\n{output}"
        ) from exc


def git_sha(repo: Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=repo)


def git_dirty(repo: Path) -> str:
    return run_checked(["git", "status", "--porcelain"], cwd=repo)


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # ``kill -0`` also succeeds for a zombie. Treat a zombie as finished so
    # stale PID files cannot block a restart forever.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text().split()
    except OSError:
        return True
    return len(fields) < 3 or fields[2] != "Z"


def process_command(pid: int) -> str:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""
    return raw.strip()


def newest_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    files = [p for p in path.rglob("*") if p.is_file()]
    return max((p.stat().st_mtime for p in files), default=None)


def require_files(root: Path, relative_paths: Iterable[str]) -> list[str]:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise MarimoOpsError(
            "Missing required artifacts:\n" + "\n".join(f"- {item}" for item in missing)
        )
    return list(relative_paths)


def preflight(
    *,
    repo: Path,
    expected_sha: str | None = None,
    python: str | None = None,
    required_paths: Iterable[str] = (),
    epochs: int | None = None,
    patience: int | None = None,
    upload_required: bool = False,
    hf_repo_id: str | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Run fail-closed checks before any expensive remote job."""
    if not repo.is_dir():
        raise MarimoOpsError(f"Repository does not exist: {repo}")
    actual_sha = git_sha(repo)
    dirty = git_dirty(repo)
    if expected_sha and actual_sha != expected_sha:
        raise MarimoOpsError(f"Wrong Git SHA: expected {expected_sha}, got {actual_sha}")
    if dirty and not allow_dirty:
        raise MarimoOpsError(f"Remote worktree is dirty:\n{dirty}")
    if python:
        py = Path(python)
        if not py.is_file() or not os.access(py, os.X_OK):
            raise MarimoOpsError(f"Python executable is not executable: {python}")
        executable = run_checked([python, "-c", "import sys; print(sys.executable)"])
    else:
        executable = sys.executable
    if upload_required and not hf_repo_id:
        raise MarimoOpsError("HF repository is required when upload is required")
    if epochs is not None and epochs <= 0:
        raise MarimoOpsError(f"epochs must be positive, got {epochs}")
    if patience is not None and patience < 0:
        raise MarimoOpsError(f"patience must be non-negative, got {patience}")
    required = require_files(repo, required_paths)
    result = {
        "checked_at": now_utc(),
        "repo": str(repo),
        "git_sha": actual_sha,
        "worktree_dirty": bool(dirty),
        "python": executable,
        "epochs": epochs,
        "patience": patience,
        "upload_required": upload_required,
        "hf_repo_id": hf_repo_id,
        "required_paths": required,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@dataclass
class LaunchResult:
    pid: int
    command: list[str]
    cwd: str
    log_path: str
    pid_path: str
    state_path: str
    started_at: str


def launch_detached(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    pid_path: Path,
    state_path: Path,
    env: Mapping[str, str] | None = None,
) -> LaunchResult:
    """Launch one durable process and record enough evidence to inspect it later."""
    if not command:
        raise MarimoOpsError("Cannot launch an empty command")
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if is_pid_alive(old_pid):
            raise MarimoOpsError(f"Refusing duplicate launch: PID {old_pid} is alive")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()
    log_file = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    # The child owns the duplicated descriptor after Popen returns. Keeping the
    # parent's descriptor open causes warnings and delays EOF on short jobs.
    log_file.close()
    # Reap the child without coupling the caller to the training duration. This
    # prevents short jobs from becoming zombies while long jobs remain detached.
    threading.Thread(target=proc.wait, name=f"marimo-reaper-{proc.pid}", daemon=True).start()
    pid_path.write_text(f"{proc.pid}\n")
    state = {
        "status": "running",
        "pid": proc.pid,
        "command": list(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "started_at": started_at,
    }
    write_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return LaunchResult(
        pid=proc.pid,
        command=list(command),
        cwd=str(cwd),
        log_path=str(log_path),
        pid_path=str(pid_path),
        state_path=str(state_path),
        started_at=started_at,
    )


def status(run_dir: Path, *, pid_file: str = "train.pid", log_file: str = "train.log") -> dict[str, object]:
    """Report liveness and progress separately. No claim of completion is inferred."""
    pid_path = run_dir / pid_file
    log_path = run_dir / log_file
    state_path = run_dir / "state.json"
    pid: int | None = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pass
    alive = is_pid_alive(pid) if pid is not None else False
    latest = newest_mtime(run_dir)
    state = read_json(state_path) if state_path.is_file() else None
    required_present = all(
        (run_dir / item).is_file() for item in DEFAULT_REQUIRED_ARTIFACTS
    )
    if alive:
        observed_status = "running"
    elif (run_dir / "upload_complete.json").is_file() and required_present:
        observed_status = "complete_verified"
    elif latest is not None or state is not None:
        observed_status = "not_running_unverified"
    else:
        observed_status = "not_started_or_unknown"
    result = {
        "run_dir": str(run_dir),
        "pid": pid,
        "process_alive": alive,
        "observed_status": observed_status,
        "process_command": process_command(pid) if alive and pid else "",
        "latest_artifact_mtime": latest,
        "log_exists": log_path.is_file(),
        "log_mtime": log_path.stat().st_mtime if log_path.is_file() else None,
        "state": state,
        "upload_verified": (run_dir / "upload_complete.json").is_file(),
        "required_artifacts": {
            item: (run_dir / item).is_file() for item in DEFAULT_REQUIRED_ARTIFACTS
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def artifacts(run_dir: Path, required: Iterable[str] = DEFAULT_REQUIRED_ARTIFACTS) -> dict[str, object]:
    present = require_files(run_dir, required)
    result = {"run_dir": str(run_dir), "required_artifacts": present}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marimo training evidence helper")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--expected-sha")
    p.add_argument("--python")
    p.add_argument("--required-path", action="append", default=[])
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--upload-required", action="store_true")
    p.add_argument("--hf-repo-id")
    p.add_argument("--allow-dirty", action="store_true")

    p = sub.add_parser("status")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pid-file", default="train.pid")
    p.add_argument("--log-file", default="train.log")

    p = sub.add_parser("artifacts")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--required-path", action="append", default=list(DEFAULT_REQUIRED_ARTIFACTS))

    p = sub.add_parser("launch")
    p.add_argument("--cwd", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--log-file", default="train.log")
    p.add_argument("--pid-file", default="train.pid")
    p.add_argument("--state-file", default="state.json")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "preflight":
            preflight(
                repo=args.repo,
                expected_sha=args.expected_sha,
                python=args.python,
                required_paths=args.required_path,
                epochs=args.epochs,
                patience=args.patience,
                upload_required=args.upload_required,
                hf_repo_id=args.hf_repo_id,
                allow_dirty=args.allow_dirty,
            )
        elif args.action == "status":
            status(args.run_dir, pid_file=args.pid_file, log_file=args.log_file)
        elif args.action == "artifacts":
            artifacts(args.run_dir, args.required_path)
        elif args.action == "launch":
            command = list(args.command)
            if command and command[0] == "--":
                command = command[1:]
            launch_env = os.environ.copy()
            launch_env["MARIMO_TRAIN_WORKFLOW"] = "1"
            if "--hf-repo-id" in command:
                index = command.index("--hf-repo-id")
                if index + 1 < len(command):
                    launch_env["MARIMO_HF_REPO_ID"] = command[index + 1]
            launch_detached(
                command,
                cwd=args.cwd,
                log_path=args.run_dir / args.log_file,
                pid_path=args.run_dir / args.pid_file,
                state_path=args.run_dir / args.state_file,
                env=launch_env,
            )
        return 0
    except MarimoOpsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
