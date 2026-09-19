"""Offline unit tests for generic completed-result Hugging Face archives."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from utils.hf_result_archive import (
    DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS,
    HFResultArchive,
    HFResultArchiveError,
    MARKER_FILENAME,
    RUN_SUMMARY_FILENAME,
)


class raises:
    """Tiny stdlib replacement for the one pytest helper these offline tests need."""

    def __init__(self, expected, match: str = ""):
        self.expected = expected
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _traceback):
        if exc is None:
            raise AssertionError(f"Expected {self.expected.__name__} to be raised")
        if not isinstance(exc, self.expected):
            return False
        if self.match and self.match not in str(exc):
            raise AssertionError(f"Expected {self.match!r} in {exc!r}")
        return True


class EnvPatch:
    def __init__(self):
        self.previous = dict(os.environ)

    def setenv(self, key: str, value: str) -> None:
        os.environ[key] = value

    def delenv(self, key: str) -> None:
        os.environ.pop(key, None)

    def undo(self) -> None:
        os.environ.clear()
        os.environ.update(self.previous)


class Commit:
    def __init__(self, oid: str):
        self.oid = oid


class FakeHfApi:
    def __init__(self, *, fail_folder: bool = False):
        self.fail_folder = fail_folder
        self.calls: list[tuple[str, dict]] = []

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        return {"id": kwargs["repo_id"]}

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        if self.fail_folder:
            raise OSError("simulated folder failure")
        return Commit("folder-oid")

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        return Commit("marker-oid")


def make_archive(tmp_path: Path, *, token: str = "test-token", **kwargs) -> tuple[HFResultArchive, FakeHfApi]:
    api = FakeHfApi(**kwargs.pop("api_kwargs", {}))
    return (
        HFResultArchive(
            repo_id="TLMHoang/FTSC_results",
            token=token,
            cache_dir=tmp_path / "cache",
            api=api,
            sleep=lambda _seconds: None,
            **kwargs,
        ),
        api,
    )


def write_complete_run(root: Path, *, manifest: str = "contract", extra: bool = True) -> Path:
    run_dir = root / "seed_42"
    for relative in DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest if relative == "experiment_manifest.json" else relative, encoding="utf-8")
    if extra:
        extra_path = run_dir / "evaluation" / "test_predictions.json"
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_path.write_text("predictions", encoding="utf-8")
    return run_dir


def publish(archive: HFResultArchive, run_dir: Path, *, dataset: str = "tinyperson", variant: str = "H2_MOSAIC"):
    return archive.publish_run(
        dataset_slug=dataset,
        variant=variant,
        seed=42,
        run_name="seed_42",
        run_dir=run_dir,
        summary_metadata={"run_result": {"run_dir": str(run_dir)}, "dataset_contract": {"seed": 42}},
    )


def copy_as_remote(run_dir: Path, remote_root: Path, *, dataset: str = "tinyperson", variant: str = "H2_MOSAIC") -> Path:
    remote = remote_root / dataset / variant / run_dir.name
    remote.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(run_dir, remote)
    return remote


def test_remote_paths_are_dataset_agnostic():
    assert HFResultArchive.remote_path("tinyperson", "H2_MOSAIC", "seed_42_corner_sw640_sh512") == "tinyperson/H2_MOSAIC/seed_42_corner_sw640_sh512"
    assert HFResultArchive.remote_path("varroa", "FTSC_P2P3", "seed_42") == "varroa/FTSC_P2P3/seed_42"
    assert HFResultArchive.remote_path("levir", "FTSC_EDGE025", "seed_44") == "levir/FTSC_EDGE025/seed_44"
    with raises(HFResultArchiveError):
        HFResultArchive.remote_path("tinyperson", "../bad", "seed_42")


def test_required_artifacts_and_token_handling(tmp_path: Path, monkeypatch):
    archive, _ = make_archive(tmp_path)
    assert archive.token == "test-token"
    incomplete = tmp_path / "missing"
    incomplete.mkdir()
    with raises(HFResultArchiveError, match="missing required artifact"):
        archive.required_artifact_metadata(incomplete)
    monkeypatch.setenv("HF_TOKEN", "env-token")
    from_env = HFResultArchive("repo/id", cache_dir=tmp_path, api=FakeHfApi())
    assert from_env.token == "env-token"
    monkeypatch.delenv("HF_TOKEN")
    with raises(HFResultArchiveError, match="Hugging Face token"):
        HFResultArchive("repo/id", cache_dir=tmp_path, api=FakeHfApi())


def test_publish_uploads_whole_folder_then_marker_last(tmp_path: Path):
    archive, api = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    result = publish(archive, run_dir)
    assert result["status"] == "uploaded"
    assert (run_dir / RUN_SUMMARY_FILENAME).is_file()
    assert (run_dir / MARKER_FILENAME).is_file()
    assert [name for name, _kwargs in api.calls] == ["repo_info", "upload_folder", "upload_file"]
    folder_kwargs = api.calls[1][1]
    assert folder_kwargs["ignore_patterns"] == [MARKER_FILENAME]
    assert folder_kwargs["folder_path"] == str(run_dir.resolve())
    assert api.calls[2][1]["path_in_repo"] == "tinyperson/H2_MOSAIC/seed_42/upload_complete.json"


def test_required_artifact_contract_is_caller_overrideable(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = tmp_path / "run" / "seed_custom"
    run_dir.mkdir(parents=True)
    (run_dir / "custom.bin").write_bytes(b"custom-result")
    archive.publish_run(
        dataset_slug="varroa",
        variant="FTSC_P2P3",
        seed="trial-a",
        run_name="seed_custom",
        run_dir=run_dir,
        required_artifacts=("custom.bin",),
    )
    marker = json.loads((run_dir / MARKER_FILENAME).read_text())
    assert set(marker["artifact_sha256"]) == {"custom.bin", RUN_SUMMARY_FILENAME}
    archive.validate_archive(
        run_dir=run_dir,
        dataset_slug="varroa",
        variant="FTSC_P2P3",
        run_name="seed_custom",
        seed="trial-a",
        required_artifacts=("custom.bin",),
    )


def test_failed_main_upload_never_creates_or_uploads_marker(tmp_path: Path):
    archive, api = make_archive(tmp_path, api_kwargs={"fail_folder": True}, retries=1)
    run_dir = write_complete_run(tmp_path / "run")
    with raises(OSError, match="folder failure"):
        publish(archive, run_dir)
    assert not (run_dir / MARKER_FILENAME).exists()
    assert [name for name, _kwargs in api.calls] == ["repo_info", "upload_folder"]


def test_validate_detects_marker_identity_and_checksum_corruption(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    publish(archive, run_dir)
    marker_path = run_dir / MARKER_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["dataset"] = "varroa"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with raises(HFResultArchiveError, match="dataset mismatch"):
        archive.validate_archive(run_dir=run_dir, dataset_slug="tinyperson", variant="H2_MOSAIC", run_name="seed_42", seed=42)
    marker["dataset"] = "tinyperson"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (run_dir / "weights" / "best.pt").write_text("corrupted", encoding="utf-8")
    with raises(HFResultArchiveError, match="checksum mismatch"):
        archive.validate_archive(run_dir=run_dir, dataset_slug="tinyperson", variant="H2_MOSAIC", run_name="seed_42", seed=42)


def test_validate_rejects_marker_schema_and_seed_mismatch(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    publish(archive, run_dir)
    marker_path = run_dir / MARKER_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["schema"] = "unknown_schema"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with raises(HFResultArchiveError, match="Unsupported archive marker"):
        archive.validate_archive(run_dir=run_dir, dataset_slug="tinyperson", variant="H2_MOSAIC", run_name="seed_42", seed=42)
    marker["schema"] = "ftsc_seed_archive_v1"
    marker["seed"] = 43
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with raises(HFResultArchiveError, match="seed mismatch"):
        archive.validate_archive(run_dir=run_dir, dataset_slug="tinyperson", variant="H2_MOSAIC", run_name="seed_42", seed=42)


def test_restore_ignores_unmarked_and_restores_verified_archive(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    publish(archive, run_dir)
    remote_root = tmp_path / "remote"
    remote = copy_as_remote(run_dir, remote_root)
    (remote_root / "tinyperson" / "H2_MOSAIC" / "seed_43").mkdir(parents=True)

    snapshot_calls = []
    archive._snapshot_download_fn = lambda **kwargs: snapshot_calls.append(kwargs) or remote_root
    result = archive.restore_completed_runs(
        dataset_slug="tinyperson",
        project=tmp_path / "project",
        variants=["H2_MOSAIC"],
        seeds=[42, 43],
        run_name_fn=lambda _variant, seed: f"seed_{seed}",
    )
    assert snapshot_calls[0]["allow_patterns"] == ["tinyperson/**"]
    assert len(result["restored"]) == 1
    restored = Path(result["restored"][0]["run_dir"])
    assert (restored / "weights" / "best.pt").is_file()
    assert (restored / "evaluation" / "test_predictions.json").is_file()
    assert HFResultArchive.restored_run_dirs(result) == {str(restored.resolve())}


def test_restore_rejects_variant_checksum_and_local_contract_mismatches(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    publish(archive, run_dir)
    remote_root = tmp_path / "remote"
    remote = copy_as_remote(run_dir, remote_root)
    archive._snapshot_download_fn = lambda **_kwargs: remote_root

    marker = json.loads((remote / MARKER_FILENAME).read_text())
    marker["dataset"] = "varroa"
    (remote / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    with raises(HFResultArchiveError, match="dataset mismatch"):
        archive.restore_completed_runs(dataset_slug="tinyperson", project=tmp_path / "project", variants=["H2_MOSAIC"], seeds=[42], run_name_fn=lambda _variant, seed: f"seed_{seed}")

    marker["dataset"] = "tinyperson"
    marker["case"] = "OTHER"
    (remote / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    with raises(HFResultArchiveError, match="case mismatch"):
        archive.restore_completed_runs(dataset_slug="tinyperson", project=tmp_path / "project", variants=["H2_MOSAIC"], seeds=[42], run_name_fn=lambda _variant, seed: f"seed_{seed}")

    marker["case"] = "H2_MOSAIC"
    (remote / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    (remote / "weights" / "best.pt").write_text("corrupt", encoding="utf-8")
    with raises(HFResultArchiveError, match="checksum mismatch"):
        archive.restore_completed_runs(dataset_slug="tinyperson", project=tmp_path / "project", variants=["H2_MOSAIC"], seeds=[42], run_name_fn=lambda _variant, seed: f"seed_{seed}")

    # Restore a clean remote archive, but retain a conflicting local manifest.
    remote = copy_as_remote(run_dir, tmp_path / "remote_clean")
    archive._snapshot_download_fn = lambda **_kwargs: tmp_path / "remote_clean"
    local = tmp_path / "project" / "H2_MOSAIC" / "seed_42"
    local.mkdir(parents=True)
    (local / "experiment_manifest.json").write_text("different", encoding="utf-8")
    with raises(HFResultArchiveError, match="different experiment contract"):
        archive.restore_completed_runs(dataset_slug="tinyperson", project=tmp_path / "project", variants=["H2_MOSAIC"], seeds=[42], run_name_fn=lambda _variant, seed: f"seed_{seed}")


def test_custom_run_name_and_marker_schema_contract(tmp_path: Path):
    archive, _ = make_archive(tmp_path)
    run_dir = write_complete_run(tmp_path / "run")
    renamed = run_dir.with_name("seed_42_corner_sw640_sh512")
    run_dir.rename(renamed)
    result = archive.publish_run(
        dataset_slug="tinyperson",
        variant="H2_MOSAIC",
        seed=42,
        run_name=renamed.name,
        run_dir=renamed,
        summary_metadata={"source_preflight": {"ok": True}},
    )
    marker = json.loads((renamed / MARKER_FILENAME).read_text())
    assert result["remote_path"] == "tinyperson/H2_MOSAIC/seed_42_corner_sw640_sh512"
    assert marker["schema"] == "ftsc_seed_archive_v1"
    assert marker["run_relative_path"] == "H2_MOSAIC/seed_42_corner_sw640_sh512"
    assert set(DEFAULT_FTSC_SEED_REQUIRED_ARTIFACTS) <= set(marker["artifact_sha256"])
    assert RUN_SUMMARY_FILENAME in marker["artifact_sha256"]


def test_module_has_no_tinyperson_causal_suite_dependency():
    module_source = Path(__file__).parents[1].joinpath("hf_result_archive.py").read_text(
        encoding="utf-8"
    )
    assert "tinyperson_causal" not in module_source
    assert "train_tinyperson" not in module_source


def _run_with_tmp(test_fn, *, needs_monkeypatch: bool = False) -> None:
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch = EnvPatch()
        try:
            if needs_monkeypatch:
                test_fn(Path(directory), monkeypatch)
            else:
                test_fn(Path(directory))
        finally:
            monkeypatch.undo()


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    functions = (
        test_remote_paths_are_dataset_agnostic,
        test_required_artifacts_and_token_handling,
        test_publish_uploads_whole_folder_then_marker_last,
        test_required_artifact_contract_is_caller_overrideable,
        test_failed_main_upload_never_creates_or_uploads_marker,
        test_validate_detects_marker_identity_and_checksum_corruption,
        test_validate_rejects_marker_schema_and_seed_mismatch,
        test_restore_ignores_unmarked_and_restores_verified_archive,
        test_restore_rejects_variant_checksum_and_local_contract_mismatches,
        test_custom_run_name_and_marker_schema_contract,
        test_module_has_no_tinyperson_causal_suite_dependency,
    )
    for function in functions:
        if function in {
            test_remote_paths_are_dataset_agnostic,
            test_module_has_no_tinyperson_causal_suite_dependency,
        }:
            suite.addTest(unittest.FunctionTestCase(function))
        elif function is test_required_artifacts_and_token_handling:
            suite.addTest(unittest.FunctionTestCase(
                lambda function=function: _run_with_tmp(function, needs_monkeypatch=True)
            ))
        else:
            suite.addTest(unittest.FunctionTestCase(
                lambda function=function: _run_with_tmp(function)
            ))
    return suite


if __name__ == "__main__":
    unittest.main()
