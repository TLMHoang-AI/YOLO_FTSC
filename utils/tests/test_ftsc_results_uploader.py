"""Unit contracts for the repository-locked FTSC archive facade."""
from unittest.mock import patch

import pytest

from utils import ftsc_results_uploader as uploader


def test_repository_identity_is_fixed_and_tokens_are_required():
    assert uploader.HF_REPO_ID == "TLMHoang/FTSC_results"
    assert uploader.HF_REPO_TYPE == "dataset"
    with pytest.raises(ValueError): uploader.FTSCResultsUploader(token="")
    with pytest.raises(ValueError): uploader.FTSCResultsUploader(token=object())


def test_wrapper_delegates_fixed_identity_and_preserves_archive_contract(tmp_path):
    with patch.object(uploader, "HFResultArchive") as archive:
        instance = uploader.FTSCResultsUploader(token="caller-token", cache_dir=tmp_path)
        archive.assert_called_once_with(uploader.HF_REPO_ID, token="caller-token", repo_type="dataset", cache_dir=tmp_path)
        archive.return_value.restore_completed_runs.return_value = {"restored": [], "verified_local_run_dirs": []}
        result = instance.restore_completed_runs(namespace="ns", project=tmp_path, variants=["case"], seeds=[42])
        assert result == {"restored": [], "verified_local_run_dirs": []}
        archive.return_value.restore_completed_runs.assert_called_once()
        run_dir = tmp_path / "seed_42"; run_dir.mkdir()
        instance.publish_run(namespace="ns", case="case", seed=42, run_dir=run_dir, required_artifacts=("results.csv",), summary_metadata={"safe": True})
        kwargs = archive.return_value.publish_run.call_args.kwargs
        assert kwargs["dataset_slug"] == "ns" and kwargs["variant"] == "case"
        assert kwargs["run_name"] == "seed_42" and kwargs["seed"] == 42
        assert "caller-token" not in repr(kwargs["summary_metadata"])


def test_publish_uses_the_actual_run_directory_basename_for_each_suite(tmp_path):
    with patch.object(uploader, "HFResultArchive") as archive:
        instance = uploader.FTSCResultsUploader(token="caller-token", cache_dir=tmp_path)
        for basename in ("seed_42", "seed_42_corner_sw640_sh512"):
            run_dir = tmp_path / basename; run_dir.mkdir()
            instance.publish_run(namespace="ns", case="case", seed=42, run_dir=run_dir, required_artifacts=("results.csv",))
            assert archive.return_value.publish_run.call_args.kwargs["run_name"] == basename
        with pytest.raises(ValueError):
            instance.publish_run(namespace="ns", case="case", seed=42, run_dir=tmp_path / "seed_42", run_name="seed_43", required_artifacts=("results.csv",))


def test_repository_override_and_token_metadata_are_rejected(tmp_path):
    with pytest.raises(TypeError): uploader.FTSCResultsUploader(token="caller-token", repo_id="elsewhere")
    with patch.object(uploader, "HFResultArchive"):
        instance = uploader.FTSCResultsUploader(token="caller-token", cache_dir=tmp_path)
        with pytest.raises(ValueError):
            instance.publish_run(namespace="ns", case="case", seed=42, run_dir=tmp_path, required_artifacts=("results.csv",), summary_metadata={"nested": {"token": "no"}})

def test_snapshot_download_timeout_options_are_forwarded(tmp_path):
    with patch.object(uploader, "HFResultArchive") as archive:
        uploader.FTSCResultsUploader(token="caller-token", cache_dir=tmp_path, snapshot_download_kwargs={"etag_timeout": 60})
        assert archive.call_args.kwargs["snapshot_download_kwargs"] == {"etag_timeout": 60}


def test_wrapper_source_has_no_secret_or_repository_override_path():
    source = uploader.__file__
    text = open(source, encoding="utf-8").read()
    assert "os.environ" not in text and "token_file" not in text and "duyle2408" not in text
    assert "repo_id" not in uploader.FTSCResultsUploader.__init__.__annotations__
