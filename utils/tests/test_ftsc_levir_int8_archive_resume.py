import json
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from unittest import mock
from types import SimpleNamespace
from utils import ftsc_levir_int8_archive_resume as mod


class ArchiveDriverTests(unittest.TestCase):
    def test_best_only_preserves_original_inventory_and_revision(self):
        original = {'hf_revision': 'a'*40, 'checkpoints': [
            {'kind': 'last.pt'}, {'kind': 'best.pt'}, {'kind': 'epoch100.pt'}]}
        result = mod.best_inventory(original)
        self.assertEqual(result['checkpoints'], [{'kind': 'best.pt'}])
        self.assertEqual(result['hf_revision'], original['hf_revision'])
        self.assertEqual(len(original['checkpoints']), 3)

    def test_upload_failure_prevents_next_checkpoint(self):
        rows = [{'kind':'best.pt','job_id':str(i),'path':str(i)} for i in range(2)]
        with tempfile.TemporaryDirectory() as td, mock.patch.object(mod, 'Persistence') as cls, \
             mock.patch.object(mod.evaluator, 'source_hashes', return_value={}), \
             mock.patch.object(mod.core, 'runtime_identity', return_value={}), \
             mock.patch.object(mod.evaluator, 'evaluate_one', return_value={}) as evaluate, \
             mock.patch.object(mod.evaluator, 'dataset_context', return_value=(None, [], {}, {})), \
             mock.patch.object(mod, 'legacy_detect_support', return_value=nullcontext()):
            store = cls.return_value
            store.inventory.return_value = {'checkpoints':rows}
            store.publish.side_effect = RuntimeError('HF unavailable')
            with self.assertRaisesRegex(RuntimeError, 'HF unavailable'):
                mod.main(['--phase','run','--project',td,'--data-yaml',td])
            self.assertEqual(evaluate.call_count, 1)
            store.reports.assert_not_called()

    def test_restore_passes_only_exact_job_namespace_to_existing_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            store = object.__new__(mod.Persistence)
            store.args = SimpleNamespace(project=Path(td), repo='owner/repo')
            store.api = mock.Mock()
            store.revision = 'a'*40
            row = {'job_id':'seed_42__best__job','checkpoint_seed':42}
            prefix = f'{mod.PREFIX}/{row["job_id"]}'
            store.files = {prefix+'/upload_complete.json'}
            marker = Path(td)/'marker.json'
            marker.write_text(json.dumps({'artifact_sha256':{'metrics_int8.json':'sha'}}))
            store.download = mock.Mock(return_value=marker)
            def restore(**kwargs):
                folder = Path(kwargs['project'])/mod.VARIANT/row['job_id']
                folder.mkdir(parents=True)
                (folder/'metrics_int8.json').write_text('{}')
                return {'restored':[{'run_dir':str(folder)}]}
            with mock.patch.object(mod, 'HFResultArchive') as archive:
                archive.return_value.restore_completed_runs.side_effect = restore
                # The hub import occurs only during real restore, not at module load.
                import sys, types
                hub = types.ModuleType('huggingface_hub')
                hub.snapshot_download = mock.Mock(return_value=td)
                with mock.patch.dict(sys.modules, {'huggingface_hub':hub}):
                    self.assertTrue(store.restore(row))
                    scoped = archive.call_args.kwargs['snapshot_download_fn']
                    scoped(repo_id='owner/repo',allow_patterns=['TinyPerson/**'])
                    self.assertEqual(hub.snapshot_download.call_args.kwargs['allow_patterns'],[prefix+'/**'])
                    self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'],store.revision)
                self.assertTrue((Path(td)/'LEVIR-Ship/jobs'/row['job_id']/'metrics_int8.json').exists())
