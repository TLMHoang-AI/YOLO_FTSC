"""CPU-only checks for isolation, evidence integrity and fail-closed persistence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import nullcontext
from types import SimpleNamespace
from utils import ftsc_hf_all_trt_int8_paper_v1 as original
from utils.ftsc_varroa_int8_evaluator import core, compare_inputs, validate_evidence
from utils import ftsc_varroa_int8_archive_resume as driver


class VarroaTests(unittest.TestCase):
    def test_export_requests_real_train_calibrated_int8(self):
        import sys
        import types
        from evaluate_test import standard_detection_metrics as standard
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            source = folder/'best.pt'
            source.write_bytes(b'checkpoint')
            engine = source.with_suffix('.engine')
            model = mock.Mock()
            def export(**kwargs):
                payload = json.dumps({'task':'detect','args':kwargs}).encode()
                engine.write_bytes(len(payload).to_bytes(4,'little',signed=True)+payload+b'engine')
                return engine
            model.export.side_effect = export
            ultra = types.ModuleType('ultralytics')
            ultra.YOLO = mock.Mock(return_value=model)
            with mock.patch.object(standard,'ensure_local_ultralytics'), \
                 mock.patch.dict(sys.modules,{'ultralytics':ultra}):
                yaml = folder/'varroa.yaml'
                yaml.write_text('train: images/train\ntest: images/test\n')
                core.engine_for(source,core.sha256(source),folder,'Varroa',yaml,43,
                                {'calibration_split':'train'}, {}, 'cuda:0',{})
            call = model.export.call_args.kwargs
            self.assertEqual((call['imgsz'],call['batch']),(640,8))
            self.assertEqual((call['split'],call['fraction']),('train',1.0))
            self.assertTrue(call['int8'] and call['dynamic'])
            self.assertFalse(call['half'] or call['nms'])

    def test_unknown_seed_never_prepares_a_guessed_split(self):
        from argparse import Namespace
        from train_levir_scripts import train_varroa_dbss_hit as native
        with mock.patch.object(native,'dataset_for_seed') as prepare:
            for seed,resolution in ((45,'checkpoint_path_unprepared'),
                                    (42,'CALIBRATION_FALLBACK_42_UNVERIFIED_TRAINING_SEED')):
                with self.assertRaisesRegex(RuntimeError,'verified checkpoint seed'):
                    core.validate_data(Namespace(),{'checkpoint_seed':seed,'seed_resolution':resolution})
            prepare.assert_not_called()

    def test_original_evaluators_are_not_modified(self):
        self.assertIsNot(core, original)
        self.assertEqual(original.DATASETS, ('LEVIR-Ship','TinyPerson'))
        self.assertNotIn('Varroa',original.METRICS)
        self.assertEqual(core.DATASETS,('Varroa',))
        self.assertIn('test_size/AP50-Small',core.METRICS['Varroa'])

    def test_inventory_scope_and_best_only_preserve_lineage(self):
        paths = ['Varroa/original_runs/V1/seed_42/weights/best.pt',
                 'Varroa/original_runs/V1/seed_42/weights/last.pt',
                 'TinyPerson/original_runs/V1/seed_42/weights/best.pt']
        selected = core.checkpoint_paths(paths,'Varroa')
        self.assertEqual(len(selected),2)
        original_inventory = {'hf_revision':'a'*40,'checkpoints':[
            {'path':p,'kind':Path(p).name} for p in selected]}
        result = driver.best_inventory(original_inventory)
        self.assertEqual(result['checkpoint_count'],1)
        self.assertEqual(result['hf_revision'],'a'*40)
        self.assertEqual(len(original_inventory['checkpoints']),2)

    def test_missing_or_changed_pixels_block_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            a,b = Path(td)/'pt32_reference',Path(td)/'evaluation'
            for folder in (a,b):
                folder.mkdir()
                for name in ('main_inputs.json','small_inputs.json','test_size_ground_truth.json'):
                    (folder/name).write_text('{}')
            compare_inputs(a,b)
            (b/'small_inputs.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'pixels or shapes'):
                compare_inputs(a,b)

    def test_evidence_rejects_incomplete_and_tampered_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            evidence = {'main_full_test_input_parity':True,'small_full_test_input_parity':True,
                        'artifacts_sha256':{}}
            record = {'evaluation_evidence':evidence}
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                validate_evidence(folder,'Varroa',record)
            for out in ('pt32_reference','evaluation'):
                (folder/out).mkdir()
                for name in ('main_inputs.json','small_inputs.json','test_size_ground_truth.json',
                             'test_size_predictions.json','metrics.json'):
                    path = folder/out/name
                    path.write_text('{}')
                    evidence['artifacts_sha256'][f'{out}/{name}'] = core.sha256(path)
            validate_evidence(folder,'Varroa',record)
            (folder/'evaluation/test_size_predictions.json').write_text('[]')
            with self.assertRaisesRegex(RuntimeError,'tampered'):
                validate_evidence(folder,'Varroa',record)

    def test_upload_failure_stops_before_next_checkpoint(self):
        rows = [{'kind':'best.pt','job_id':str(i),'path':str(i)} for i in range(2)]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(driver,'Persistence') as store_cls, \
             mock.patch.object(core,'hashes',return_value={}), \
             mock.patch.object(core,'runtime_identity',return_value={}), \
             mock.patch.object(core,'evaluate_job',return_value={'status':'EVALUATED_INT8'}) as evaluate, \
             mock.patch.object(driver,'legacy_detect_support',side_effect=lambda *a:nullcontext()):
            store = store_cls.return_value
            store.inventory.return_value = {'checkpoints':rows}
            store.publish.side_effect = RuntimeError('HF unavailable')
            with self.assertRaisesRegex(RuntimeError,'HF unavailable'):
                driver.main(['--phase','run','--project',td,'--data-root',td,'--prepared-root',td])
            self.assertEqual(evaluate.call_count,1)
            store.reports.assert_not_called()

    def test_restore_passes_only_exact_job_namespace_to_existing_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            store = object.__new__(driver.Persistence)
            store.args = SimpleNamespace(project=Path(td), repo='owner/repo')
            store.api = mock.Mock()
            store.revision = 'a'*40
            row = {'job_id':'seed_42__best__job','checkpoint_seed':42}
            prefix = f'{driver.PREFIX}/{row["job_id"]}'
            store.files = {prefix+'/upload_complete.json'}
            marker = Path(td)/'marker.json'
            marker.write_text(json.dumps({'artifact_sha256':{'metrics_int8.json':'sha'}}))
            store.download = mock.Mock(return_value=marker)
            def restore(**kwargs):
                folder = Path(kwargs['project'])/driver.VARIANT/row['job_id']
                folder.mkdir(parents=True)
                (folder/'metrics_int8.json').write_text('{}')
                return {'restored':[{'run_dir':str(folder)}]}
            with mock.patch.object(driver, 'HFResultArchive') as archive:
                archive.return_value.restore_completed_runs.side_effect = restore
                # The hub import occurs only during real restore, not at module load.
                import sys, types
                hub = types.ModuleType('huggingface_hub')
                hub.snapshot_download = mock.Mock(return_value=td)
                with mock.patch.dict(sys.modules, {'huggingface_hub':hub}):
                    self.assertTrue(store.restore(row))
                    scoped = archive.call_args.kwargs['snapshot_download_fn']
                    scoped(repo_id='owner/repo',allow_patterns=['Varroa/**'])
                    self.assertEqual(hub.snapshot_download.call_args.kwargs['allow_patterns'],[prefix+'/**'])
                    self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'],store.revision)
                self.assertTrue((Path(td)/'Varroa/jobs'/row['job_id']/'metrics_int8.json').exists())
