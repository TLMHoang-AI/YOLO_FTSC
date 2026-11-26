"""Varroa INT8 protocol built on an isolated instance of the frozen V1 machinery.

The LEVIR/Tiny module and its source hashes are never mutated. Only this private
instance uses Varroa's seed-specific preparation, native metrics and evidence.
"""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path

_CORE_FILE = Path(__file__).with_name('ftsc_hf_all_trt_int8_paper_v1.py')
_spec = importlib.util.spec_from_file_location('utils._varroa_int8_core', _CORE_FILE)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
core.CONTRACT = 'FTSC_VARROA_TRT_INT8_NATIVE640_PARITY_V1'
core.DATASETS = ('Varroa',)
core.METRICS = {'Varroa': ('test/metrics/mAP50(B)', 'test/metrics/mAP50-95(B)',
                           'test_size/AP50-Small')}
core.SOURCE_FILES = tuple(dict.fromkeys((*core.SOURCE_FILES,
    'utils/ftsc_varroa_int8_evaluator.py',
    'utils/ftsc_varroa_int8_archive_resume.py',
    'utils/ftsc_legacy_detect_support.py',
    'train_levir_scripts/train_varroa_dbss_hit.py', 'misc/prepare_dataset.py',
    'models_related/ultralytics/ultralytics/nn/modules/head.py',
    'models_related/ultralytics/ultralytics/nn/tasks.py',
)))


def validate_data(args, row):
    from train_levir_scripts.train_varroa_dbss_hit import dataset_for_seed, validate_dataset
    from evaluate_test.size_bucket_evaluator import _resolve_test_images, _label_path
    import yaml
    seed = row['checkpoint_seed']
    core.need(seed in (42,43,44) and row['seed_resolution'] == 'checkpoint_path',
              'Varroa requires a verified checkpoint seed 42/43/44; no fallback split')
    config = Namespace(data_root=args.data_root.resolve(), dataset_root=args.prepared_root.resolve())
    data_yaml = dataset_for_seed(config, seed).resolve()
    validate_dataset(data_yaml)
    payload = yaml.safe_load(data_yaml.read_text())
    root = Path(payload.get('path', data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent/root).resolve()
    sets = {}
    for split, count in (('train',2762),('val',592),('test',592)):
        folder = (root/payload[split]).resolve()
        images = sorted(p.resolve() for p in folder.iterdir()
                        if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        core.need(len(images) == count, f'Invalid canonical {split} count')
        sets[split] = set(images)
    core.need(all(not sets[a] & sets[b] for a,b in (('train','val'),('train','test'),('val','test'))),
              'Train calibration and evaluation splits overlap')
    images = _resolve_test_images(data_yaml)
    digest = __import__('hashlib').sha256()
    for image in sorted(images):
        label = _label_path(image)
        digest.update((str(image.resolve())+'\0'+core.sha256(image)+'\0'+core.sha256(label)+'\n').encode())
    context = {'train_calibration':core.calibration_fingerprint(data_yaml),
               'calibration_seed':seed, 'calibration_yaml_sha256':core.sha256(data_yaml),
               'test_image_label_sha256':digest.hexdigest(), 'test_count':len(images),
               'protocol':'native_val640_rect_batch4_iou07_and_predict640_conf025_size'}
    return data_yaml, (data_yaml,images), context, None


def compare_inputs(reference, measured):
    for kind in ('main','small'):
        left, right = reference/f'{kind}_inputs.json', measured/f'{kind}_inputs.json'
        core.need(left.read_bytes() == right.read_bytes(),
                  f'Varroa INT8/{kind} input pixels or shapes differ from PT32')
    core.need((reference/'test_size_ground_truth.json').read_bytes() ==
              (measured/'test_size_ground_truth.json').read_bytes(), 'Varroa PT32/INT8 GT differs')


def evaluate_native(engine, unused_root, inputs, where, device, workers):
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from utils import ftsc_deployment_paper_backend_v1 as paper
    data_yaml, images = inputs
    args = Namespace(dataset='Varroa',device=device,workers=workers)
    reference, measured = where/'pt32_reference', where/'evaluation'
    reference.mkdir(parents=True,exist_ok=True)
    measured.mkdir(parents=True,exist_ok=True)
    source = engine.with_suffix('.pt')
    scores = {}
    for backend, model, out in (('pt32',source,reference),('trt8',engine,measured)):
        main, _ = paper._val_varroa(model,backend,args,out,data_yaml,images)
        small, _ = paper._predict_native_small(model,backend,images,args,out)
        scores[backend] = {**main,**small}
    compare_inputs(reference, measured)
    core.atomic_json(reference/'metrics.json', scores['pt32'])
    evidence = {'main_full_test_input_parity':True,'small_full_test_input_parity':True,
                'artifacts_sha256':{p.relative_to(where).as_posix():core.sha256(p)
                    for folder in (reference,measured) for p in folder.rglob('*.json')}}
    return scores['trt8'], evidence


def validate_evidence(where, dataset, record):
    core.need(dataset == 'Varroa', 'Wrong evaluator dataset')
    evidence = record.get('evaluation_evidence',{})
    core.need(evidence.get('main_full_test_input_parity') is True and
              evidence.get('small_full_test_input_parity') is True, 'Missing full test PT32 parity')
    hashes = evidence.get('artifacts_sha256',{})
    required = {f'{folder}/{name}' for folder in ('pt32_reference','evaluation')
                for name in ('main_inputs.json','small_inputs.json','test_size_ground_truth.json',
                             'test_size_predictions.json')} | {'pt32_reference/metrics.json'}
    core.need(required <= hashes.keys(), 'Incomplete native Varroa evaluation evidence')
    for relative,digest in hashes.items():
        path = Path(relative)
        core.need(not path.is_absolute() and '..' not in path.parts, 'Invalid evidence path')
        core.need((where/path).is_file() and core.sha256(where/path) == digest,
                  f'Missing/tampered Varroa evidence: {relative}')
    compare_inputs(where/'pt32_reference',where/'evaluation')


core.validate_data = validate_data
# Frozen V1 dispatches non-LEVIR datasets through this hook. In this isolated
# instance it executes native Varroa evaluation, never TinyPerson corner merging.
core.tinyperson_evaluate = evaluate_native
core.validate_evaluation_evidence = validate_evidence
