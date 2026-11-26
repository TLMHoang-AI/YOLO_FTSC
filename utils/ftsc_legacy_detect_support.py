"""Process-scoped, state-preserving compatibility for historical disabled Detect branches.

Evaluator and model source files are never rewritten. Legacy heads with a
missing *enabled* learned branch are rejected, not reconstructed.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def patch_disabled_box_detail(model, detect_type):
    import torch
    config = getattr(model, 'yaml', {}) or {}
    before = {key: (tuple(value.shape), value.dtype, value.data_ptr())
              for key, value in model.state_dict().items()}
    heads = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, detect_type) or hasattr(module, 'box_detail'):
            continue
        if getattr(module, 'box_detail_head', False) or config.get('box_detail_head', False):
            raise RuntimeError('Checkpoint enables box_detail but its learned branch is missing; refusing reconstruction')
        module.box_detail = torch.nn.ModuleList([torch.nn.Identity() for _ in range(module.nl)])
        heads.append({'module': name, 'levels': module.nl, 'replacement': 'Identity',
                      'box_detail_head_enabled': False})
    after = {key: (tuple(value.shape), value.dtype, value.data_ptr())
             for key, value in model.state_dict().items()}
    if before != after:
        raise RuntimeError('Legacy compatibility changed model parameter/buffer state')
    return heads


@contextmanager
def legacy_detect_support(where, row):
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from ultralytics.nn import tasks
    from ultralytics.nn.modules.head import Detect
    from utils import ftsc_hf_all_trt_int8_paper_v1 as core
    original = tasks.torch_safe_load
    observed = {}

    def compatible_load(*args, **kwargs):
        checkpoint, filename = original(*args, **kwargs)
        seen = set()
        for key in ('model', 'ema'):
            model = checkpoint.get(key)
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            for head in patch_disabled_box_detail(model, Detect):
                observed[head['module']] = head
        return checkpoint, filename

    tasks.torch_safe_load = compatible_load
    try:
        yield
    finally:
        tasks.torch_safe_load = original
        if observed:
            where = Path(where)
            checkpoint = where/'source'/row['kind']
            helper = Path(__file__)
            payload = {
                'schema': 'legacy_disabled_box_detail_identity_v1',
                'source': row['path'], 'checkpoint_sha256': core.sha256(checkpoint),
                'helper_sha256': core.sha256(helper),
                'heads': [observed[key] for key in sorted(observed)],
                'state_dict_unchanged': True,
                'policy': 'Only absent, disabled box_detail branches receive parameter-free Identity; enabled missing branches fail closed.',
            }
            snapshot = where/'compatibility'/helper.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.exists() and snapshot.read_bytes() != helper.read_bytes():
                raise RuntimeError('Previous compatibility helper differs; inspect partial job before retry')
            snapshot.write_bytes(helper.read_bytes())
            core.atomic_json(where/'legacy_detect_compatibility.json', payload)
            print('LEGACY_DETECT_COMPATIBILITY', json.dumps(payload), flush=True)
