import tempfile
from pathlib import Path
from unittest import TestCase, skipIf
try:
    import torch
except ImportError:
    torch = None
from utils.ftsc_legacy_detect_support import patch_disabled_box_detail, legacy_detect_support


@skipIf(torch is None, 'PyTorch required for actual tensor/state compatibility checks')
class LegacyDetectTests(TestCase):
    def make_model(self, enabled=False, modern=False):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nl=2
                self.box_detail_head=enabled
                self.weight=torch.nn.Parameter(torch.tensor([2.]))
                if modern:self.box_detail=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
            def forward(self, x):return self.box_detail[0](x)*self.weight
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=Head();self.yaml={'box_detail_head':enabled}
        return Model(),Head

    def test_legacy_identity_preserves_tensors_and_forward_math(self):
        model,kind=self.make_model()
        before={key:value.clone() for key,value in model.state_dict().items()}
        changed=patch_disabled_box_detail(model,kind)
        self.assertEqual(len(changed),1)
        self.assertEqual(set(before),set(model.state_dict()))
        for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        value=torch.tensor([3.])
        self.assertTrue(torch.equal(model.head(value),torch.tensor([6.])))
        self.assertEqual(patch_disabled_box_detail(model,kind),[])

    def test_missing_enabled_learned_branch_is_rejected(self):
        model,kind=self.make_model(enabled=True)
        with self.assertRaisesRegex(RuntimeError,'learned branch is missing'):
            patch_disabled_box_detail(model,kind)
        self.assertFalse(hasattr(model.head,'box_detail'))

    def test_modern_existing_branch_is_not_replaced(self):
        model,kind=self.make_model(modern=True)
        original=model.head.box_detail
        self.assertEqual(patch_disabled_box_detail(model,kind),[])
        self.assertIs(model.head.box_detail,original)

    def test_process_loader_hook_is_restored_after_error(self):
        from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
        ensure_local_ultralytics()
        from ultralytics.nn import tasks
        original=tasks.torch_safe_load
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,'test interrupt'):
                with legacy_detect_support(Path(td),{'kind':'best.pt','path':'unused'}):
                    self.assertIsNot(tasks.torch_safe_load,original)
                    raise RuntimeError('test interrupt')
        self.assertIs(tasks.torch_safe_load,original)
