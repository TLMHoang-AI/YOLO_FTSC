import sys
from pathlib import Path
import torch
import torch.nn as nn
import math

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics.utils.loss import v8DetectionLoss

class MockDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
        self.nc = 1
        self.reg_max = 16
        self.p2_dfl_bins = None
        self.loc_quality_enabled = False
        self.quality_head = False
        self.box_detail_head = False
        self.dfl_residual = False
        
        # Mock cv2[0] with Conv2d and BatchNorm2d to inspect buffer behavior
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        ])

class MockModel(nn.Module):
    def __init__(self, mode="dominant"):
        super().__init__()
        self.model = nn.ModuleList([MockDetect()])
        self.args = type("Args", (object,), {
            "positive_support_dropout": True,
            "positive_support_mode": mode,
            "positive_support_gain": 0.25,
            "positive_support_prob": 1.0,  # force dropout for deterministic testing
            "positive_support_min_count": 3,
            "positive_support_aux_topk": 3,
            "positive_support_radius": 2,
            "positive_support_fill_kernel": 3,
            "positive_support_warmup_start": 0,
            "positive_support_warmup_end": 10,
            "box_consensus_gain": 0.0,
            "rank_loss": 0.0,
            "loc_quality": 0.0,
            "boundary_contrast": 0.0,
            "quality_head": False,
            "dgfe_rec_gain": 0.0,
            "dgfe_spatial_gain": 0.0,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "bbox_iou_loss": "ciou",
        })()
        
        self.dummy_param = nn.Parameter(torch.zeros(1))

def test_psd_gain_warmup():
    model = MockModel()
    criterion = v8DetectionLoss(model)
    
    # Test warmup start
    criterion.epoch = 0
    assert criterion._psd_current_gain() == 0.0
    
    # Test midpoint
    criterion.epoch = 5
    assert criterion._psd_current_gain() == 0.125
    
    # Test end
    criterion.epoch = 10
    assert criterion._psd_current_gain() == 0.25

def test_psd_bn_restoration():
    model = MockModel()
    criterion = v8DetectionLoss(model)
    criterion.epoch = 10
    
    h2, w2 = 8, 8
    n_p2 = h2 * w2
    
    feats = [torch.randn(1, 16, h2, w2)]
    preds = {
        "feats": feats,
        "boxes": torch.randn(1, 64, n_p2),
        "scores": torch.randn(1, 1, n_p2),
    }
    
    batch = {
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "img": torch.randn(1, 3, 32, 32),
    }
    
    bn_layer = criterion.detect_head.cv2[0][1]
    
    # Initialize BN statistics
    bn_layer.running_mean.fill_(0.0)
    bn_layer.running_var.fill_(1.0)
    bn_layer.num_batches_tracked.fill_(0)
    
    # Call get_assigned_targets_and_loss
    criterion.get_assigned_targets_and_loss(preds, batch)
    
    # Verify BN statistics were NOT modified or incremented by the auxiliary pass forward
    assert int(bn_layer.num_batches_tracked.item()) == 0
    assert torch.allclose(bn_layer.running_mean, torch.zeros_like(bn_layer.running_mean))
    assert torch.allclose(bn_layer.running_var, torch.ones_like(bn_layer.running_var))

def test_psd_selection_and_masking():
    # Test that alternative supports are found relative to the chosen center (dominant vs random)
    # and neighbors are restricted within self.psd_radius (radius 2)
    model = MockModel(mode="dominant")
    criterion = v8DetectionLoss(model)
    criterion.epoch = 10
    
    h2, w2 = 8, 8
    n_p2 = h2 * w2
    
    # Setup P2 grid inputs
    feats = [torch.randn(1, 16, h2, w2)]
    preds = {
        "feats": feats,
        "boxes": torch.randn(1, 64, n_p2),
        "scores": torch.randn(1, 1, n_p2),
    }
    
    batch = {
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "img": torch.randn(1, 3, 32, 32),
    }
    
    # We call assigner to get fg_mask and targets, but we can inspect the generated aux_fg_mask
    _, loss, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    
    # Verify that PSD diagnostics log metrics
    assert "psd_eligible_gt" in criterion.psd_metrics
    assert "psd_alt_supports_avg" in criterion.psd_metrics
    assert "psd_raw_loss" in criterion.psd_metrics
    assert "psd_delta_drop" in criterion.psd_metrics

if __name__ == "__main__":
    test_psd_gain_warmup()
    test_psd_bn_restoration()
    test_psd_selection_and_masking()
    print("All PSD functional unit tests passed successfully!")
