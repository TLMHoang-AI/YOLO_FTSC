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
        
        # Mock cv2[0] as a simple Conv/Linear layers with BatchNorm to test eval state switching
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        ])

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([MockDetect()])
        self.args = type("Args", (object,), {
            "positive_support_dropout": True,
            "positive_support_mode": "dominant",
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
        })()
        
        # Required parameter for device lookup
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
    
    criterion.epoch = 15
    assert criterion._psd_current_gain() == 0.25

def test_psd_loss_computation():
    model = MockModel()
    criterion = v8DetectionLoss(model)
    criterion.epoch = 10 # fully warmed up
    
    # Batch size 1, 1 class, grid H2=8, W2=8 (64 anchors for P2)
    h2, w2 = 8, 8
    n_p2 = h2 * w2
    
    # Mock preds dict
    # features shape: (1, 16, H2, W2)
    feats = [torch.randn(1, 16, h2, w2)]
    # boxes shape: (1, reg_max * 4, n_p2)
    # scores shape: (1, nc, n_p2)
    preds = {
        "feats": feats,
        "boxes": torch.randn(1, 64, n_p2),
        "scores": torch.randn(1, 1, n_p2),
    }
    
    # Mock batch dict
    # batch_idx, cls, bboxes (xywh normalized)
    batch = {
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "img": torch.randn(1, 3, 32, 32),
    }
    
    # We call get_assigned_targets_and_loss
    # Let's ensure no crash and psd loss is computed
    try:
        targets_and_loss = criterion.get_assigned_targets_and_loss(preds, batch)
        assert len(targets_and_loss) == 3
        loss_tensor = targets_and_loss[1]
        assert loss_tensor.shape[0] == 4 # box, cls, dfl, psd
        assert criterion.psd_metrics["psd_eligible_gt"] >= 0
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise e

if __name__ == "__main__":
    test_psd_gain_warmup()
    test_psd_loss_computation()
    print("All PSD unit tests passed successfully!")
