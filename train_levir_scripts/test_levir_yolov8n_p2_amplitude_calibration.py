#!/usr/bin/env python3
"""Unit tests for Amplitude Calibration and Perturbation modules."""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.conv import P2AmplitudeCalibrator, LearnableGlobalScalar, AmplitudePerturbation

def test_identity_initialization():
    print("Testing identity initialization...")
    # Dynamic calibrator
    calibrator = P2AmplitudeCalibrator(channels=32, hidden_dim=16)
    x = torch.randn(2, 32, 64, 64)
    out = calibrator(x)
    assert torch.allclose(out, x, atol=1e-5), "Calibrator is not identity initialized"
    
    # Global scalar
    global_scalar = LearnableGlobalScalar(channels=32)
    out_global = global_scalar(x)
    assert torch.allclose(out_global, x, atol=1e-5), "LearnableGlobalScalar is not identity initialized"
    print("Identity initialization tests passed!")

def test_perturbation_behavior():
    print("Testing perturbation training vs evaluation mode...")
    perturb = AmplitudePerturbation(mode="image", scale_range=(0.7, 1.3))
    x = torch.ones(2, 32, 64, 64)
    
    # Eval mode: should be identity
    perturb.eval()
    out_eval = perturb(x)
    assert torch.equal(out_eval, x), "Perturbation should be identity in eval mode"
    
    # Train mode: should be random scaling
    perturb.train()
    out_train = perturb(x)
    assert not torch.equal(out_train, x), "Perturbation did not perturb in train mode"
    print("Perturbation behavior tests passed!")

def test_model_topology():
    print("Testing model topologies from YAML configs...")
    configs = {
        "global_scalar": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_global_scalar.yaml",
        "amplitude_calibrator": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_calibrator.yaml",
        "amplitude_perturbation": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_perturbation.yaml",
        "calibrator_perturbation": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_calibrator_perturbation.yaml"
    }
    
    for name, path in configs.items():
        print(f"Loading {name} model...")
        model = YOLO(path)
        
        # Verify the custom modules are in the right position
        if name == "global_scalar":
            assert isinstance(model.model.model[19], LearnableGlobalScalar)
            assert model.model.model[20].f == [19]
        elif name == "amplitude_calibrator":
            assert isinstance(model.model.model[19], P2AmplitudeCalibrator)
            assert model.model.model[20].f == [19]
        elif name == "amplitude_perturbation":
            assert isinstance(model.model.model[19], AmplitudePerturbation)
            assert model.model.model[20].f == [19]
        elif name == "calibrator_perturbation":
            assert isinstance(model.model.model[19], P2AmplitudeCalibrator)
            assert isinstance(model.model.model[20], AmplitudePerturbation)
            assert model.model.model[21].f == [20]
            
        print(f"{name} model topology is correct!")

if __name__ == "__main__":
    test_identity_initialization()
    test_perturbation_behavior()
    test_model_topology()
    print("All tests passed successfully!")
