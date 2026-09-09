"""Lightweight orientation-selective edge cue fusion.

This is a BDNet-inspired project adaptation, not the full BDNet OrSM. Fixed
oriented filters preserve directional information; a learned softmax gate selects
orientations before a zero-initialized P2 residual projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("P2EdgeCueFusion", "oriented_edge_responses")


def _oriented_kernels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Derivative-like 3x3 filters for 0, 45, 90, 135 degrees.
    kernels = torch.tensor(
        [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        ],
        dtype=dtype,
    )
    return kernels / kernels.flatten(1).norm(dim=1).view(4, 1, 1).clamp_min(1e-8)


def oriented_edge_responses(image: torch.Tensor) -> torch.Tensor:
    """Return four finite oriented luminance responses with shape ``B,4,H,W``."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB input, got {tuple(image.shape)}")
    rgb = image.float().clamp(0.0, 1.0)
    gray = (0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    kernels = _oriented_kernels(gray.dtype).to(gray.device).unsqueeze(1)
    return F.conv2d(gray, kernels, padding=1).nan_to_num(0.0, 0.0, 0.0)


class P2EdgeCueFusion(nn.Module):
    """Orientation-selective edge branch with identity-friendly P2 fusion."""

    needs_image = True
    orientation_count = 4

    def __init__(self, p2_channels: int, hidden: int = 32):
        super().__init__()
        hidden = max(8, int(hidden))
        self.encoder = nn.Sequential(
            Conv(self.orientation_count, hidden, 3, 2),
            Conv(hidden, hidden, 3, 2),
            Conv(hidden, hidden, 3, 1),
        )
        self.gate = nn.Linear(self.orientation_count, self.orientation_count)
        self.projection = nn.Conv2d(hidden, p2_channels, 1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.register_buffer("orientation_kernels", _oriented_kernels(), persistent=False)
        self.last_stats: dict[str, torch.Tensor] = {}

    def estimated_gflops(self, image_size: int = 512) -> float:
        """Static multiply-add estimate for the cue branch at a square input."""
        h = image_size
        total = 2 * 1 * self.orientation_count * 3 * 3 * image_size * image_size
        for cin, cout, k, stride in ((self.orientation_count, self.encoder[0].conv.out_channels, 3, 2), (self.encoder[0].conv.out_channels, self.encoder[1].conv.out_channels, 3, 2), (self.encoder[1].conv.out_channels, self.encoder[2].conv.out_channels, 3, 1)):
            h = (h + 2 * (k // 2) - k) // stride + 1
            total += 2 * cin * cout * k * k * h * h
        total += 2 * self.projection.in_channels * self.projection.out_channels * h * h
        total += 2 * self.gate.in_features * self.gate.out_features
        return total / 1e9

    def forward(self, p2: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        responses = oriented_edge_responses(image).to(device=p2.device, dtype=p2.dtype)
        # Gate from pooled directional responses; softmax guarantees normalized weights.
        pooled = responses.abs().mean(dim=(-2, -1))
        weights = torch.softmax(self.gate(pooled), dim=-1)
        selected = responses * weights.unsqueeze(-1).unsqueeze(-1)
        feature = self.encoder(selected)
        if feature.shape[-2:] != p2.shape[-2:]:
            feature = F.interpolate(feature, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.projection(feature)
        rgb_norm = p2.detach().flatten(1).norm(dim=1).mean().clamp_min(1e-8)
        edge_norm = residual.detach().flatten(1).norm(dim=1).mean()
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_stats = {
            "orientation_gate_weights": weights.detach().mean(dim=0),
            "orientation_entropy": entropy.detach(),
            "edge_feature_norm": edge_norm.detach(),
            "rgb_p2_norm": rgb_norm,
            "edge_rgb_norm_ratio": edge_norm / rgb_norm,
            "edge_projection_norm": self.projection.weight.detach().norm(),
            "edge_activation_mean": feature.detach().mean(),
            "edge_activation_std": feature.detach().std(unbiased=False),
        }
        return p2 + residual
