# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FTSC modules for tiny-object FPN supervision experiments.

The modules in this file are intentionally independent from the older DALA/DBSS
experiments.  They are designed as plug-and-play feature calibrators that can be
inserted before a YOLO Detect head without changing anchor-free assignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("FTSCFeatureCalibrator",)


class FTSCFeatureCalibrator(nn.Module):
    """Identity-initialized P2 detail calibrator for tiny object detection.

    The module estimates local high-frequency evidence from a feature map and
    uses it to build a gentle spatial-channel gate:

        detail = |x - AvgPool(x)|
        z      = detail / mean(detail)
        out    = x * (1 + alpha * (gate - 1))

    `alpha` is initialized near zero, so the module starts close to identity and
    lets training decide whether the tiny-detail cue is useful.
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.25,
        detail_kernel: int = 3,
        alpha_init: float = 1e-3,
        alpha_max: float = 0.35,
        spatial_temperature: float = 1.0,
        use_channel_gate: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.detail_kernel = max(int(detail_kernel) | 1, 3)
        self.alpha_max = max(float(alpha_max), 0.0)
        self.spatial_temperature = float(spatial_temperature)
        self.use_channel_gate = bool(use_channel_gate)

        hidden = max(int(self.channels * float(hidden_ratio)), 8)
        self.channel_gate = (
            nn.Sequential(
                nn.Conv2d(self.channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, self.channels, kernel_size=1),
            )
            if self.use_channel_gate
            else None
        )

        p = max(min(float(alpha_init) / max(self.alpha_max, 1e-12), 1.0 - 1e-6), 1e-6)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(p)))
        self.spatial_bias = nn.Parameter(torch.zeros(1))
        self.last_gate: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        """Return the bounded residual gate strength."""
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply detail-aware feature calibration."""
        detail = (x - F.avg_pool2d(x, self.detail_kernel, stride=1, padding=self.detail_kernel // 2)).abs()
        energy = detail.mean(dim=1, keepdim=True)
        normalizer = energy.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        spatial_gate = torch.sigmoid(self.spatial_temperature * (energy / normalizer - 1.0) - self.spatial_bias)

        if self.channel_gate is not None:
            channel_gate = torch.sigmoid(
                self.channel_gate(F.adaptive_avg_pool2d(detail, 1))
                + self.channel_gate(F.adaptive_max_pool2d(detail, 1))
            )
            gate = 1.0 + spatial_gate * channel_gate
        else:
            gate = 1.0 + spatial_gate

        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        self.last_gate = gate.detach() if self.training else None
        return x * (1.0 + alpha * (gate - 1.0))
