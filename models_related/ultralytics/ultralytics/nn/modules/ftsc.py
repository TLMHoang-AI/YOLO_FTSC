# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FTSC modules for tiny-object FPN supervision experiments.

The modules are independent from the older DALA/DBSS experiments. The
assignment-preserving calibrator is owned by Detect but runs only in the loss
after TAL, while the legacy detail calibrator remains a plug-in feature layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "AnchorFreeFTSCCalibrator",
    "DFLDistributionEvidence",
    "FTSCFeatureCalibrator",
    "PositionGaussianEvidence",
)


class PositionGaussianEvidence(nn.Module):
    """Return stable log-evidence from an anchor point's position inside its assigned GT box."""

    def __init__(self, alpha: float = 6.0, eps: float = 1e-9) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("FTSC position alpha must be positive.")
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        anchor_points_px: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log Position-Gaussian evidence for TAL positives only."""
        points = anchor_points_px.unsqueeze(0).expand(target_bboxes_px.shape[0], -1, -1)[fg_mask]
        boxes = target_bboxes_px[fg_mask]
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(self.eps)
        normalized_offset = (points - centers) / (sizes / self.alpha).clamp_min(self.eps)
        return -0.5 * normalized_offset.square().sum(-1)


class DFLDistributionEvidence(nn.Module):
    """Return detached log-reliability from entropy/variance of positive DFL distributions."""

    def __init__(
        self,
        reg_max: int,
        entropy_tau: float = 1.0,
        variance_tau: float = 0.0,
        detach: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if reg_max <= 1:
            raise ValueError("DFL evidence requires reg_max > 1.")
        if entropy_tau < 0 or variance_tau < 0:
            raise ValueError("DFL entropy/variance strengths must be non-negative.")
        self.reg_max = int(reg_max)
        self.entropy_tau = float(entropy_tau)
        self.variance_tau = float(variance_tau)
        self.detach = bool(detach)
        self.eps = float(eps)
        self.register_buffer("bins", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)
        self.last_entropy: torch.Tensor | None = None
        self.last_variance: torch.Tensor | None = None

    def forward(self, pred_distri: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """Compute one log-reliability value per TAL positive."""
        logits = pred_distri[fg_mask].reshape(-1, 4, self.reg_max).float()
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(-1) / math.log(self.reg_max)
        bins = self.bins.to(device=probabilities.device)
        expected = (probabilities * bins).sum(-1, keepdim=True)
        variance = (probabilities * (bins - expected).square()).sum(-1)
        variance = variance / max(float((self.reg_max - 1) ** 2), self.eps)
        entropy = entropy.mean(-1)
        variance = variance.mean(-1)
        log_evidence = -self.entropy_tau * entropy - self.variance_tau * variance
        if self.detach:
            log_evidence = log_evidence.detach()
        self.last_entropy = entropy.detach()
        self.last_variance = variance.detach()
        return log_evidence.to(dtype=pred_distri.dtype)


class AnchorFreeFTSCCalibrator(nn.Module):
    """Assignment-preserving supervision calibration for anchor-free YOLO detection.

    The module is owned by the Detect head so F5 evidence-strength parameters exist
    before the trainer builds its optimizer. It is called only by the criterion
    after TAL assignment and therefore adds no inference-time work.
    """

    SUPPORTED_EVIDENCE = {"position_gaussian", "dfl_distribution"}

    def __init__(self, config: dict, reg_max: int) -> None:
        super().__init__()
        config = dict(config or {})
        self.policy = str(config.get("policy", "e4")).lower()
        if self.policy not in {"e4", "f5"}:
            raise ValueError("FTSC policy must be 'e4' or 'f5'.")

        evidence = config.get("evidence", ["position_gaussian"])
        self.evidence_names = tuple(str(name).lower() for name in evidence)
        unknown = set(self.evidence_names) - self.SUPPORTED_EVIDENCE
        if unknown:
            raise ValueError(f"Unsupported FTSC evidence: {sorted(unknown)}")
        if not self.evidence_names:
            raise ValueError("FTSC requires at least one evidence provider.")

        self.log_clip = float(config.get("log_clip", 0.35))
        if self.log_clip <= 0:
            raise ValueError("FTSC log_clip must be positive.")
        self.per_gt_norm = bool(config.get("per_gt_norm", True))
        self.warmup_epochs = max(int(config.get("warmup_epochs", 5)), 0)
        self.ramp_epochs = max(int(config.get("ramp_epochs", 10)), 1)
        self.apply_cls = bool(config.get("apply_cls", True))
        self.apply_box = bool(config.get("apply_box", True))
        self.apply_dfl = bool(config.get("apply_dfl", True))
        self.dfl_apply_cls = bool(config.get("dfl_apply_cls", True))
        self.dfl_apply_box = bool(config.get("dfl_apply_box", False))
        self.dfl_apply_dfl = bool(config.get("dfl_apply_dfl", False))
        self.strength_max = float(config.get("strength_max", 2.0))
        self.strength_init = float(config.get("strength_init", 1.0))
        self.strength_reg_weight = float(config.get("strength_reg_weight", 1e-4))
        if not 0 < self.strength_init < self.strength_max:
            raise ValueError("FTSC strength_init must lie strictly between zero and strength_max.")
        if self.strength_reg_weight < 0:
            raise ValueError("FTSC strength_reg_weight must be non-negative.")

        providers = {}
        if "position_gaussian" in self.evidence_names:
            providers["position_gaussian"] = PositionGaussianEvidence(float(config.get("position_alpha", 6.0)))
        if "dfl_distribution" in self.evidence_names:
            providers["dfl_distribution"] = DFLDistributionEvidence(
                reg_max=reg_max,
                entropy_tau=float(config.get("dfl_entropy_tau", 1.0)),
                variance_tau=float(config.get("dfl_variance_tau", 0.0)),
                detach=bool(config.get("dfl_detach", True)),
            )
        self.providers = nn.ModuleDict(providers)

        self.strength_logits = nn.ParameterDict()
        if self.policy == "f5":
            probability = self.strength_init / self.strength_max
            initial_logit = math.log(probability / (1.0 - probability))
            for name in self.evidence_names:
                self.strength_logits[name] = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.last_metrics: dict[str, float] = {}

    def strength(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        """Return fixed E4 strength or bounded learnable F5 strength."""
        if self.policy == "e4":
            return reference.new_tensor(self.strength_init)
        return torch.sigmoid(self.strength_logits[name]).to(reference) * self.strength_max

    def residual_fraction(self, epoch: int) -> float:
        """Return the F5 identity-to-full-gate schedule for the current zero-based epoch."""
        if self.policy == "e4":
            return 1.0
        if epoch < self.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)

    @staticmethod
    def _center_per_gt(
        values: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Center log-evidence independently inside each image/GT positive group."""
        if not values.numel():
            return values
        batch_size, num_anchors = fg_mask.shape
        batch_ids = torch.arange(batch_size, device=fg_mask.device).view(-1, 1).expand_as(fg_mask)[fg_mask]
        group_ids = batch_ids * num_anchors + target_gt_idx[fg_mask].long()
        group_count = batch_size * num_anchors
        sums = values.new_zeros(group_count).scatter_add_(0, group_ids, values)
        counts = values.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(values))
        return values - sums[group_ids] / counts[group_ids].clamp_min(1.0)

    @staticmethod
    def _normalize_per_gt(
        values: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the arithmetic mean weight of every image/GT group to one."""
        if not values.numel():
            return values
        batch_size, num_anchors = fg_mask.shape
        batch_ids = torch.arange(batch_size, device=fg_mask.device).view(-1, 1).expand_as(fg_mask)[fg_mask]
        group_ids = batch_ids * num_anchors + target_gt_idx[fg_mask].long()
        group_count = batch_size * num_anchors
        sums = values.new_zeros(group_count).scatter_add_(0, group_ids, values)
        counts = values.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(values))
        means = sums / counts.clamp_min(1.0)
        return values / means[group_ids].clamp_min(1e-9)

    def _gate(
        self,
        log_gate: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        clipped = log_gate.clamp(-self.log_clip, self.log_clip)
        raw_gate = clipped.exp()
        if self.per_gt_norm:
            # Log-centering controls evidence scale before clipping; this second
            # normalization prevents clipping from changing total GT loss mass.
            raw_gate = self._normalize_per_gt(raw_gate, fg_mask, target_gt_idx)
        rho = self.residual_fraction(epoch)
        return 1.0 + rho * (raw_gate - 1.0)

    @staticmethod
    def _stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
        if not values.numel():
            return {
                f"{prefix}_mean": 1.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 1.0,
                f"{prefix}_max": 1.0,
            }
        values = values.detach().float()
        return {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_min": float(values.min().item()),
            f"{prefix}_max": float(values.max().item()),
        }

    def forward(
        self,
        anchor_points_px: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        pred_distri: torch.Tensor,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Build positive-only classification, box and DFL weights after TAL assignment."""
        positive_count = int(fg_mask.sum().item())
        if positive_count == 0:
            empty = pred_distri.new_empty(0)
            zero = pred_distri.sum() * 0.0
            self.last_metrics = {"ftsc_positive_count": 0.0, "ftsc_rho": self.residual_fraction(epoch)}
            return {"cls": empty, "box": empty, "dfl": empty, "regularization": zero}

        centered = {}
        if "position_gaussian" in self.providers:
            position = self.providers["position_gaussian"](anchor_points_px, target_bboxes_px, fg_mask)
            centered["position_gaussian"] = (
                self._center_per_gt(position, fg_mask, target_gt_idx) if self.per_gt_norm else position
            )
        if "dfl_distribution" in self.providers:
            distribution = self.providers["dfl_distribution"](pred_distri, fg_mask)
            centered["dfl_distribution"] = (
                self._center_per_gt(distribution, fg_mask, target_gt_idx) if self.per_gt_norm else distribution
            )

        reference = next(iter(centered.values()))
        zero = torch.zeros_like(reference)
        task_logs = {"cls": zero.clone(), "box": zero.clone(), "dfl": zero.clone()}
        for name, values in centered.items():
            contribution = self.strength(name, values) * values
            if name == "position_gaussian":
                if self.apply_cls:
                    task_logs["cls"] = task_logs["cls"] + contribution
                if self.apply_box:
                    task_logs["box"] = task_logs["box"] + contribution
                if self.apply_dfl:
                    task_logs["dfl"] = task_logs["dfl"] + contribution
            else:
                if self.dfl_apply_cls:
                    task_logs["cls"] = task_logs["cls"] + contribution
                if self.dfl_apply_box:
                    task_logs["box"] = task_logs["box"] + contribution
                if self.dfl_apply_dfl:
                    task_logs["dfl"] = task_logs["dfl"] + contribution

        weights = {}
        for task, values in task_logs.items():
            weights[task] = self._gate(values, fg_mask, target_gt_idx, epoch)

        regularization = reference.sum() * 0.0
        if self.policy == "f5" and self.strength_reg_weight:
            regularization = self.strength_reg_weight * sum(
                (self.strength(name, reference) - self.strength_init).square() for name in self.evidence_names
            )

        metrics = {
            "ftsc_positive_count": float(positive_count),
            "ftsc_rho": self.residual_fraction(epoch),
            "ftsc_negative_weight": 1.0,
        }
        for name, values in centered.items():
            metrics.update(self._stats(f"ftsc_{name}", values))
            metrics[f"ftsc_strength_{name}"] = float(self.strength(name, reference).detach().item())
        for task, values in weights.items():
            metrics.update(self._stats(f"ftsc_weight_{task}", values))
            metrics[f"ftsc_{task}_clamp_low_fraction"] = float(
                (task_logs[task] < -self.log_clip).float().mean().item()
            )
            metrics[f"ftsc_{task}_clamp_high_fraction"] = float(
                (task_logs[task] > self.log_clip).float().mean().item()
            )
        metrics["ftsc_positive_negative_weight_ratio"] = float(weights["cls"].detach().float().mean().item())
        distribution_provider = self.providers["dfl_distribution"] if "dfl_distribution" in self.providers else None
        if distribution_provider is not None:
            metrics.update(self._stats("ftsc_dfl_entropy", distribution_provider.last_entropy))
            metrics.update(self._stats("ftsc_dfl_variance", distribution_provider.last_variance))
        self.last_metrics = metrics
        return {**weights, "regularization": regularization}


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
