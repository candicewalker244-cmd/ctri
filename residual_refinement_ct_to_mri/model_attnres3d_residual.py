#!/usr/bin/env python3
"""Residual-refinement CT->MRI model.

This model keeps the E0 3D AttnRes U-Net backbone and adds a small
zero-initialized image-domain residual head:

  coarse_pred = E0_backbone(ct)
  residual = residual_scale * tanh(Conv3D(GELU(Conv3D(coarse_pred))))
  final_pred = clamp(coarse_pred + residual, 0, 1)

The residual branch is intentionally lightweight and can be disabled with
--no_refiner for pure E0 ablations.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model_attnres3d_gtmedreclpp as e0

set_seed = e0.set_seed
ensure_dir = e0.ensure_dir
discover_cases = e0.discover_cases
validate_case_splits = e0.validate_case_splits
pad_tensor_to_divisible = e0.pad_tensor_to_divisible
unpad_tensor = e0.unpad_tensor
crop_pair_around_foreground = e0.crop_pair_around_foreground
parse_eval_crop_size = e0.parse_eval_crop_size
CTMRIDataset = e0.CTMRIDataset
ReconstructionLoss = e0.ReconstructionLoss
train_one_epoch = e0.train_one_epoch
validate_one_epoch = e0.validate_one_epoch
validate_with_uncertainty = e0.validate_with_uncertainty
fit_moment_variance_scale = e0.fit_moment_variance_scale
save_checkpoint = e0.save_checkpoint
write_log_row = e0.write_log_row


class ResidualRefinementHead3D(nn.Module):
    """Small 3D residual head with zero-initialized output branch."""

    def __init__(self, hidden_channels: int = 8, residual_scale: float = 0.10) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")

        self.residual_scale = float(residual_scale)
        self.conv1 = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)

        # Start from zero residual so training begins as a pure E0 model.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.conv1(coarse))
        residual = self.residual_scale * torch.tanh(self.conv2(hidden))
        return torch.clamp(coarse + residual, 0.0, 1.0)

    def forward_mu_var(
        self,
        coarse_mu: torch.Tensor,
        coarse_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """First-order diagonal moment propagation through the residual head."""
        hidden_mu = self.conv1(coarse_mu)
        hidden_var = F.conv3d(
            coarse_var,
            self.conv1.weight.square(),
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )
        gelu_mu, gelu_derivative = e0.gelu_with_derivative(hidden_mu)
        hidden_var = gelu_derivative.square() * hidden_var

        residual_pre_mu = self.conv2(gelu_mu)
        residual_pre_var = F.conv3d(
            hidden_var,
            self.conv2.weight.square(),
            bias=None,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
        )
        tanh_mu = torch.tanh(residual_pre_mu)
        tanh_derivative = 1.0 - tanh_mu.square()
        residual_mu = self.residual_scale * tanh_mu
        residual_var = (
            self.residual_scale**2 * tanh_derivative.square() * residual_pre_var
        )

        pre_clamp_mu = coarse_mu + residual_mu
        refined_mu = torch.clamp(pre_clamp_mu, 0.0, 1.0)
        refined_var = coarse_var + residual_var
        inside = (pre_clamp_mu > 0.0) & (pre_clamp_mu < 1.0)
        refined_var = torch.where(inside, refined_var, torch.zeros_like(refined_var))
        return refined_mu, refined_var.clamp_min(0.0)


class AttnResCTtoMRI(e0.AttnResCTtoMRI):
    """E0 backbone + optional residual refinement head."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bottleneck_blocks: int = 6,
        dropout: float = 0.2,
        final_activation: str = "sigmoid",
        use_refiner: bool = True,
        refiner_channels: int = 8,
        residual_scale: float = 0.10,
    ) -> None:
        # E0 Med-ReCL branch is always disabled in this experiment.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bottleneck_blocks=bottleneck_blocks,
            dropout=dropout,
            final_activation=final_activation,
            use_medrecl=False,
        )
        self.use_refiner = bool(use_refiner)
        self.refiner = (
            ResidualRefinementHead3D(refiner_channels, residual_scale)
            if self.use_refiner
            else None
        )

    def _apply_refiner(self, coarse: torch.Tensor) -> torch.Tensor:
        if self.refiner is None:
            return coarse
        return self.refiner(coarse)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        coarse, _ = e0.AttnResCTtoMRI._forward_backbone(
            self,
            source,
            return_features=False,
        )
        return self._apply_refiner(coarse)

    def forward_with_features(
        self,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        coarse, features = e0.AttnResCTtoMRI._forward_backbone(
            self,
            source,
            return_features=True,
        )
        assert features is not None
        # Keep a direct coarse branch for potential diagnostics.
        features["coarse_pred"] = [coarse]
        return self._apply_refiner(coarse), features

    def forward_mu_var_cov(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_mu, coarse_var = e0.AttnResCTtoMRI.forward_mu_var_cov(self, source)
        if self.refiner is None:
            return coarse_mu, coarse_var
        return self.refiner.forward_mu_var(coarse_mu, coarse_var)

    def extract_decoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return decoder feature maps for medical feature-regression supervision."""
        _, features = e0.AttnResCTtoMRI._forward_backbone(
            self,
            x,
            return_features=True,
        )
        if features is None or "dec" not in features:
            raise RuntimeError("Failed to extract decoder features.")
        return features["dec"]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load E0 checkpoints by initializing absent refiner weights safely."""
        state_dict = state_dict.copy()
        current_state = self.state_dict()
        for key, value in current_state.items():
            if key.startswith("refiner.") and key not in state_dict:
                state_dict[key] = value.detach().clone()
        return e0.AttnResCTtoMRI.load_state_dict(self, state_dict, strict=strict, assign=assign)
