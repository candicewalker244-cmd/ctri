#!/usr/bin/env python3
"""RA-ReCL v2 for paired 3D CT-to-MRI reconstruction.

The new experiment keeps E0 reconstruction intact and adds exactly two
mechanisms:

1. a zero-initialized image-domain residual refinement head;
2. curriculum-controlled multi-scale Region-aware PatchNCE.

The active training objective is:

    L_total = L_rec + lambda_patchnce * L_multiscale_region_patchnce

During the first curriculum stage lambda_patchnce is zero. Moment propagation
remains inference-only, but the residual refiner is included in that path so
ordinary and Moment predictions represent the same trained model.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RG_DIR = PROJECT_ROOT / "RG-ReCL"
for dependency_dir in (PROJECT_ROOT, RG_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

rg = importlib.import_module("model_attnres3d_rgrecl")


# Re-export the established E0 data, reconstruction, evaluation, and I/O API.
np = rg.np
random = rg.random
set_seed = rg.set_seed
ensure_dir = rg.ensure_dir
discover_cases = rg.discover_cases
validate_case_splits = rg.validate_case_splits
pad_tensor_to_divisible = rg.pad_tensor_to_divisible
unpad_tensor = rg.unpad_tensor
crop_pair_around_foreground = rg.crop_pair_around_foreground
parse_eval_crop_size = rg.parse_eval_crop_size
CTMRIDataset = rg.CTMRIDataset
ReconstructionLoss = rg.ReconstructionLoss
SSIM3D = rg.SSIM3D
compute_mae = rg.compute_mae
compute_psnr = rg.compute_psnr
save_checkpoint = rg.save_checkpoint
write_log_row = rg.write_log_row
validate_one_epoch = rg.validate_one_epoch
validate_with_uncertainty = rg.validate_with_uncertainty
fit_moment_variance_scale = rg.fit_moment_variance_scale


@dataclass
class RAReCLV2Config:
    """Multi-scale Region-PatchNCE and curriculum configuration."""

    temperature: float = 0.07
    feature_dim: int = 64
    num_patches: int = 256
    negative_pool_size: int = 1024
    # feature_dict["dec"] is ordered shallow-to-deep: d1, d2, d3.
    level_weights: Tuple[float, float, float] = (0.20, 0.30, 0.50)
    warmup_end_ratio: float = 0.20
    middle_end_ratio: float = 0.60
    middle_hard_ratio: float = 0.30
    final_hard_ratio: float = 0.20
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        for name, value in (
            ("feature_dim", self.feature_dim),
            ("num_patches", self.num_patches),
            ("negative_pool_size", self.negative_pool_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.level_weights) != 3 or any(weight < 0.0 for weight in self.level_weights):
            raise ValueError("level_weights must contain three non-negative values")
        if sum(self.level_weights) <= 0.0:
            raise ValueError("at least one level weight must be positive")
        if not 0.0 <= self.warmup_end_ratio < self.middle_end_ratio <= 1.0:
            raise ValueError(
                "curriculum ratios must satisfy 0 <= warmup < middle <= 1"
            )
        for name, value in (
            ("middle_hard_ratio", self.middle_hard_ratio),
            ("final_hard_ratio", self.final_hard_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def curriculum(self, progress: float) -> Tuple[int, float, bool]:
        """Return stage index, hard ratio, and whether PatchNCE is active."""
        progress = min(1.0, max(0.0, float(progress)))
        if progress < self.warmup_end_ratio:
            return 0, 0.0, False
        if progress < self.middle_end_ratio:
            return 1, self.middle_hard_ratio, True
        return 2, self.final_hard_ratio, True


class ResidualRefinementHead3D(nn.Module):
    """Small image-domain refiner that predicts a bounded local residual."""

    def __init__(self, hidden_channels: int = 8, residual_scale: float = 0.10) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = float(residual_scale)
        self.conv1 = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)
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
        """First-order diagonal moment propagation through the refiner."""
        hidden_mu = self.conv1(coarse_mu)
        hidden_var = F.conv3d(
            coarse_var,
            self.conv1.weight.square(),
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )
        gelu_mu, gelu_derivative = rg.e0.gelu_with_derivative(hidden_mu)
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
            self.residual_scale**2
            * tanh_derivative.square()
            * residual_pre_var
        )

        unclamped_mu = coarse_mu + residual_mu
        refined_mu = torch.clamp(unclamped_mu, 0.0, 1.0)
        # The diagonal approximation omits coarse/residual covariance, matching
        # the project's existing diagonal moment treatment outside attention.
        refined_var = coarse_var + residual_var
        inside = (unclamped_mu > 0.0) & (unclamped_mu < 1.0)
        refined_var = torch.where(inside, refined_var, torch.zeros_like(refined_var))
        return refined_mu, refined_var.clamp_min(0.0)


class AttnResCTtoMRI(rg.AttnResCTtoMRI):
    """E0 backbone with residual refinement and multi-scale PatchNCE heads."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bottleneck_blocks: int = 6,
        dropout: float = 0.0,
        final_activation: str = "sigmoid",
        use_rareclv2: bool = True,
        rareclv2_feature_dim: int = 64,
        use_refiner: bool = True,
        refiner_channels: int = 8,
        residual_scale: float = 0.10,
    ) -> None:
        if out_channels != 1:
            raise ValueError("RA-ReCL v2 residual refiner currently requires one output channel")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bottleneck_blocks=bottleneck_blocks,
            dropout=dropout,
            final_activation=final_activation,
            use_rgrecl=use_rareclv2,
            rgrecl_feature_dim=rareclv2_feature_dim,
        )
        self.use_rareclv2 = bool(use_rareclv2)
        self.rareclv2_feature_dim = int(rareclv2_feature_dim)
        self.rareclv2_projectors = self.rgrecl_projectors
        del self.rgrecl_projectors
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
        coarse, _ = rg.e0.AttnResCTtoMRI._forward_backbone(
            self,
            source,
            return_features=False,
        )
        return self._apply_refiner(coarse)

    def forward_with_features(
        self,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        coarse, features = rg.e0.AttnResCTtoMRI._forward_backbone(
            self,
            source,
            return_features=True,
        )
        assert features is not None
        features["coarse_pred"] = [coarse]
        return self._apply_refiner(coarse), features

    def forward_mu_var_cov(
        self,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_mu, coarse_var = rg.e0.AttnResCTtoMRI.forward_mu_var_cov(self, source)
        if self.refiner is None:
            return coarse_mu, coarse_var
        return self.refiner.forward_mu_var(coarse_mu, coarse_var)

    def extract_rareclv2_target_features(self, target: torch.Tensor) -> List[torch.Tensor]:
        return self.extract_rgrecl_target_features(target)

    def project_rareclv2_vectors(
        self,
        vectors: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        if not self.use_rareclv2:
            raise RuntimeError("RA-ReCL v2 PatchNCE is disabled")
        return self.rareclv2_projectors[level](vectors)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load E0 weights strictly while retaining new zero-safe modules."""
        state_dict = state_dict.copy()
        current_state = self.state_dict()
        new_prefixes = ("rareclv2_projectors.", "refiner.")
        for key, value in current_state.items():
            if key not in state_dict and key.startswith(new_prefixes):
                state_dict[key] = value.detach().clone()
        return rg.e0.AttnResCTtoMRI.load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )


class MultiScaleRegionPatchNCELoss(nn.Module):
    """Paired multi-scale PatchNCE with curriculum-selected hard regions."""

    def __init__(self, config: Optional[RAReCLV2Config] = None) -> None:
        super().__init__()
        self.config = config or RAReCLV2Config()
        self.last_metrics: Dict[str, float] = {
            "loss": 0.0,
            "hard_region_fraction": 0.0,
            "sampled_hard_fraction": 0.0,
            "positive_similarity": 0.0,
            "curriculum_stage": 0.0,
            "effective_hard_ratio": 0.0,
            "patchnce_active": 0.0,
        }

    def forward(
        self,
        model: AttnResCTtoMRI,
        pred: torch.Tensor,
        target: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
        hard_ratio: float,
        curriculum_stage: int,
    ) -> torch.Tensor:
        if not model.use_rareclv2:
            return pred.new_zeros(())

        target_features = model.extract_rareclv2_target_features(target)
        error_map = torch.abs(
            torch.clamp(pred.detach().float(), 0.0, 1.0)
            - torch.clamp(target.detach().float(), 0.0, 1.0)
        )
        level_losses: List[torch.Tensor] = []
        hard_fractions: List[float] = []
        sampled_hard_fractions: List[float] = []
        positive_similarities: List[float] = []

        for level, (pred_feature, target_feature) in enumerate(
            zip(feature_dict["dec"], target_features)
        ):
            error_level = F.interpolate(
                error_map,
                size=pred_feature.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            batch_losses: List[torch.Tensor] = []
            for batch_index in range(pred_feature.shape[0]):
                case_loss, case_metrics = self._single_case_loss(
                    model,
                    level,
                    pred_feature[batch_index],
                    target_feature[batch_index],
                    error_level[batch_index],
                    hard_ratio,
                )
                batch_losses.append(case_loss)
                hard_fractions.append(case_metrics["hard_region_fraction"])
                sampled_hard_fractions.append(case_metrics["sampled_hard_fraction"])
                positive_similarities.append(case_metrics["positive_similarity"])
            level_losses.append(torch.stack(batch_losses).mean())

        weights = pred.new_tensor(self.config.level_weights, dtype=torch.float32)
        weights = weights[:len(level_losses)]
        loss = torch.sum(weights * torch.stack(level_losses)) / weights.sum().clamp_min(
            self.config.eps
        )
        self.last_metrics = {
            "loss": float(loss.detach().item()),
            "hard_region_fraction": self._mean_or_zero(hard_fractions),
            "sampled_hard_fraction": self._mean_or_zero(sampled_hard_fractions),
            "positive_similarity": self._mean_or_zero(positive_similarities),
            "curriculum_stage": float(curriculum_stage),
            "effective_hard_ratio": float(hard_ratio),
            "patchnce_active": 1.0,
        }
        return loss

    def _single_case_loss(
        self,
        model: AttnResCTtoMRI,
        level: int,
        pred_feature: torch.Tensor,
        target_feature: torch.Tensor,
        error_map: torch.Tensor,
        hard_ratio: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        channels, depth, height, width = pred_feature.shape
        num_positions = depth * height * width
        if num_positions < 2:
            zero = pred_feature.new_zeros(())
            return zero, {
                "hard_region_fraction": 0.0,
                "sampled_hard_fraction": 0.0,
                "positive_similarity": 0.0,
            }

        pred_flat = pred_feature.reshape(channels, num_positions).transpose(0, 1)
        target_flat = target_feature.reshape(channels, num_positions).transpose(0, 1)
        error_flat = error_map.reshape(-1).float()
        hard_count = 0
        if hard_ratio > 0.0:
            hard_count = min(
                num_positions,
                max(1, int(math.ceil(float(hard_ratio) * num_positions))),
            )
        hard_mask = torch.zeros(num_positions, dtype=torch.bool, device=error_flat.device)
        if hard_count > 0:
            hard_indices = torch.topk(
                error_flat,
                k=hard_count,
                largest=True,
                sorted=False,
            ).indices
            hard_mask[hard_indices] = True

        sampling_weight = torch.ones_like(error_flat)
        sampling_weight[hard_mask] = 1.0 + error_flat[hard_mask]
        anchor_count = min(cfg.num_patches, num_positions)
        anchor_indices = torch.multinomial(
            sampling_weight,
            num_samples=anchor_count,
            replacement=False,
        )
        negative_count = min(cfg.negative_pool_size, num_positions)
        negative_indices = torch.randperm(
            num_positions,
            device=pred_feature.device,
        )[:negative_count]

        queries = model.project_rareclv2_vectors(pred_flat[anchor_indices], level)
        positive_keys = model.project_rareclv2_vectors(
            target_flat[anchor_indices],
            level,
        )
        negative_keys = model.project_rareclv2_vectors(
            target_flat[negative_indices],
            level,
        )
        positive_similarity = torch.sum(queries * positive_keys, dim=1)
        positive_logits = positive_similarity[:, None] / cfg.temperature
        negative_logits = queries @ negative_keys.transpose(0, 1)
        same_position = anchor_indices[:, None] == negative_indices[None, :]
        negative_logits = (negative_logits / cfg.temperature).masked_fill(
            same_position,
            torch.finfo(negative_logits.dtype).min,
        )
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(anchor_count, dtype=torch.long, device=logits.device)
        per_anchor_loss = F.cross_entropy(logits, labels, reduction="none")

        sampled_hard = hard_mask[anchor_indices]
        anchor_weight = torch.ones_like(per_anchor_loss)
        anchor_weight[sampled_hard] = 1.0 + error_flat[anchor_indices[sampled_hard]]
        loss = torch.sum(anchor_weight * per_anchor_loss) / anchor_weight.sum().clamp_min(
            cfg.eps
        )
        return loss, {
            "hard_region_fraction": float(hard_mask.float().mean().item()),
            "sampled_hard_fraction": float(sampled_hard.float().mean().item()),
            "positive_similarity": float(positive_similarity.detach().mean().item()),
        }

    @staticmethod
    def _mean_or_zero(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0


def train_one_epoch(
    model: AttnResCTtoMRI,
    loader,
    optimizer,
    criterion: ReconstructionLoss,
    device: torch.device,
    scaler: Optional[GradScaler],
    amp: bool,
    rareclv2_criterion: Optional[MultiScaleRegionPatchNCELoss] = None,
    lambda_patchnce: float = 0.05,
    start_step: int = 0,
    total_steps: int = 1,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """Train refinement and curriculum PatchNCE in one optimizer step per batch."""
    model.train()
    running = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "rareclv2_patchnce_loss": 0.0,
        "rareclv2_weighted_patchnce_loss": 0.0,
        "rareclv2_effective_lambda": 0.0,
        "rareclv2_curriculum_stage": 0.0,
        "rareclv2_effective_hard_ratio": 0.0,
        "rareclv2_patchnce_active": 0.0,
        "rareclv2_hard_region_fraction": 0.0,
        "rareclv2_sampled_hard_fraction": 0.0,
        "rareclv2_positive_similarity": 0.0,
        "refinement_abs_delta": 0.0,
        "refinement_max_abs_delta": 0.0,
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    rec_component_names = (
        "l1", "ms_ssim", "edge", "frequency",
        "weighted_l1", "weighted_ms_ssim", "weighted_edge", "weighted_frequency",
    )
    running_rec_components = {name: 0.0 for name in rec_component_names}
    ssim_fn = SSIM3D(channels=1).to(device)
    num_batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    consecutive_skipped_steps = 0
    running_grad_norm = 0.0
    grad_norm_batches = 0

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        global_step = start_step + num_batches
        progress = min(1.0, max(0.0, global_step / max(1, total_steps)))
        stage = 0
        hard_ratio = 0.0
        patchnce_active = False
        if rareclv2_criterion is not None:
            stage, hard_ratio, patchnce_active = (
                rareclv2_criterion.config.curriculum(progress)
            )
        effective_lambda = float(lambda_patchnce) if patchnce_active else 0.0

        with autocast(device_type=device.type, enabled=amp):
            pred, feature_dict = model.forward_with_features(source)
            coarse_pred = feature_dict["coarse_pred"][0]
            if patchnce_active and rareclv2_criterion is not None:
                rec_loss = criterion(pred, target)
                patchnce_loss = rareclv2_criterion(
                    model,
                    pred,
                    target,
                    feature_dict,
                    hard_ratio=hard_ratio,
                    curriculum_stage=stage,
                )
                weighted_patchnce_loss = effective_lambda * patchnce_loss
            else:
                rec_loss = criterion(pred, target)
                patchnce_loss = rec_loss.new_zeros(())
                weighted_patchnce_loss = rec_loss.new_zeros(())
                if rareclv2_criterion is not None:
                    rareclv2_criterion.last_metrics = {
                        "loss": 0.0,
                        "hard_region_fraction": 0.0,
                        "sampled_hard_fraction": 0.0,
                        "positive_similarity": 0.0,
                        "curriculum_stage": float(stage),
                        "effective_hard_ratio": float(hard_ratio),
                        "patchnce_active": 0.0,
                    }
            loss = rec_loss + weighted_patchnce_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss before backward: "
                f"total={loss.detach().item()}, rec={rec_loss.detach().item()}, "
                f"patchnce={weighted_patchnce_loss.detach().item()}"
            )

        optimizer_step_succeeded = False
        grad_norm_value = float("nan")
        if scaler is not None and amp:
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False
            )
            grad_norm_value = float(grad_norm.detach().float().item())
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale and math.isfinite(grad_norm_value):
                optimizer_steps += 1
                optimizer_step_succeeded = True
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False
            )
            grad_norm_value = float(grad_norm.detach().float().item())
            if not math.isfinite(grad_norm_value):
                raise FloatingPointError(
                    f"Non-finite gradient norm in FP32 training: {grad_norm_value}"
                )
            optimizer.step()
            optimizer_steps += 1
            optimizer_step_succeeded = True

        if optimizer_step_succeeded:
            consecutive_skipped_steps = 0
        else:
            skipped_optimizer_steps += 1
            consecutive_skipped_steps += 1
            if consecutive_skipped_steps >= 12:
                raise FloatingPointError("AMP skipped 12 consecutive optimizer steps")
        if math.isfinite(grad_norm_value):
            running_grad_norm += grad_norm_value
            grad_norm_batches += 1

        with torch.no_grad():
            pred_clamped = torch.clamp(pred, 0.0, 1.0)
            target_clamped = torch.clamp(target, 0.0, 1.0)
            mae = compute_mae(pred_clamped, target_clamped)
            psnr = compute_psnr(pred_clamped, target_clamped)
            ssim = ssim_fn(pred_clamped, target_clamped)

        metrics = (
            rareclv2_criterion.last_metrics
            if rareclv2_criterion is not None
            else {
                "hard_region_fraction": 0.0,
                "sampled_hard_fraction": 0.0,
                "positive_similarity": 0.0,
                "curriculum_stage": 0.0,
                "effective_hard_ratio": 0.0,
                "patchnce_active": 0.0,
            }
        )
        running["loss"] += float(loss.detach().item())
        running["rec_loss"] += float(rec_loss.detach().item())
        running["rareclv2_patchnce_loss"] += float(patchnce_loss.detach().item())
        running["rareclv2_weighted_patchnce_loss"] += float(
            weighted_patchnce_loss.detach().item()
        )
        running["rareclv2_effective_lambda"] += effective_lambda
        running["rareclv2_curriculum_stage"] += metrics["curriculum_stage"]
        running["rareclv2_effective_hard_ratio"] += metrics["effective_hard_ratio"]
        running["rareclv2_patchnce_active"] += metrics["patchnce_active"]
        running["rareclv2_hard_region_fraction"] += metrics["hard_region_fraction"]
        running["rareclv2_sampled_hard_fraction"] += metrics["sampled_hard_fraction"]
        running["rareclv2_positive_similarity"] += metrics["positive_similarity"]
        refinement_delta = torch.abs(pred.detach() - coarse_pred.detach())
        running["refinement_abs_delta"] += float(refinement_delta.mean().item())
        running["refinement_max_abs_delta"] += float(refinement_delta.max().item())
        running["mae"] += float(mae.item())
        running["psnr"] += float(psnr.item())
        running["ssim"] += float(ssim.item())
        batch_rec_components = getattr(criterion, "last_components", {})
        for name in rec_component_names:
            running_rec_components[name] += float(batch_rec_components.get(name, 0.0))
        num_batches += 1

    denominator = max(1, num_batches)
    results = {name: value / denominator for name, value in running.items()}
    for name in rec_component_names:
        results[f"rec_{name}"] = running_rec_components[name] / denominator
    results.update(
        {
            "optimizer_steps": optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "grad_norm": running_grad_norm / max(1, grad_norm_batches),
            "amp_scale": float(scaler.get_scale()) if scaler is not None else 1.0,
            "next_step": start_step + num_batches,
        }
    )
    return results
