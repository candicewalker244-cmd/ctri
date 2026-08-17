#!/usr/bin/env python3
"""Slim RA-ReCL for paired CT-to-MRI reconstruction.

Training contains exactly three terms:
    L_total = L_rec + lambda_latent * L_latent
                    + lambda_region * L_region_patchnce

This file replaces the unrun RA-MedReCL++ experiment in place. The E0
reconstruction and inference paths are unchanged. Image, gradient, and
frequency consistency are not duplicated here; EMA and Moment-based training
weights are absent. Moment propagation remains available only for evaluation.
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


# Re-export the unchanged project API required by the training entry point.
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
class RAReCLConfig:
    """Slim latent-alignment and Region-PatchNCE configuration."""

    temperature: float = 0.07
    hard_ratio: float = 0.20
    feature_dim: int = 64
    latent_samples: int = 256
    region_patches: int = 256
    negative_pool_size: int = 1024
    level_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.hard_ratio <= 1.0:
            raise ValueError("hard_ratio must be in [0, 1]")
        for name, value in (
            ("feature_dim", self.feature_dim),
            ("latent_samples", self.latent_samples),
            ("region_patches", self.region_patches),
            ("negative_pool_size", self.negative_pool_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.level_weights) != 3 or any(weight < 0.0 for weight in self.level_weights):
            raise ValueError("level_weights must contain three non-negative values")
        if sum(self.level_weights) <= 0.0:
            raise ValueError("at least one level weight must be positive")


class AttnResCTtoMRI(rg.AttnResCTtoMRI):
    """E0 backbone with one shared projector set for both slim RA objectives."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bottleneck_blocks: int = 6,
        dropout: float = 0.0,
        final_activation: str = "sigmoid",
        use_rarecl: bool = True,
        rarecl_feature_dim: int = 64,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bottleneck_blocks=bottleneck_blocks,
            dropout=dropout,
            final_activation=final_activation,
            use_rgrecl=use_rarecl,
            rgrecl_feature_dim=rarecl_feature_dim,
        )
        self.use_rarecl = bool(use_rarecl)
        self.rarecl_feature_dim = int(rarecl_feature_dim)
        self.rarecl_projectors = self.rgrecl_projectors
        del self.rgrecl_projectors

    def extract_rarecl_target_features(self, target: torch.Tensor) -> List[torch.Tensor]:
        return self.extract_rgrecl_target_features(target)

    def project_rarecl_vectors(self, vectors: torch.Tensor, level: int) -> torch.Tensor:
        if not self.use_rarecl:
            raise RuntimeError("Slim RA-ReCL is disabled")
        return self.rarecl_projectors[level](vectors)

    # RegionPatchNCELoss uses this small interface; it points to the same RA heads.
    def project_rgrecl_vectors(self, vectors: torch.Tensor, level: int) -> torch.Tensor:
        return self.project_rarecl_vectors(vectors, level)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Allow E0 initialization while keeping newly initialized RA projectors."""
        if self.use_rarecl and not any(key.startswith("rarecl_projectors.") for key in state_dict):
            state_dict = state_dict.copy()
            for key, value in self.state_dict().items():
                if key.startswith("rarecl_projectors."):
                    state_dict[key] = value.detach().clone()
        return rg.e0.AttnResCTtoMRI.load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )


class RAReCLLoss(nn.Module):
    """Shared-feature latent alignment plus Region-aware PatchNCE."""

    def __init__(self, config: Optional[RAReCLConfig] = None) -> None:
        super().__init__()
        self.config = config or RAReCLConfig()
        self.region_loss = rg.RegionPatchNCELoss(
            rg.RegionPatchNCEConfig(
                temperature=self.config.temperature,
                hard_ratio=self.config.hard_ratio,
                feature_dim=self.config.feature_dim,
                num_patches=self.config.region_patches,
                negative_pool_size=self.config.negative_pool_size,
                level_weights=self.config.level_weights,
            )
        )
        self.last_metrics: Dict[str, float] = {
            "latent_loss": 0.0,
            "region_loss": 0.0,
            "latent_similarity": 0.0,
            "region_positive_similarity": 0.0,
            "hard_region_fraction": 0.0,
            "sampled_hard_fraction": 0.0,
        }

    def components(
        self,
        model: AttnResCTtoMRI,
        pred: torch.Tensor,
        target: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not model.use_rarecl:
            zero = pred.new_zeros(())
            return zero, zero

        target_features = model.extract_rarecl_target_features(target)
        error_map = torch.abs(
            torch.clamp(pred.detach().float(), 0.0, 1.0)
            - torch.clamp(target.detach().float(), 0.0, 1.0)
        )
        latent_level_losses: List[torch.Tensor] = []
        region_level_losses: List[torch.Tensor] = []
        latent_similarities: List[float] = []
        hard_fractions: List[float] = []
        sampled_hard_fractions: List[float] = []
        region_similarities: List[float] = []

        for level, (pred_feature, target_feature) in enumerate(
            zip(feature_dict["dec"], target_features)
        ):
            error_level = F.interpolate(
                error_map,
                size=pred_feature.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            latent_batch_losses: List[torch.Tensor] = []
            region_batch_losses: List[torch.Tensor] = []
            for batch_index in range(pred_feature.shape[0]):
                latent_loss, latent_similarity = self._single_case_latent_loss(
                    model,
                    level,
                    pred_feature[batch_index],
                    target_feature[batch_index],
                )
                region_loss, region_metrics = self.region_loss._single_case_loss(
                    model,
                    level,
                    pred_feature[batch_index],
                    target_feature[batch_index],
                    error_level[batch_index],
                )
                latent_batch_losses.append(latent_loss)
                region_batch_losses.append(region_loss)
                latent_similarities.append(latent_similarity)
                hard_fractions.append(region_metrics["hard_region_fraction"])
                sampled_hard_fractions.append(region_metrics["sampled_hard_fraction"])
                region_similarities.append(region_metrics["positive_similarity"])
            latent_level_losses.append(torch.stack(latent_batch_losses).mean())
            region_level_losses.append(torch.stack(region_batch_losses).mean())

        weights = pred.new_tensor(self.config.level_weights, dtype=torch.float32)
        weights = weights[:len(latent_level_losses)]
        denominator = weights.sum().clamp_min(self.config.eps)
        latent_loss = torch.sum(weights * torch.stack(latent_level_losses)) / denominator
        region_loss = torch.sum(weights * torch.stack(region_level_losses)) / denominator
        self.last_metrics = {
            "latent_loss": float(latent_loss.detach().item()),
            "region_loss": float(region_loss.detach().item()),
            "latent_similarity": self._mean_or_zero(latent_similarities),
            "region_positive_similarity": self._mean_or_zero(region_similarities),
            "hard_region_fraction": self._mean_or_zero(hard_fractions),
            "sampled_hard_fraction": self._mean_or_zero(sampled_hard_fractions),
        }
        return latent_loss, region_loss

    def forward(
        self,
        model: AttnResCTtoMRI,
        pred: torch.Tensor,
        target: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        latent_loss, region_loss = self.components(model, pred, target, feature_dict)
        return latent_loss + region_loss

    def _single_case_latent_loss(
        self,
        model: AttnResCTtoMRI,
        level: int,
        pred_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        channels, depth, height, width = pred_feature.shape
        num_positions = depth * height * width
        sample_count = min(self.config.latent_samples, num_positions)
        sample_indices = torch.randperm(
            num_positions,
            device=pred_feature.device,
        )[:sample_count]
        pred_vectors = pred_feature.reshape(channels, num_positions).transpose(0, 1)
        target_vectors = target_feature.reshape(channels, num_positions).transpose(0, 1)
        pred_projected = model.project_rarecl_vectors(pred_vectors[sample_indices], level)
        target_projected = model.project_rarecl_vectors(target_vectors[sample_indices], level)
        similarity = torch.sum(pred_projected * target_projected, dim=1)
        loss = (1.0 - similarity).mean()
        return loss, float(similarity.detach().mean().item())

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
    rarecl_criterion: Optional[RAReCLLoss] = None,
    lambda_latent: float = 0.05,
    lambda_region: float = 0.05,
    start_step: int = 0,
    total_steps: int = 1,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """Train E0, latent alignment, and Region-PatchNCE in one backward pass."""
    del total_steps
    model.train()
    running = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "rarecl_latent_loss": 0.0,
        "rarecl_region_loss": 0.0,
        "rarecl_weighted_latent_loss": 0.0,
        "rarecl_weighted_region_loss": 0.0,
        "rarecl_hard_region_fraction": 0.0,
        "rarecl_sampled_hard_fraction": 0.0,
        "rarecl_latent_similarity": 0.0,
        "rarecl_region_positive_similarity": 0.0,
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
        with autocast(device_type=device.type, enabled=amp):
            if rarecl_criterion is not None:
                pred, feature_dict = model.forward_with_features(source)
                rec_loss = criterion(pred, target)
                latent_loss, region_loss = rarecl_criterion.components(
                    model,
                    pred,
                    target,
                    feature_dict,
                )
                weighted_latent_loss = float(lambda_latent) * latent_loss
                weighted_region_loss = float(lambda_region) * region_loss
            else:
                pred = model(source)
                rec_loss = criterion(pred, target)
                latent_loss = rec_loss.new_zeros(())
                region_loss = rec_loss.new_zeros(())
                weighted_latent_loss = rec_loss.new_zeros(())
                weighted_region_loss = rec_loss.new_zeros(())
            loss = rec_loss + weighted_latent_loss + weighted_region_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss before backward: "
                f"total={loss.detach().item()}, rec={rec_loss.detach().item()}, "
                f"latent={weighted_latent_loss.detach().item()}, "
                f"region={weighted_region_loss.detach().item()}"
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

        metrics = rarecl_criterion.last_metrics if rarecl_criterion is not None else {
            "hard_region_fraction": 0.0,
            "sampled_hard_fraction": 0.0,
            "latent_similarity": 0.0,
            "region_positive_similarity": 0.0,
        }
        running["loss"] += float(loss.detach().item())
        running["rec_loss"] += float(rec_loss.detach().item())
        running["rarecl_latent_loss"] += float(latent_loss.detach().item())
        running["rarecl_region_loss"] += float(region_loss.detach().item())
        running["rarecl_weighted_latent_loss"] += float(weighted_latent_loss.detach().item())
        running["rarecl_weighted_region_loss"] += float(weighted_region_loss.detach().item())
        running["rarecl_hard_region_fraction"] += metrics["hard_region_fraction"]
        running["rarecl_sampled_hard_fraction"] += metrics["sampled_hard_fraction"]
        running["rarecl_latent_similarity"] += metrics["latent_similarity"]
        running["rarecl_region_positive_similarity"] += metrics["region_positive_similarity"]
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
