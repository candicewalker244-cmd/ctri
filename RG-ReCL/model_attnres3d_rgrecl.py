#!/usr/bin/env python3
"""E0 CT-to-MRI reconstruction with Region-aware PatchNCE.

The reconstruction backbone, dataset, evaluation, checkpoint, Moment inference,
and MC-Dropout utilities are reused from the established E0 implementation.
Training adds exactly one auxiliary objective: paired, same-location PatchNCE
whose anchors are sampled with higher probability from top-error regions.

Moment propagation remains an inference-only analysis path. This module does
not instantiate the old Med-ReCL target encoder, EMA teacher, appearance loss,
or extra image/gradient/frequency consistency losses.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

e0 = importlib.import_module("model_attnres3d_gtmedreclpp")


# Re-export the unchanged E0 data, reconstruction, evaluation, and inference API.
np = e0.np
random = e0.random
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
SSIM3D = e0.SSIM3D
compute_mae = e0.compute_mae
compute_psnr = e0.compute_psnr
save_checkpoint = e0.save_checkpoint
write_log_row = e0.write_log_row
validate_one_epoch = e0.validate_one_epoch
validate_with_uncertainty = e0.validate_with_uncertainty
fit_moment_variance_scale = e0.fit_moment_variance_scale


@dataclass
class RegionPatchNCEConfig:
    """Fixed RG-ReCL objective and bounded sampling sizes."""

    temperature: float = 0.07
    hard_ratio: float = 0.20
    feature_dim: int = 64
    num_patches: int = 256
    negative_pool_size: int = 1024
    level_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.hard_ratio <= 1.0:
            raise ValueError("hard_ratio must be in [0, 1]")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.num_patches <= 0:
            raise ValueError("num_patches must be positive")
        if self.negative_pool_size <= 0:
            raise ValueError("negative_pool_size must be positive")
        if len(self.level_weights) != 3 or any(weight < 0.0 for weight in self.level_weights):
            raise ValueError("level_weights must contain three non-negative values")
        if sum(self.level_weights) <= 0.0:
            raise ValueError("at least one level weight must be positive")


class PatchProjectionHead3D(nn.Module):
    """Shared MLP applied only after patch vectors have been sampled."""

    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        hidden_channels = max(in_channels, feature_dim)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, feature_dim, bias=True),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(vectors.float()), dim=-1, eps=1e-6)


class AttnResCTtoMRI(e0.AttnResCTtoMRI):
    """Unchanged E0 backbone plus shared Region-PatchNCE projection heads."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bottleneck_blocks: int = 6,
        dropout: float = 0.0,
        final_activation: str = "sigmoid",
        use_rgrecl: bool = True,
        rgrecl_feature_dim: int = 64,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bottleneck_blocks=bottleneck_blocks,
            dropout=dropout,
            final_activation=final_activation,
            use_medrecl=False,
            medrecl_proj_dim=rgrecl_feature_dim,
        )
        self.use_rgrecl = bool(use_rgrecl)
        self.rgrecl_feature_dim = int(rgrecl_feature_dim)
        level_channels = (base_channels, base_channels * 2, base_channels * 4)
        self.rgrecl_projectors = nn.ModuleList(
            [
                PatchProjectionHead3D(channels, self.rgrecl_feature_dim)
                for channels in level_channels
            ]
        ) if self.use_rgrecl else nn.ModuleList()

    def extract_rgrecl_target_features(self, target: torch.Tensor) -> List[torch.Tensor]:
        """Encode paired GT with the existing encoder and a deterministic stop-gradient path."""
        encoder_blocks = (self.enc1, self.enc2, self.enc3)
        previous_modes = [block.training for block in encoder_blocks]
        try:
            for block in encoder_blocks:
                block.eval()
            with torch.no_grad():
                feature_1 = self.enc1(target.float())
                feature_2 = self.enc2(feature_1)
                feature_3 = self.enc3(feature_2)
        finally:
            for block, previous_mode in zip(encoder_blocks, previous_modes):
                block.train(previous_mode)
        return [feature_1.detach(), feature_2.detach(), feature_3.detach()]

    def project_rgrecl_vectors(
        self,
        vectors: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        if not self.use_rgrecl:
            raise RuntimeError("Region-aware PatchNCE is disabled")
        return self.rgrecl_projectors[level](vectors)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Allow an E0 checkpoint to initialize the backbone for a new RG-ReCL run."""
        if self.use_rgrecl and not any(key.startswith("rgrecl_projectors.") for key in state_dict):
            state_dict = state_dict.copy()
            for key, value in self.state_dict().items():
                if key.startswith("rgrecl_projectors."):
                    state_dict[key] = value.detach().clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class RegionPatchNCELoss(nn.Module):
    """Paired same-position PatchNCE with top-error region-aware sampling."""

    def __init__(self, config: Optional[RegionPatchNCEConfig] = None) -> None:
        super().__init__()
        self.config = config or RegionPatchNCEConfig()
        self.last_metrics: Dict[str, float] = {
            "loss": 0.0,
            "hard_region_fraction": 0.0,
            "sampled_hard_fraction": 0.0,
            "positive_similarity": 0.0,
        }

    def forward(
        self,
        model: AttnResCTtoMRI,
        pred: torch.Tensor,
        target: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        if not model.use_rgrecl:
            return pred.new_zeros(())

        target_features = model.extract_rgrecl_target_features(target)
        error_map = torch.abs(
            torch.clamp(pred.detach().float(), 0.0, 1.0)
            - torch.clamp(target.detach().float(), 0.0, 1.0)
        )

        level_losses: List[torch.Tensor] = []
        level_hard_fractions: List[float] = []
        level_sampled_hard_fractions: List[float] = []
        level_positive_similarities: List[float] = []
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
                )
                batch_losses.append(case_loss)
                level_hard_fractions.append(case_metrics["hard_region_fraction"])
                level_sampled_hard_fractions.append(case_metrics["sampled_hard_fraction"])
                level_positive_similarities.append(case_metrics["positive_similarity"])
            level_losses.append(torch.stack(batch_losses).mean())

        weights = pred.new_tensor(self.config.level_weights, dtype=torch.float32)
        weights = weights[:len(level_losses)]
        loss = torch.sum(weights * torch.stack(level_losses)) / weights.sum().clamp_min(
            self.config.eps
        )
        self.last_metrics = {
            "loss": float(loss.detach().item()),
            "hard_region_fraction": self._mean_or_zero(level_hard_fractions),
            "sampled_hard_fraction": self._mean_or_zero(level_sampled_hard_fractions),
            "positive_similarity": self._mean_or_zero(level_positive_similarities),
        }
        return loss

    def _single_case_loss(
        self,
        model: AttnResCTtoMRI,
        level: int,
        pred_feature: torch.Tensor,
        target_feature: torch.Tensor,
        error_map: torch.Tensor,
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
        if cfg.hard_ratio > 0.0:
            hard_count = min(
                num_positions,
                max(1, int(math.ceil(cfg.hard_ratio * num_positions))),
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

        queries = model.project_rgrecl_vectors(pred_flat[anchor_indices], level)
        positive_keys = model.project_rgrecl_vectors(target_flat[anchor_indices], level)
        negative_keys = model.project_rgrecl_vectors(target_flat[negative_indices], level)
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

        anchor_weight = torch.ones_like(per_anchor_loss)
        sampled_hard = hard_mask[anchor_indices]
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
    rgrecl_criterion: Optional[RegionPatchNCELoss] = None,
    lambda_rgrecl: float = 0.05,
    start_step: int = 0,
    total_steps: int = 1,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """Train E0 and Region-PatchNCE together in one optimizer step per batch."""
    del total_steps
    model.train()
    running = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "rgrecl_loss": 0.0,
        "rgrecl_weighted_loss": 0.0,
        "rgrecl_hard_region_fraction": 0.0,
        "rgrecl_sampled_hard_fraction": 0.0,
        "rgrecl_positive_similarity": 0.0,
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
            if rgrecl_criterion is not None:
                pred, feature_dict = model.forward_with_features(source)
                rec_loss = criterion(pred, target)
                rgrecl_loss = rgrecl_criterion(model, pred, target, feature_dict)
                weighted_rgrecl_loss = float(lambda_rgrecl) * rgrecl_loss
            else:
                pred = model(source)
                rec_loss = criterion(pred, target)
                rgrecl_loss = rec_loss.new_zeros(())
                weighted_rgrecl_loss = rec_loss.new_zeros(())
            loss = rec_loss + weighted_rgrecl_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss before backward: "
                f"total={loss.detach().item()}, rec={rec_loss.detach().item()}, "
                f"rgrecl={weighted_rgrecl_loss.detach().item()}"
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

        rgrecl_metrics = (
            rgrecl_criterion.last_metrics
            if rgrecl_criterion is not None
            else {
                "hard_region_fraction": 0.0,
                "sampled_hard_fraction": 0.0,
                "positive_similarity": 0.0,
            }
        )
        running["loss"] += float(loss.detach().item())
        running["rec_loss"] += float(rec_loss.detach().item())
        running["rgrecl_loss"] += float(rgrecl_loss.detach().item())
        running["rgrecl_weighted_loss"] += float(weighted_rgrecl_loss.detach().item())
        running["rgrecl_hard_region_fraction"] += rgrecl_metrics["hard_region_fraction"]
        running["rgrecl_sampled_hard_fraction"] += rgrecl_metrics["sampled_hard_fraction"]
        running["rgrecl_positive_similarity"] += rgrecl_metrics["positive_similarity"]
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
