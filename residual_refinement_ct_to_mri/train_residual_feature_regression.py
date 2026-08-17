#!/usr/bin/env python3
"""Train CT->MRI residual refinement + medical feature regression."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import model_attnres3d_residual as net
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from train_gtmedreclpp import save_test_comparison_figures

set_seed = net.set_seed
discover_cases = net.discover_cases
validate_case_splits = net.validate_case_splits
CTMRIDataset = net.CTMRIDataset
AttnResCTtoMRI = net.AttnResCTtoMRI
ReconstructionLoss = net.ReconstructionLoss
validate_one_epoch = net.validate_one_epoch
save_checkpoint = net.save_checkpoint
write_log_row = net.write_log_row
SSIM3D = net.e0.SSIM3D
compute_mae = net.e0.compute_mae
compute_psnr = net.e0.compute_psnr


class MedicalFeatureRegressionLoss(nn.Module):
    """Weak MSE/L1/cosine feature consistency on decoder features."""

    def __init__(
        self,
        weight: float = 0.01,
        level_weights: Tuple[float, float, float] = (0.20, 0.30, 0.50),
        feature_loss: str = "l2",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError("feature_reg_weight must be >= 0")
        if feature_loss not in {"l1", "l2", "cosine"}:
            raise ValueError("feature_loss must be one of: l1, l2, cosine")
        if any(v < 0.0 for v in level_weights):
            raise ValueError("feature_reg_level_weights must be non-negative")
        if sum(level_weights) <= 0.0:
            raise ValueError("sum(feature_reg_level_weights) must be > 0")

        self.weight = float(weight)
        self.level_weights = tuple(float(v) for v in level_weights)
        self.feature_loss = feature_loss
        self.eps = float(eps)
        self.last_metrics: Dict[str, float] = {
            "feature_reg_raw_loss": 0.0,
            "feature_reg_weighted_loss": 0.0,
        }

    def _extract_target_features(self, model: nn.Module, target: torch.Tensor) -> List[torch.Tensor]:
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad():
                target_features = model.extract_decoder_features(target)
        finally:
            if was_training:
                model.train()
        return target_features

    def _pair_loss(self, pred_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
        if self.feature_loss == "l1":
            return F.l1_loss(pred_feature, target_feature)
        if self.feature_loss == "cosine":
            pred_flat = pred_feature.reshape(pred_feature.size(0), -1)
            target_flat = target_feature.reshape(target_feature.size(0), -1)
            pred_n = F.normalize(pred_flat, p=2.0, dim=1, eps=self.eps)
            target_n = F.normalize(target_flat, p=2.0, dim=1, eps=self.eps)
            return 1.0 - torch.mean((pred_n * target_n).sum(dim=1))
        return F.mse_loss(pred_feature, target_feature)

    def __call__(
        self,
        pred_features: Dict[str, List[torch.Tensor]],
        target: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        if self.weight <= 0.0:
            self.last_metrics.update(
                feature_reg_raw_loss=0.0,
                feature_reg_weighted_loss=0.0,
            )
            return target.new_zeros(())

        target_features = self._extract_target_features(model, target)
        pred_features = pred_features["dec"]
        level_n = min(len(pred_features), len(target_features), len(self.level_weights))
        if level_n == 0:
            self.last_metrics.update(
                feature_reg_raw_loss=0.0,
                feature_reg_weighted_loss=0.0,
            )
            return target.new_zeros(())

        weighted_sum = pred_features[0].new_zeros(())
        total_weight = 0.0
        level_vals: List[float] = []
        for idx, level_weight in zip(range(level_n), self.level_weights):
            w = float(level_weight)
            if w <= 0.0:
                continue
            level_loss = self._pair_loss(pred_features[idx], target_features[idx].detach())
            weighted_sum = weighted_sum + level_loss * w
            total_weight += w
            level_vals.append(float(level_loss.detach().item()))

        if total_weight <= 0.0:
            raw = pred_features[0].new_zeros(())
        else:
            raw = weighted_sum / total_weight

        weighted = raw * self.weight
        self.last_metrics.update(
            feature_reg_raw_loss=float(raw.detach().item()),
            feature_reg_weighted_loss=float(weighted.detach().item()),
            feature_reg_level0=level_vals[0] if len(level_vals) > 0 else 0.0,
            feature_reg_level1=level_vals[1] if len(level_vals) > 1 else 0.0,
            feature_reg_level2=level_vals[2] if len(level_vals) > 2 else 0.0,
        )
        return weighted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D CT->MRI residual + medical feature-regression training."
    )
    default_data_dir = str(Path(__file__).resolve().parent.parent / "data" / "dataset")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Dataset root with train/val/test.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        metavar=("D", "H", "W"),
    )
    parser.add_argument("--ct_norm", type=str, default="clip01", choices=["clip01", "zscore_nonzero"])
    parser.add_argument("--mri_norm", type=str, default="clip01", choices=["clip01", "zscore_nonzero"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--bottleneck_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--final_activation", type=str, default="sigmoid")
    parser.add_argument("--refiner_channels", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=0.10)

    parser.add_argument("--feature_reg_weight", type=float, default=0.01)
    parser.add_argument(
        "--feature_reg_loss",
        type=str,
        default="l2",
        choices=["l1", "l2", "cosine"],
    )
    parser.add_argument(
        "--feature_reg_level_weights",
        type=float,
        nargs=3,
        default=(0.20, 0.30, 0.50),
        metavar=("L1", "L2", "L3"),
    )

    parser.add_argument("--l1_weight", type=float, default=0.45)
    parser.add_argument("--ssim_weight", type=float, default=0.30)
    parser.add_argument("--edge_weight", type=float, default=0.15)
    parser.add_argument("--frequency_weight", type=float, default=0.10)
    parser.add_argument("--frequency_alpha", type=float, default=1.0)

    default_save_dir = str(Path(__file__).resolve().parent / "output_residual_feature_regression")
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--no_medrecl",
        dest="medrecl",
        action="store_false",
        help=(
            "Keep interface-compatible flag; Med-ReCL is already disabled in "
            "residual experiments."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--divisor", type=int, default=8)
    parser.add_argument("--eval_crop_size", nargs="+", default="150,150,150")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_test_figures", dest="save_test_figures", action="store_false")
    parser.add_argument("--max_test_figures", type=int, default=0)
    parser.add_argument("--run_test_only", action="store_true")
    parser.set_defaults(save_test_figures=True)
    parser.set_defaults(medrecl=True)
    return parser


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, switch to cpu.")
        return torch.device("cpu")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def _build_dataloaders(args, device):
    data_dir = Path(args.data_dir)
    train_cases = discover_cases(data_dir / "train")
    val_cases = discover_cases(data_dir / "val")
    test_cases = discover_cases(data_dir / "test")
    validate_case_splits({"train": train_cases, "val": val_cases, "test": test_cases})

    pin_memory = device.type == "cuda"
    common_kwargs = dict(num_workers=args.num_workers, pin_memory=pin_memory)

    train_dataset = CTMRIDataset(
        train_cases,
        patch_size=tuple(args.patch_size),
        training=True,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )
    val_dataset = CTMRIDataset(
        val_cases,
        patch_size=tuple(args.patch_size),
        training=False,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )
    test_dataset = CTMRIDataset(
        test_cases,
        patch_size=tuple(args.patch_size),
        training=False,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )

    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **common_kwargs),
        DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, **common_kwargs),
        DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **common_kwargs),
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )


def _build_model(args) -> AttnResCTtoMRI:
    return AttnResCTtoMRI(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
        final_activation=args.final_activation,
        use_refiner=True,
        refiner_channels=args.refiner_channels,
        residual_scale=args.residual_scale,
    )


def _build_rec_loss(args) -> ReconstructionLoss:
    return ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        frequency_weight=args.frequency_weight,
        frequency_alpha=args.frequency_alpha,
    )


def _build_feature_reg(args) -> MedicalFeatureRegressionLoss:
    return MedicalFeatureRegressionLoss(
        weight=args.feature_reg_weight,
        level_weights=tuple(args.feature_reg_level_weights),
        feature_loss=args.feature_reg_loss,
    )


def _train_one_epoch(
    model,
    loader,
    optimizer,
    rec_criterion,
    feature_criterion: MedicalFeatureRegressionLoss,
    device,
    scaler,
    amp,
    max_grad_norm=5.0,
):
    model.train()
    ssim_fn = SSIM3D(channels=1).to(device)
    num_batches = 0
    rec_component_names = ("l1", "ms_ssim", "edge", "frequency", "weighted_l1", "weighted_ms_ssim", "weighted_edge", "weighted_frequency")

    running = {k: 0.0 for k in (
        "loss", "rec_loss", "feature_raw_loss", "feature_weighted_loss",
        "mae", "psnr", "ssim", "grad_norm",
    )}
    running.update({f"rec_{name}": 0.0 for name in rec_component_names})

    from torch.amp import autocast
    grad_norm_sum = 0.0
    grad_norm_batches = 0

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp):
            pred, pred_features = model.forward_with_features(source)
            rec_loss = rec_criterion(pred, target)
            feature_loss = feature_criterion(pred_features, target, model)
            loss = rec_loss + feature_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: total={loss.item()}, rec={rec_loss.item()}, feature={feature_loss.item()}"
            )

        if scaler is not None and amp:
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
            grad_norm_sum += float(grad_norm.detach().float().item())
            grad_norm_batches += 1
            scaler.step(optimizer)
            scaler.update()
            if torch.isfinite(grad_norm) and scaler.get_scale() >= old_scale:
                pass
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
            grad_norm_sum += float(grad_norm.detach().float().item())
            grad_norm_batches += 1
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
            optimizer.step()

        with torch.no_grad():
            pred_c = torch.clamp(pred, 0.0, 1.0)
            mae = compute_mae(pred_c, target)
            psnr = compute_psnr(pred_c, target)
            ssim = ssim_fn(pred_c, target)

        running["loss"] += loss.item()
        running["rec_loss"] += rec_loss.item()
        running["feature_raw_loss"] += float(feature_criterion.last_metrics["feature_reg_raw_loss"])
        running["feature_weighted_loss"] += float(feature_criterion.last_metrics["feature_reg_weighted_loss"])
        running["mae"] += mae.item()
        running["psnr"] += psnr.item()
        running["ssim"] += ssim.item()
        for name in rec_component_names:
            running[f"rec_{name}"] += float(getattr(rec_criterion, "last_components", {}).get(name, 0.0))
        num_batches += 1

    denom = max(1, num_batches)
    running["grad_norm"] = grad_norm_sum / max(1, grad_norm_batches)
    running["loss"] /= denom
    running["rec_loss"] /= denom
    running["feature_raw_loss"] /= denom
    running["feature_weighted_loss"] /= denom
    running["mae"] /= denom
    running["psnr"] /= denom
    running["ssim"] /= denom
    for name in rec_component_names:
        running[f"rec_{name}"] /= denom
    return running


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.patch_size = tuple(args.patch_size)
    args.eval_crop_size = net.parse_eval_crop_size(args.eval_crop_size)

    set_seed(args.seed)
    device = _resolve_device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_len, val_len, test_len = _build_dataloaders(args, device)
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Samples -> train/val/test: {train_len}/{val_len}/{test_len}")

    model = _build_model(args).to(device)
    rec_criterion = _build_rec_loss(args).to(device)
    feature_criterion = _build_feature_reg(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_ssim = -999.0
    best_path = checkpoint_dir / "best.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume file not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metrics = ckpt.get("best_metrics", {})
        best_ssim = float(best_metrics.get("ssim", ckpt.get("best_metric", best_ssim)))

    if args.run_test_only and not (args.resume or best_path.exists()):
        raise ValueError(
            f"run_test_only needs --resume or existing {best_path}"
        )

    log_csv = save_dir / "log.csv"

    if not args.run_test_only:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_metrics = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                rec_criterion=rec_criterion,
                feature_criterion=feature_criterion,
                device=device,
                scaler=scaler,
                amp=bool(args.amp and device.type == "cuda"),
                max_grad_norm=args.max_grad_norm,
            )
            print(
                f"Train: loss={train_metrics['loss']:.6f}, rec={train_metrics['rec_loss']:.6f}, "
                f"feat={train_metrics['feature_weighted_loss']:.6f}, mae={train_metrics['mae']:.6f}, "
                f"psnr={train_metrics['psnr']:.4f}, ssim={train_metrics['ssim']:.6f}"
            )

            if scheduler is not None:
                scheduler.step()

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=rec_criterion,
                device=device,
                amp=bool(args.amp and device.type == "cuda"),
                divisor=args.divisor,
                eval_crop_size=args.eval_crop_size,
            )
            print(
                f"Val: loss={val_metrics['loss']:.6f}, mae={val_metrics['mae']:.6f}, "
                f"psnr={val_metrics['psnr']:.4f}, ssim={val_metrics['ssim']:.6f}, "
                f"hfen={val_metrics['hfen']:.6f}, gmae={val_metrics['gradient_mae']:.6f}"
            )

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_rec_loss": train_metrics["rec_loss"],
                "train_feature_reg_raw_loss": train_metrics["feature_raw_loss"],
                "train_feature_reg_weighted_loss": train_metrics["feature_weighted_loss"],
                "train_mae": train_metrics["mae"],
                "train_psnr": train_metrics["psnr"],
                "train_ssim": train_metrics["ssim"],
                "train_grad_norm": train_metrics["grad_norm"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_foreground_mae": val_metrics["foreground_mae"],
                "val_foreground_psnr": val_metrics["foreground_psnr"],
                "val_foreground_ssim": val_metrics["foreground_ssim"],
                "val_gradient_mae": val_metrics["gradient_mae"],
                "val_hfen": val_metrics["hfen"],
                "feature_reg_weight": feature_criterion.weight,
                "feature_reg_loss": feature_criterion.feature_loss,
                "feature_reg_level_weights": " ".join(
                    f"{w:.4f}" for w in feature_criterion.level_weights
                ),
            }
            for name in ("l1", "ms_ssim", "edge", "frequency", "weighted_l1", "weighted_ms_ssim", "weighted_edge", "weighted_frequency"):
                row[f"train_rec_{name}"] = train_metrics[f"rec_{name}"]
                row[f"val_rec_{name}"] = val_metrics[f"rec_{name}"]
            write_log_row(log_csv, row)

            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                save_checkpoint(
                    best_path, model, optimizer, epoch, best_ssim, args,
                    scheduler=scheduler, scaler=scaler, metric_name="ssim",
                    best_metrics={"ssim": best_ssim, "psnr": val_metrics["psnr"], "hfen": val_metrics["hfen"]}
                )
                print(f"  saved best checkpoint -> {best_path}")

            latest_path = checkpoint_dir / "latest.pth"
            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(
                    latest_path, model, optimizer, epoch, best_ssim, args,
                    scheduler=scheduler, scaler=scaler, metric_name="ssim",
                    best_metrics={"ssim": best_ssim, "psnr": val_metrics["psnr"], "hfen": val_metrics["hfen"]}
                )

    if args.skip_test:
        print("skip_test enabled; stop before evaluation.")
        return

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

    print("Running final test...")
    test_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=rec_criterion,
        device=device,
        amp=bool(args.amp and device.type == "cuda"),
        divisor=args.divisor,
        eval_crop_size=args.eval_crop_size,
    )
    print(
        f"Test: loss={test_metrics['loss']:.6f}, mae={test_metrics['mae']:.6f}, "
        f"psnr={test_metrics['psnr']:.4f}, ssim={test_metrics['ssim']:.6f}, "
        f"hfen={test_metrics['hfen']:.6f}, gmae={test_metrics['gradient_mae']:.6f}"
    )
    write_log_row(
        save_dir / "test_results.csv",
        {
            "loss": test_metrics["loss"],
            "mae": test_metrics["mae"],
            "psnr": test_metrics["psnr"],
            "ssim": test_metrics["ssim"],
            "foreground_mae": test_metrics["foreground_mae"],
            "foreground_psnr": test_metrics["foreground_psnr"],
            "foreground_ssim": test_metrics["foreground_ssim"],
            "gradient_mae": test_metrics["gradient_mae"],
            "hfen": test_metrics["hfen"],
            "best_ssim": best_ssim,
            "checkpoint": str(best_path if best_path.exists() else "latest_or_resumed"),
            "feature_reg_weight": feature_criterion.weight,
            "feature_reg_loss": feature_criterion.feature_loss,
        },
    )

    if args.save_test_figures:
        fig_dir = save_dir / "figures" / "test"
        save_test_comparison_figures(
            model=model,
            loader=test_loader,
            device=device,
            amp=bool(args.amp and device.type == "cuda"),
            divisor=args.divisor,
            out_dir=fig_dir,
            max_cases=args.max_test_figures,
            mc_passes=None,
            eval_crop_size=args.eval_crop_size,
        )


if __name__ == "__main__":
    main()
