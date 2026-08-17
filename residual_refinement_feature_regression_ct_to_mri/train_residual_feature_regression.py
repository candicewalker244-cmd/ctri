#!/usr/bin/env python3
"""Train CT->MRI residual refinement + medical feature regression model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import model_attnres3d_residual_feature_regression as net
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


class MedicalFeatureRegressionCriterion(torch.nn.Module):
    """Weak feature-consistency loss between decoder features."""

    def __init__(
        self,
        weight: float = 0.01,
        level_weights: Tuple[float, float, float] = (0.20, 0.30, 0.50),
        feature_loss: str = "l2",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError("feature_reg_weight must be non-negative")
        if feature_loss not in {"l1", "l2", "cosine"}:
            raise ValueError("feature_loss must be one of: l1, l2, cosine")
        if any(w < 0 for w in level_weights):
            raise ValueError("feature_reg_level_weights must be non-negative")
        if sum(level_weights) <= 0.0:
            raise ValueError("sum(feature_reg_level_weights) must be > 0")

        self.weight = float(weight)
        self.level_weights = tuple(float(v) for v in level_weights)
        self.feature_loss = str(feature_loss)
        self.eps = float(eps)
        self.last_metrics: Dict[str, float] = {
            "feature_reg_raw_loss": 0.0,
            "feature_reg_weight": self.weight,
            "feature_reg_weighted_loss": 0.0,
            "feature_reg_level_losses": "",
            "feature_reg_target_gradable": 0.0,
        }

    @staticmethod
    def _extract_features_for_target(model: torch.nn.Module, target: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract medical target features with deterministic behavior (dropout off).
        """
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

    def _feature_pair_loss(
        self,
        pred_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.feature_loss == "l1":
            return F.l1_loss(pred_feature, target_feature)
        if self.feature_loss == "cosine":
            pred_flat = pred_feature.reshape(pred_feature.size(0), -1)
            target_flat = target_feature.reshape(target_feature.size(0), -1)
            pred_n = F.normalize(pred_flat, p=2.0, dim=1, eps=self.eps)
            target_n = F.normalize(target_flat, p=2.0, dim=1, eps=self.eps)
            return 1.0 - torch.mean((pred_n * target_n).sum(dim=1))
        return F.mse_loss(pred_feature, target_feature)

    def forward(
        self,
        pred_features: Dict[str, List[torch.Tensor]],
        target: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        if self.weight <= 0.0:
            self.last_metrics["feature_reg_raw_loss"] = 0.0
            self.last_metrics["feature_reg_weighted_loss"] = 0.0
            self.last_metrics["feature_reg_level_losses"] = ""
            self.last_metrics["feature_reg_target_gradable"] = 0.0
            return target.new_zeros(())

        target_features = self._extract_features_for_target(model, target)
        pred_dec_features = pred_features["dec"]

        level_count = min(len(pred_dec_features), len(target_features), len(self.level_weights))
        if level_count == 0:
            self.last_metrics["feature_reg_raw_loss"] = 0.0
            self.last_metrics["feature_reg_weighted_loss"] = 0.0
            self.last_metrics["feature_reg_level_losses"] = ""
            self.last_metrics["feature_reg_target_gradable"] = 0.0
            return target.new_zeros(())

        raw_losses: List[torch.Tensor] = []
        level_names: List[str] = []
        raw_vals: List[float] = []

        total_weight = 0.0
        weighted_sum = pred_dec_features[0].new_zeros(())
        for level, w in zip(range(level_count), self.level_weights):
            level_weight = float(self.level_weights[level])
            if level_weight <= 0.0:
                continue
            pred_f = pred_dec_features[level]
            target_f = target_features[level].detach()
            level_loss = self._feature_pair_loss(pred_f, target_f)
            raw_losses.append(level_loss)
            raw_vals.append(float(level_loss.detach().item()))
            level_names.append(f"l{level + 1}")
            weighted_sum = weighted_sum + level_weight * level_loss
            total_weight += level_weight

        if total_weight <= 0.0:
            raw_loss = pred_dec_features[0].new_zeros(())
        else:
            raw_loss = weighted_sum / total_weight

        weighted_loss = self.weight * raw_loss

        self.last_metrics.update(
            {
                "feature_reg_raw_loss": float(raw_loss.detach().item()),
                "feature_reg_weight": self.weight,
                "feature_reg_weighted_loss": float(weighted_loss.detach().item()),
                "feature_reg_level_losses": "|".join(
                    f"{name}:{value:.6f}"
                    for name, value in zip(level_names, raw_vals)
                ),
                "feature_reg_target_gradable": 0.0,
            }
        )
        return weighted_loss


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D CT->MRI residual refinement + feature regression."
    )

    default_data_dir = str(Path(__file__).resolve().parent.parent / "data" / "dataset")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Dataset root directory containing train/val/test subfolders.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        metavar=("D", "H", "W"),
        help="3D patch size for training.",
    )
    parser.add_argument(
        "--ct_norm",
        type=str,
        default="clip01",
        choices=["clip01", "zscore_nonzero"],
    )
    parser.add_argument(
        "--mri_norm",
        type=str,
        default="clip01",
        choices=["clip01", "zscore_nonzero"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
    )
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
        help="Feature-regression term for decoder feature maps.",
    )
    parser.add_argument(
        "--feature_reg_level_weights",
        type=float,
        nargs=3,
        default=(0.20, 0.30, 0.50),
        metavar=("L1", "L2", "L3"),
        help="Weights for encoder/decoder feature levels (d1,d2,d3 in this model).",
    )

    parser.add_argument(
        "--l1_weight",
        type=float,
        default=0.45,
        help="ReconstructionLoss L1 weight.",
    )
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.30,
        help="ReconstructionLoss MS-SSIM weight.",
    )
    parser.add_argument(
        "--edge_weight",
        type=float,
        default=0.15,
        help="ReconstructionLoss edge weight.",
    )
    parser.add_argument(
        "--frequency_weight",
        type=float,
        default=0.10,
        help="ReconstructionLoss frequency weight.",
    )
    parser.add_argument(
        "--frequency_alpha",
        type=float,
        default=1.0,
        help="Frequency loss alpha.",
    )

    default_save_dir = str(
        Path(__file__).resolve().parent / "output_residual_feature_regression"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_save_dir,
        help="Directory for checkpoints, logs, and test outputs.",
    )
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--no_medrecl",
        dest="medrecl",
        action="store_false",
        help=(
            "Keep interface-compatible flag. Med-ReCL is intentionally disabled "
            "in residual experiments."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--divisor", type=int, default=8)
    parser.add_argument("--eval_crop_size", nargs="+", default="150,150,150")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--run_test_only", action="store_true")
    parser.add_argument(
        "--no_test_figures",
        dest="save_test_figures",
        action="store_false",
    )
    parser.add_argument("--max_test_figures", type=int, default=0)
    parser.set_defaults(save_test_figures=True)
    parser.set_defaults(medrecl=True)

    return parser


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, switching to CPU.")
        return torch.device("cpu")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloaders(args, device: torch.device):
    data_dir = Path(args.data_dir)
    train_cases = discover_cases(data_dir / "train")
    val_cases = discover_cases(data_dir / "val")
    test_cases = discover_cases(data_dir / "test")

    validate_case_splits(
        {"train": train_cases, "val": val_cases, "test": test_cases}
    )

    pin_memory = device.type == "cuda"
    common_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    train_dataset = CTMRIDataset(
        train_cases,
        patch_size=args.patch_size,
        training=True,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )
    val_dataset = CTMRIDataset(
        val_cases,
        patch_size=args.patch_size,
        training=False,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )
    test_dataset = CTMRIDataset(
        test_cases,
        patch_size=args.patch_size,
        training=False,
        ct_norm=args.ct_norm,
        mri_norm=args.mri_norm,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)


def _build_model(args) -> torch.nn.Module:
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


def _build_criterion(args) -> torch.nn.Module:
    return ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        frequency_weight=args.frequency_weight,
        frequency_alpha=args.frequency_alpha,
    )


def _build_feature_reg_criterion(args):
    return MedicalFeatureRegressionCriterion(
        weight=args.feature_reg_weight,
        level_weights=tuple(args.feature_reg_level_weights),
        feature_loss=args.feature_reg_loss,
    )


def _train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    criterion: torch.nn.Module,
    feature_criterion: MedicalFeatureRegressionCriterion,
    device: torch.device,
    scaler: GradScaler,
    amp: bool,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    model.train()
    from torch.amp import autocast
    from model_attnres3d_gtmedreclpp import SSIM3D, compute_mae, compute_psnr

    running_loss = 0.0
    running_rec_loss = 0.0
    running_feature_raw = 0.0
    running_feature_weighted = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_grad_norm = 0.0
    grad_norm_batches = 0
    num_batches = 0

    rec_component_names = (
        "l1",
        "ms_ssim",
        "edge",
        "frequency",
        "weighted_l1",
        "weighted_ms_ssim",
        "weighted_edge",
        "weighted_frequency",
    )
    running_rec_components = {name: 0.0 for name in rec_component_names}
    ssim_fn = SSIM3D(channels=1).to(device)

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp):
            pred, feature_dict = model.forward_with_features(source)
            rec_loss = criterion(pred, target)
            feature_loss = feature_criterion(
                pred_features=feature_dict,
                target=target,
                model=model,
            )
            loss = rec_loss + feature_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: total={loss.item()}, rec={rec_loss.item()}, "
                f"feature={feature_loss.item()}"
            )

        optimizer_step_succeeded = False
        if scaler is not None and amp:
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite grad norm during training.")
            new_scale = scaler.get_scale()
            if new_scale >= old_scale:
                optimizer_step_succeeded = True
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite grad norm during training.")
            optimizer.step()
            optimizer_step_succeeded = True

        if optimizer_step_succeeded:
            running_grad_norm += float(grad_norm.detach().float().item())
            grad_norm_batches += 1

        with torch.no_grad():
            pred_clamped = torch.clamp(pred, 0.0, 1.0)
            mae = compute_mae(pred_clamped, target)
            psnr = compute_psnr(pred_clamped, target)
            ssim = ssim_fn(pred_clamped, target)

        running_loss += loss.item()
        running_rec_loss += rec_loss.item()
        running_feature_raw += float(feature_criterion.last_metrics["feature_reg_raw_loss"])
        running_feature_weighted += float(feature_criterion.last_metrics["feature_reg_weighted_loss"])
        running_mae += mae.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()
        batch_rec_components = getattr(criterion, "last_components", {})
        for name in rec_component_names:
            running_rec_components[name] += float(batch_rec_components.get(name, 0.0))
        num_batches += 1

    denom = max(1, num_batches)
    results = {
        "loss": running_loss / denom,
        "rec_loss": running_rec_loss / denom,
        "feature_raw_loss": running_feature_raw / denom,
        "feature_weighted_loss": running_feature_weighted / denom,
        "mae": running_mae / denom,
        "psnr": running_psnr / denom,
        "ssim": running_ssim / denom,
        "grad_norm": running_grad_norm / max(1, grad_norm_batches),
    }
    for name in rec_component_names:
        results[f"rec_{name}"] = running_rec_components[name] / denom
    return results


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

    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {save_dir}")

    train_loader, val_loader, test_loader, train_len, val_len, test_len = _build_dataloaders(
        args,
        device,
    )
    print(f"Samples -> train/val/test: {train_len}/{val_len}/{test_len}")

    model = _build_model(args).to(device)
    criterion = _build_criterion(args).to(device)
    feature_criterion = _build_feature_reg_criterion(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metrics = ckpt.get("best_metrics", {})
        best_ssim = float(best_metrics.get("ssim", ckpt.get("best_metric", best_ssim)))
        print(f"Resume from {resume_path}, epoch={start_epoch - 1}, best_ssim={best_ssim:.6f}")

    log_csv = save_dir / "log.csv"

    if args.run_test_only and not args.resume and not best_path.exists():
        raise ValueError(
            "run_test_only requires --resume or existing checkpoints/best.pth in "
            f"{save_dir}"
        )

    if not args.run_test_only:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_metrics = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                feature_criterion=feature_criterion,
                device=device,
                scaler=scaler,
                amp=bool(args.amp and device.type == "cuda"),
                max_grad_norm=args.max_grad_norm,
            )
            print(
                f"Train: loss={train_metrics['loss']:.6f}, "
                f"rec={train_metrics['rec_loss']:.6f}, "
                f"feat={train_metrics['feature_weighted_loss']:.6f}, "
                f"mae={train_metrics['mae']:.6f}, "
                f"psnr={train_metrics['psnr']:.4f}, "
                f"ssim={train_metrics['ssim']:.6f}"
            )

            if scheduler is not None:
                scheduler.step()

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp=bool(args.amp and device.type == "cuda"),
                divisor=args.divisor,
                eval_crop_size=args.eval_crop_size,
            )
            print(
                f"Val:   loss={val_metrics['loss']:.6f}, "
                f"mae={val_metrics['mae']:.6f}, "
                f"psnr={val_metrics['psnr']:.4f}, "
                f"ssim={val_metrics['ssim']:.6f}, "
                f"hfen={val_metrics['hfen']:.6f}, "
                f"gmae={val_metrics['gradient_mae']:.6f}"
            )

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_rec_loss": train_metrics["rec_loss"],
                "train_rec_total": train_metrics["rec_loss"],
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
                "val_rec_total": val_metrics["loss"],
                "feature_reg_weight": feature_criterion.weight,
                "feature_reg_loss": feature_criterion.feature_loss,
                "feature_reg_level_weights": " ".join(
                    f"{w:.4f}" for w in feature_criterion.level_weights
                ),
            }
            for name in (
                "l1",
                "ms_ssim",
                "edge",
                "frequency",
                "weighted_l1",
                "weighted_ms_ssim",
                "weighted_edge",
                "weighted_frequency",
            ):
                row[f"train_rec_{name}"] = train_metrics[f"rec_{name}"]
                row[f"val_rec_{name}"] = val_metrics[f"rec_{name}"]

            write_log_row(log_csv, row)

            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    best_ssim,
                    args,
                    scheduler=scheduler,
                    scaler=scaler,
                    metric_name="ssim",
                    best_metrics={
                        "ssim": best_ssim,
                        "psnr": val_metrics["psnr"],
                        "hfen": val_metrics["hfen"],
                    },
                )
                print(f"  saved best checkpoint -> {best_path}")

            latest_path = checkpoint_dir / "latest.pth"
            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    epoch,
                    best_ssim,
                    args,
                    scheduler=scheduler,
                    scaler=scaler,
                    metric_name="ssim",
                    best_metrics={
                        "ssim": best_ssim,
                        "psnr": val_metrics["psnr"],
                        "hfen": val_metrics["hfen"],
                    },
                )

    if args.skip_test:
        print("skip_test enabled; stop before evaluation.")
        return

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint for test: {best_path}")
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded resumed checkpoint for test: {args.resume}")
    elif run_model_path := next(checkpoint_dir.glob("latest.pth"), None):
        ckpt = torch.load(run_model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded latest checkpoint for test: {run_model_path}")

    print("Running final test evaluation...")
    test_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=bool(args.amp and device.type == "cuda"),
        divisor=args.divisor,
        eval_crop_size=args.eval_crop_size,
    )
    print(
        f"Test: loss={test_metrics['loss']:.6f}, "
        f"mae={test_metrics['mae']:.6f}, "
        f"psnr={test_metrics['psnr']:.4f}, "
        f"ssim={test_metrics['ssim']:.6f}, "
        f"hfen={test_metrics['hfen']:.6f}, "
        f"gmae={test_metrics['gradient_mae']:.6f}"
    )

    test_row = {
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
        "feature_reg_weight": feature_criterion.weight,
        "feature_reg_loss": feature_criterion.feature_loss,
        "checkpoint": str(best_path if best_path.exists() else "latest_or_resumed"),
    }
    write_log_row(save_dir / "test_results.csv", test_row)

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
