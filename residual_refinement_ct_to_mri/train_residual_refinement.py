#!/usr/bin/env python3
"""Train CT->MRI residual refinement model in a dedicated experiment folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
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
train_one_epoch = net.train_one_epoch
validate_one_epoch = net.validate_one_epoch
save_checkpoint = net.save_checkpoint
write_log_row = net.write_log_row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D CT->MRI residual refinement training")

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

    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--no_refiner", action="store_true")
    parser.add_argument("--refiner_channels", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=0.10)

    parser.add_argument(
        "--l1_weight", type=float, default=0.45, help="ReconstructionLoss l1 weight."
    )
    parser.add_argument(
        "--ssim_weight", type=float, default=0.30, help="ReconstructionLoss ms-ssim weight."
    )
    parser.add_argument(
        "--edge_weight", type=float, default=0.15, help="ReconstructionLoss edge loss weight."
    )
    parser.add_argument(
        "--frequency_weight",
        type=float,
        default=0.10,
        help="ReconstructionLoss frequency loss weight.",
    )
    parser.add_argument(
        "--frequency_alpha", type=float, default=1.0, help="Frequency loss alpha."
    )

    default_save_dir = str(Path(__file__).resolve().parent / "output_residual_refinement")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_save_dir,
        help="Directory for checkpoints, logs, and test outputs.",
    )
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
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
    parser.add_argument(
        "--no_medrecl",
        dest="medrecl",
        action="store_false",
        help=(
            "Keep interface-compatible flag; Med-ReCL is already disabled in "
            "residual experiments."
        ),
    )
    parser.add_argument("--no_test_figures", dest="save_test_figures", action="store_false")
    parser.add_argument("--max_test_figures", type=int, default=0)
    parser.add_argument("--run_test_only", action="store_true")
    parser.set_defaults(save_test_figures=True)
    parser.set_defaults(medrecl=True)

    return parser


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, switching to CPU.")
        return torch.device("cpu")
    return torch.device(requested if requested else ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_cases(data_dir: Path, split: str):
    path = data_dir / split
    cases = discover_cases(path)
    return cases


def _build_dataloaders(args, device: torch.device):
    data_dir = Path(args.data_dir)
    train_cases = _load_cases(data_dir, "train")
    val_cases = _load_cases(data_dir, "val")
    test_cases = _load_cases(data_dir, "test")

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
        use_refiner=(not args.no_refiner),
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
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {save_dir}")

    train_loader, val_loader, test_loader, train_len, val_len, test_len = _build_dataloaders(
        args, device
    )
    print(f"Samples -> train/val/test: {train_len}/{val_len}/{test_len}")

    model = _build_model(args).to(device)
    criterion = _build_criterion(args).to(device)

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
        if ckpt.get("scaler") is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metrics = ckpt.get("best_metrics", {})
        best_ssim = float(best_metrics.get("ssim", ckpt.get("best_metric", best_ssim)))
        print(f"Resume from {resume_path}, epoch={start_epoch - 1}, best_ssim={best_ssim:.6f}")

    log_csv = save_dir / "log.csv"
    if args.run_test_only and not args.resume and not (save_dir / "checkpoints" / "best.pth").exists():
        raise ValueError(
            "run_test_only requires either --resume or an existing checkpoints/best.pth "
            f"in {save_dir}"
        )

    if not args.run_test_only:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp=bool(args.amp and device.type == "cuda"),
                max_grad_norm=args.max_grad_norm,
            )
            print(
                f"Train: loss={train_metrics['loss']:.6f}, "
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
                "train_mae": train_metrics["mae"],
                "train_psnr": train_metrics["psnr"],
                "train_ssim": train_metrics["ssim"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_foreground_mae": val_metrics["foreground_mae"],
                "val_foreground_psnr": val_metrics["foreground_psnr"],
                "val_foreground_ssim": val_metrics["foreground_ssim"],
                "val_gradient_mae": val_metrics["gradient_mae"],
                "val_hfen": val_metrics["hfen"],
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

            row["train_rec_total"] = train_metrics["rec_loss"]
            row["val_rec_total"] = val_metrics["loss"]

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
                    best_metrics={"ssim": best_ssim, "psnr": val_metrics["psnr"], "hfen": val_metrics["hfen"]},
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
                    best_metrics={"ssim": best_ssim, "psnr": val_metrics["psnr"], "hfen": val_metrics["hfen"]},
                )

    if args.skip_test:
        print("skip_test enabled; stop before evaluation.")
        return

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint for test: {best_path}")
    elif args.resume:
        # If no best checkpoint yet, still test the explicitly resumed model.
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        print(f"Loaded resumed checkpoint for test: {args.resume}")

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
