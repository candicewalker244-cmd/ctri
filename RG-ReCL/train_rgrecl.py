#!/usr/bin/env python3
"""
train_rgrecl.py

E0 + Region-aware PatchNCE (RG-ReCL) 训练入口。

这份脚本只负责“实验流程编排”，真正的网络结构、数据集、损失函数、
矩传播不确定性计算和 MC Dropout 对照都定义在
model_attnres3d_rgrecl.py 中。

当前实验主线：
1. 训练阶段：用普通 forward 训练 CT->MRI 重建模型，dropout 按训练模式随机失活。
2. 验证阶段：每个 epoch 后用普通 forward 验证重建质量，model.eval()，dropout 关闭，
   记录 MAE / PSNR / SSIM / HFEN / Gradient MAE，并分别保存最优断点。
3. 默认测试阶段：加载 best.pth，在独立 test 集上输出重建指标和可视化图；
   测试图默认包含本文主方法的 Moment propagation variance map。
4. MC 对照阶段：默认运行 MC Dropout 200 次，得到 MC variance map，
   与 Moment variance map 做对比；如需关闭可加 --no_eval_mc_dropout_compare。
"""

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# matplotlib 默认可能尝试打开 GUI 窗口；服务器/远程训练环境通常没有显示器。
# Agg 后端只负责把图保存成 png 文件，不弹窗，适合训练脚本自动保存测试图。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 标准 import 保留 IDE 跳转、补全和静态检查能力；不要再用 exec 动态加载模型文件。
import model_attnres3d_rgrecl as net


set_seed = net.set_seed
ensure_dir = net.ensure_dir
discover_cases = net.discover_cases
validate_case_splits = net.validate_case_splits
pad_tensor_to_divisible = net.pad_tensor_to_divisible
unpad_tensor = net.unpad_tensor
CTMRIDataset = net.CTMRIDataset
AttnResCTtoMRI = net.AttnResCTtoMRI
ReconstructionLoss = net.ReconstructionLoss
RegionPatchNCEConfig = net.RegionPatchNCEConfig
RegionPatchNCELoss = net.RegionPatchNCELoss
train_one_epoch = net.train_one_epoch
validate_one_epoch = net.validate_one_epoch
save_checkpoint = net.save_checkpoint
write_log_row = net.write_log_row
validate_with_uncertainty = net.validate_with_uncertainty
fit_moment_variance_scale = net.fit_moment_variance_scale


def _case_id_from_batch(batch, fallback: str) -> str:
    """从 DataLoader 的 batch 中取出病例名，用作保存图片/NIfTI 文件名。"""
    case_id = batch.get("case_id", fallback)
    if isinstance(case_id, (list, tuple)):
        case_id = case_id[0]
    return str(case_id).replace("/", "_").replace("\\", "_")


def _middle_slice(volume: torch.Tensor):
    """
    取 3D 体数据的中间层切片用于 PNG 可视化。

    注意：训练和评估仍然使用完整 3D 体数据；这里只是为了把 3D 结果画成
    人能快速检查的 2D 图。真正的指标不是从这一张切片算的。
    """
    arr = volume.detach().float().cpu().squeeze().numpy()
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 2D/3D volume for visualization, got shape {arr.shape}")


def _imshow_panel(ax, image, title: str, cmap: str = "gray", vmin=None, vmax=None):
    """统一绘制单个图像面板，关闭坐标轴，让保存的测试图更干净。"""
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _quantile_float(values: torch.Tensor, q: float):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.detach().float(), q).item())


@torch.no_grad()
def save_precomputed_uncertainty_comparison(
    *,
    case_id: str,
    source: torch.Tensor,
    target: torch.Tensor,
    pred_moment: torch.Tensor,
    uncertainty_moment_var: torch.Tensor,
    uncertainty_moment_calibrated_var: torch.Tensor,
    pred_mc: torch.Tensor,
    uncertainty_mc_var: torch.Tensor,
    mc_passes: int,
    out_dir: Path,
    moment_calibration_scale: float = 1.0,
):
    """Render the exact Moment/MC tensors used by uncertainty_results.csv."""
    ensure_dir(out_dir)
    source_c = torch.clamp(source, 0.0, 1.0)
    target_c = torch.clamp(target, 0.0, 1.0)
    pred_moment_c = torch.clamp(pred_moment, 0.0, 1.0)
    pred_mc_c = torch.clamp(pred_mc, 0.0, 1.0)
    moment_var_c = uncertainty_moment_var.clamp_min(0.0)
    mc_var_c = uncertainty_mc_var.clamp_min(0.0)
    moment_error = torch.abs(pred_moment_c - target_c)
    mc_error = torch.abs(pred_mc_c - target_c)
    reconstruction_difference = torch.abs(pred_moment_c - pred_mc_c)
    variance_difference = torch.abs(moment_var_c - mc_var_c)
    brain_mask = target_c > 0.01

    error_values = [
        values for values in (moment_error[brain_mask], mc_error[brain_mask])
        if values.numel() > 0
    ]
    variance_values = [
        values for values in (moment_var_c[brain_mask], mc_var_c[brain_mask])
        if values.numel() > 0
    ]
    error_vmax = _quantile_float(torch.cat(error_values), 0.99) if error_values else None
    variance_vmax = (
        _quantile_float(torch.cat(variance_values), 0.99)
        if variance_values else None
    )
    reconstruction_diff_vmax = _quantile_float(
        reconstruction_difference[brain_mask], 0.99
    )
    variance_diff_vmax = _quantile_float(variance_difference[brain_mask], 0.99)
    if error_vmax is not None and error_vmax <= 1e-12:
        error_vmax = None
    if variance_vmax is not None and variance_vmax <= 1e-12:
        variance_vmax = None
    if reconstruction_diff_vmax is not None and reconstruction_diff_vmax <= 1e-12:
        reconstruction_diff_vmax = None
    if variance_diff_vmax is not None and variance_diff_vmax <= 1e-12:
        variance_diff_vmax = None

    # The figure compares the raw analytical Moment variance with raw MC200.
    # Calibration is evaluated separately against validation/test squared error.
    moment_title = "Moment (raw)"
    panel_rows = [
        [
            (_middle_slice(source_c[0, 0]), "Input CT", "gray", 0.0, 1.0),
            (_middle_slice(target_c[0, 0]), "GT MRI", "gray", 0.0, 1.0),
            (_middle_slice(pred_moment_c[0, 0]), f"{moment_title} pred MRI", "gray", 0.0, 1.0),
            (_middle_slice(moment_error[0, 0]), "|Moment-GT|", "magma", 0.0, error_vmax),
            (_middle_slice(moment_var_c[0, 0]), f"{moment_title} variance", "inferno", 0.0, variance_vmax),
            (
                _middle_slice(reconstruction_difference[0, 0]),
                "|Moment-MC MRI|",
                "magma",
                0.0,
                reconstruction_diff_vmax,
            ),
        ],
        [
            (_middle_slice(source_c[0, 0]), "Input CT", "gray", 0.0, 1.0),
            (_middle_slice(target_c[0, 0]), "GT MRI", "gray", 0.0, 1.0),
            (_middle_slice(pred_mc_c[0, 0]), f"MC mean MRI ({mc_passes})", "gray", 0.0, 1.0),
            (_middle_slice(mc_error[0, 0]), "|MC-GT|", "magma", 0.0, error_vmax),
            (_middle_slice(mc_var_c[0, 0]), f"MC variance ({mc_passes})", "inferno", 0.0, variance_vmax),
            (
                _middle_slice(variance_difference[0, 0]),
                "|Moment-MC variance|",
                "magma",
                0.0,
                variance_diff_vmax,
            ),
        ],
    ]
    fig, axes = plt.subplots(2, 6, figsize=(19.2, 6.8), squeeze=False)
    for row_index, row in enumerate(panel_rows):
        for column_index, (image, title, cmap, vmin, vmax) in enumerate(row):
            _imshow_panel(
                axes[row_index, column_index],
                image,
                title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
    axes[0, 0].text(
        -0.08, 0.5, "Moment", rotation=90, va="center", ha="right",
        transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold",
    )
    axes[1, 0].text(
        -0.08, 0.5, f"MC Dropout {mc_passes}", rotation=90,
        va="center", ha="right", transform=axes[1, 0].transAxes,
        fontsize=11, fontweight="bold",
    )
    fig.suptitle(case_id, fontsize=12)
    fig.tight_layout(rect=(0.035, 0.02, 0.995, 0.95), pad=1.2, h_pad=2.8, w_pad=0.8)
    fig.savefig(out_dir / f"{case_id}_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_test_comparison_figures(
    model,
    loader,
    device,
    amp: bool,
    divisor: int,
    out_dir: Path,
    max_cases: int = 0,
    mc_passes=None,
    eval_crop_size=None,
):
    """
    保存测试集可视化对比图。

    默认模式（mc_passes=None）：
        保存 CT / GT MRI / Moment pred MRI / Error / Moment variance。
        这里的 Moment variance 是本文主方法：一次 forward_mu_var_cov() 得到的
        认知不确定性方差图，不需要多次采样。

    MC 对照模式（mc_passes=200 等整数）：
        在默认面板基础上额外保存 MC mean MRI、MC error、MC variance 和 Moment-MC 差异。
        这个模式只在 --eval_mc_dropout_compare 打开时使用，用来检查矩传播
        方差图和 MC Dropout 参考方差图是否接近。

    重要区别：
        - Moment variance：主方法，单次矩传播，速度快，默认测试图就有。
        - MC variance：参考对照，多次随机 forward，耗时大，只在需要对比时跑。
    """
    ensure_dir(out_dir)
    model.eval()
    saved = 0

    for batch_idx, batch in enumerate(loader):
        if max_cases > 0 and saved >= max_cases:
            break

        # source 是输入 CT，target 是真实 MRI 标签；形状通常为 [B, 1, D, H, W]。
        # non_blocking=True 在 pin_memory=True 且使用 GPU 时能减少 CPU->GPU 等待时间。
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source, target = net.crop_pair_around_foreground(source, target, eval_crop_size)
        case_id = _case_id_from_batch(batch, f"case_{batch_idx:04d}")

        # 网络经历多次下采样/上采样，D/H/W 最好能被 divisor 整除。
        # pad 只是在边缘补 0 以满足网络尺寸要求，推理结束后会 unpad 回原始大小。
        source_padded, pad = pad_tensor_to_divisible(source, divisor=divisor)
        with autocast(device_type=device.type, enabled=amp):
            # 主方法：单次矩传播，输出重建图 pred 和 Moment 方差图 moment_var。
            # 这里不更新权重、不 backward，只使用已经训练好的 best.pth 权重做推理。
            pred, moment_var = model.forward_mu_var_cov(source_padded)
            moment_var = unpad_tensor(moment_var, pad)
            pred = unpad_tensor(pred, pad)

            mc_pred = None
            mc_var = None
            if mc_passes is not None:
                # 可选对照：MC Dropout 只打开 dropout 随机性，多次采样后求方差。
                # pred_mc 在这里不用画，图中统一展示主方法 pred，MC 只提供方差参考。
                mc_pred, mc_var = model.mc_dropout_inference(
                    source_padded, num_passes=mc_passes
                )
                mc_pred = unpad_tensor(mc_pred, pad)
                mc_var = unpad_tensor(mc_var, pad)

        # clamp 到 0~1 只是为了可视化和误差图稳定，不改变原始保存的模型权重。
        source_c = torch.clamp(source, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        pred_c = torch.clamp(pred, 0.0, 1.0)
        error = torch.abs(pred_c - target_c)
        mc_pred_c = torch.clamp(mc_pred, 0.0, 1.0) if mc_pred is not None else None
        mc_error = torch.abs(mc_pred_c - target_c) if mc_pred_c is not None else None
        brain_mask = target_c > 0.01
        error_values = [error[brain_mask]]
        if mc_error is not None:
            error_values.append(mc_error[brain_mask])
        valid_error_values = [values for values in error_values if values.numel() > 0]
        error_vmax = (
            _quantile_float(torch.cat(valid_error_values), 0.99)
            if valid_error_values
            else None
        )
        if error_vmax is not None and error_vmax <= 1e-12:
            error_vmax = None

        if mc_var is not None:
            moment_var_c = moment_var.clamp_min(0.0)
            mc_var_c = mc_var.clamp_min(0.0)
            variance_values = [
                values for values in (
                    moment_var_c[brain_mask],
                    mc_var_c[brain_mask],
                )
                if values.numel() > 0
            ]
            variance_vmax = (
                _quantile_float(torch.cat(variance_values), 0.99)
                if variance_values
                else None
            )
            if variance_vmax is not None and variance_vmax <= 1e-12:
                variance_vmax = None

            recon_difference = torch.abs(pred_c - mc_pred_c)
            variance_difference = torch.abs(moment_var_c - mc_var_c)
            recon_diff_values = recon_difference[brain_mask]
            variance_diff_values = variance_difference[brain_mask]
            recon_diff_vmax = _quantile_float(recon_diff_values, 0.99)
            variance_diff_vmax = _quantile_float(variance_diff_values, 0.99)
            if recon_diff_vmax is not None and recon_diff_vmax <= 1e-12:
                recon_diff_vmax = None
            if variance_diff_vmax is not None and variance_diff_vmax <= 1e-12:
                variance_diff_vmax = None

            panel_rows = [
                [
                    (_middle_slice(source_c[0, 0]), "Input CT", "gray", 0.0, 1.0),
                    (_middle_slice(target_c[0, 0]), "GT MRI", "gray", 0.0, 1.0),
                    (_middle_slice(pred_c[0, 0]), "Moment pred MRI", "gray", 0.0, 1.0),
                    (_middle_slice(error[0, 0]), "|Moment-GT|", "magma", 0.0, error_vmax),
                    (
                        _middle_slice(moment_var_c[0, 0]),
                        "Moment variance",
                        "inferno",
                        0.0,
                        variance_vmax,
                    ),
                    (
                        _middle_slice(recon_difference[0, 0]),
                        "|Moment-MC MRI|",
                        "magma",
                        0.0,
                        recon_diff_vmax,
                    ),
                ],
                [
                    (_middle_slice(source_c[0, 0]), "Input CT", "gray", 0.0, 1.0),
                    (_middle_slice(target_c[0, 0]), "GT MRI", "gray", 0.0, 1.0),
                    (
                        _middle_slice(mc_pred_c[0, 0]),
                        f"MC mean MRI ({mc_passes})",
                        "gray",
                        0.0,
                        1.0,
                    ),
                    (
                        _middle_slice(mc_error[0, 0]),
                        "|MC-GT|",
                        "magma",
                        0.0,
                        error_vmax,
                    ),
                    (
                        _middle_slice(mc_var_c[0, 0]),
                        f"MC variance ({mc_passes})",
                        "inferno",
                        0.0,
                        variance_vmax,
                    ),
                    (
                        _middle_slice(variance_difference[0, 0]),
                        "|Moment-MC variance|",
                        "magma",
                        0.0,
                        variance_diff_vmax,
                    ),
                ],
            ]
        else:
            moment_values = moment_var.clamp_min(0.0)[brain_mask]
            variance_vmax = _quantile_float(moment_values, 0.99)
            if variance_vmax is not None and variance_vmax <= 1e-12:
                variance_vmax = None
            panel_rows = [[
                (_middle_slice(source_c[0, 0]), "Input CT", "gray", 0.0, 1.0),
                (_middle_slice(target_c[0, 0]), "GT MRI", "gray", 0.0, 1.0),
                (_middle_slice(pred_c[0, 0]), "Moment pred MRI", "gray", 0.0, 1.0),
                (_middle_slice(error[0, 0]), "|Moment-GT|", "magma", 0.0, error_vmax),
                (
                    _middle_slice(moment_var[0, 0].clamp_min(0.0)),
                    "Moment variance",
                    "inferno",
                    0.0,
                    variance_vmax,
                ),
            ]]

        num_rows = len(panel_rows)
        num_columns = max(len(row) for row in panel_rows)
        fig, axes = plt.subplots(
            num_rows,
            num_columns,
            figsize=(3.2 * num_columns, 3.4 * num_rows),
            squeeze=False,
        )
        for row_index, row in enumerate(panel_rows):
            for column_index, (image, title, cmap, vmin, vmax) in enumerate(row):
                _imshow_panel(
                    axes[row_index, column_index],
                    image,
                    title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            for column_index in range(len(row), num_columns):
                axes[row_index, column_index].axis("off")
        if num_rows == 2:
            axes[0, 0].text(
                -0.08, 0.5, "Moment", rotation=90, va="center", ha="right",
                transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold",
            )
            axes[1, 0].text(
                -0.08, 0.5, f"MC Dropout {mc_passes}", rotation=90,
                va="center", ha="right", transform=axes[1, 0].transAxes,
                fontsize=11, fontweight="bold",
            )
        fig.suptitle(case_id, fontsize=12)
        fig.tight_layout(
            rect=(0.035, 0.02, 0.995, 0.95),
            pad=1.2,
            h_pad=2.8,
            w_pad=0.8,
        )
        fig.savefig(out_dir / f"{case_id}_compare.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print(f"测试对比图已保存: {out_dir} ({saved} cases)")



# 程序主入口函数：从这里开始按“参数解析 -> 数据 -> 模型 -> 训练 -> 验证 -> 测试”顺序执行。
def main():
    # argparse 负责从命令行读取配置。
    # 例如 python train.py --lr 5e-5 --batch_size 1
    # 这样可以不改代码就调整实验超参数，便于多组 SCI 实验复现。
    parser = argparse.ArgumentParser(description="3D CT→MRI AttnRes U-Net training")

    # ========== 数据相关参数 ==========
    # 数据集根目录要求包含 train/val/test 三个子目录。
    # train：参与反向传播，更新权重。
    # val：每个 epoch 后评估，决定是否保存 best.pth。
    # test：训练全部结束后只跑一次，报告论文最终指标和保存测试图。
    default_data_dir = str(Path(__file__).parent / "data" / "dataset")
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help="数据集根目录，内部包含train/val/test三个子文件夹，默认当前项目下的data/dataset")
    # 训练时随机裁剪 3D patch，避免整幅 3D 体数据直接塞入显卡导致 OOM。
    # 参数顺序固定为 D H W，也就是深度、高度、宽度。
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96],
                        metavar=("D", "H", "W"),
                        help="训练随机截取3D小块尺寸，默认96 96 96")
    # CT 和 MRI 分开设置归一化，便于后续对比 clip01 与 zscore_nonzero。
    # 当前主线建议先用 clip01，因为 SSIM/PSNR 和输出 sigmoid 都默认围绕 0~1 范围。
    parser.add_argument("--ct_norm", type=str, default="clip01",
                        choices=["clip01", "zscore_nonzero"],
                        help="CT预处理归一化方案，默认clip01")
    parser.add_argument("--mri_norm", type=str, default="clip01",
                        choices=["clip01", "zscore_nonzero"],
                        help="MRI预处理归一化方案，默认clip01")

    # ========== 网络模型参数 ==========
    # base_channels 控制模型宽度。数值越大，表达能力越强，但显存和训练时间也越高。
    parser.add_argument("--base_channels", type=int, default=32,
                        help="网络首层通道数量，默认32")
    # bottleneck_blocks 控制瓶颈层 AttnRes 块数量，是深层语义建模能力的主要来源之一。
    parser.add_argument("--bottleneck_blocks", type=int, default=6,
                        help="瓶颈层AttnRes块数量，默认6")
    # dropout 是训练时随机丢弃中间特征的比例。
    # 这里不是丢 CT 原图像素，而是丢网络内部特征。
    # 当前 MomentDrop3d 使用标准 inverted dropout：训练时 x*mask/keep，eval 时直接 x。
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout比例，0代表关闭，默认0.2")
    # sigmoid 会把重建 MRI 限制在 0~1，和 clip01 归一化及 PSNR/SSIM 计算范围一致。
    parser.add_argument("--final_activation", type=str, default="sigmoid",
                        help="输出激活函数，默认sigmoid，输出限制0~1")

    # ========== 训练超参数 ==========
    # 一个 epoch 表示训练集被完整遍历一遍。
    parser.add_argument("--epochs", type=int, default=100,
                        help="完整训练总轮数，默认100")
    # batch_size 是一次送进显卡的 patch 数量；3D 医学图像显存占用很高，默认 2。
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小，3D图像显存占用高，默认1")
    # 学习率控制每次 optimizer.step() 更新权重的幅度。
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="模型学习率，默认0.0001")
    # cosine 调度会让学习率从 lr 平滑下降到 min_lr，后期更利于稳定收敛。
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"],
                        help="学习率调度器，默认cosine；none表示固定学习率")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="CosineAnnealingLR最低学习率，默认1e-6")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="反缩放后的全局梯度裁剪阈值，默认5.0；用于稳定AMP训练")
    # 重建损失 = 0.45 L1 + 0.30 MS-SSIM + 0.15 Edge + 0.10 Focal Frequency。
    # 频域项保持为弱约束，用于补充细节，不取代像素和结构监督。
    parser.add_argument("--l1_weight", type=float, default=0.45,
                        help="L1强度损失权重，默认0.45")
    parser.add_argument("--ssim_weight", type=float, default=0.30,
                        help="多尺度3D SSIM损失权重，默认0.30")
    parser.add_argument("--edge_weight", type=float, default=0.15,
                        help="3D梯度差损失权重，默认0.15")
    parser.add_argument("--frequency_weight", type=float, default=0.10,
                        help="3D Focal Frequency损失权重，默认0.10")
    parser.add_argument("--frequency_alpha", type=float, default=1.0,
                        help="Focal Frequency难频率聚焦指数，默认1.0")
    # Windows 下多进程 DataLoader 容易因为 spawn/路径/交互环境出问题，所以默认 0 最稳。
    parser.add_argument("--num_workers", type=int, default=0,
                        help="数据加载线程，Windows固定填0")

    # ========== 硬件设备参数 ==========
    # device 控制训练放在 GPU 还是 CPU。
    # 如果用户写 cuda 但机器没有可用 NVIDIA GPU，下面会自动切回 CPU 并关闭 AMP。
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备，有显卡用cuda，无显卡自动切cpu")
    # AMP 会在合适的算子中使用半精度，减少显存并加速；GradScaler 负责防止梯度下溢。
    parser.add_argument("--amp", action="store_true", default=True,
                        help="开启混合精度训练，默认开启")
    # --no_amp 与 --amp 共用同一个 args.amp 变量；加上该参数就强制全精度 FP32。
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="添加该参数则关闭混合精度")
    # divisor 用于 pad 输入体数据，使 D/H/W 可被网络下采样倍数整除。
    # 推理结束后会 unpad 回原尺寸，不会改变最终输出尺寸。
    parser.add_argument("--divisor", type=int, default=8,
                        help="推理图像尺寸需整除的数值，默认8")

    # ========== 日志与模型保存参数 ==========
    # save_dir 下会保存日志、模型权重、测试指标、测试可视化图。
    parser.add_argument("--save_dir", type=str, default="./output",
                        help="训练输出文件保存目录，默认output文件夹")
    # save_every 保存周期性断点，防止长时间训练中断后只能重头跑。
    parser.add_argument("--save_every", type=int, default=5,
                        help="每N轮保存一次周期断点；latest.pth仍每轮保存，默认5")
    # resume 用于继续训练，加载内容包括 model / optimizer / scheduler / scaler / epoch。
    parser.add_argument("--resume", type=str, default="",
                        help="断点续训pth路径；为空则从头训练")
    # 默认会保存测试 PNG。加 --no_test_figures 可以只要 CSV 指标，不保存图。
    parser.add_argument("--no_test_figures", dest="save_test_figures", action="store_false",
                        help="关闭测试集CT/GT/预测/误差/Moment方差图保存")
    # max_test_figures=0 表示保存全部测试病例；大于 0 时只保存前 N 个病例，适合快速检查。
    parser.add_argument("--max_test_figures", type=int, default=0,
                        help="最多保存多少个测试病例对比图，0表示全部保存")
    parser.set_defaults(save_test_figures=True)
    # 固定随机种子能提高可复现性；注意 GPU 某些算子仍可能存在极小非确定性。
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证实验可复现，默认42")

    # ========== Region-aware PatchNCE ==========
    # 训练总损失固定为 L_rec + lambda * L_region_patchnce。
    parser.add_argument("--rgrecl_lambda", type=float, default=0.05,
                        help="Region-aware PatchNCE固定权重，默认0.05")
    parser.add_argument("--rgrecl_temperature", type=float, default=0.07,
                        help="PatchNCE temperature，默认0.07")
    parser.add_argument("--rgrecl_hard_ratio", type=float, default=0.20,
                        help="按当前绝对重建误差选择的hard区域比例，默认0.20")
    parser.add_argument("--rgrecl_feature_dim", type=int, default=64,
                        help="PatchNCE投影特征维度，默认64")
    parser.add_argument("--rgrecl_num_patches", type=int, default=256,
                        help="每病例每尺度anchor patch数量，默认256")
    parser.add_argument("--rgrecl_negative_pool", type=int, default=1024,
                        help="每病例每尺度negative候选数量，默认1024")
    parser.add_argument("--no_rgrecl", dest="rgrecl", action="store_false",
                        help="关闭Region-aware PatchNCE，仅运行原始E0重建")
    parser.set_defaults(rgrecl=True)

    # ========== 测试阶段不确定性参数 ==========
    # 默认测试一定会输出本文主方法的 Moment variance 图：
    #   output/figures/test/*_compare.png
    # MC Dropout 不是主方法，它是 Moment uncertainty 的参考对照。
    # 默认额外跑 200 次随机 dropout 前向，输出 MC variance、Moment-MC 差异图和方差对比指标。
    parser.add_argument("--eval_mc_dropout_compare", dest="eval_mc_dropout_compare",
                        action="store_true", default=True,
                        help="测试阶段运行MC Dropout方差图对照；默认开启")
    parser.add_argument("--no_eval_mc_dropout_compare", dest="eval_mc_dropout_compare",
                        action="store_false",
                        help="关闭测试阶段MC Dropout 200对照，只保存Moment结果")
    parser.add_argument("--mc_passes", type=int, default=200,
                        help="MC Dropout推理次数；默认200，除非使用--no_eval_mc_dropout_compare关闭")
    parser.add_argument("--eval_crop_size", nargs="+", default="150,150,150",
                        help="validation/test crop size, e.g. 150,150,150; use full to disable")
    parser.add_argument("--save_uncertainty", action="store_true", default=False,
                        help="运行MC Dropout对照时，将单次矩传播/MC方差图和重建结果额外保存为nii文件")
    parser.add_argument("--moment_calibration", dest="moment_calibration",
                        action="store_true", default=True,
                        help="fit one non-negative Moment variance scale on validation data")
    parser.add_argument("--no_moment_calibration", dest="moment_calibration",
                        action="store_false",
                        help="disable validation-only Moment variance calibration")

    # 读取终端输入的所有参数，校验类型和 choices，最后打包成 args 对象。
    args = parser.parse_args()
    # argparse 会把 --patch_size 96 96 96 读成 list；这里转成 tuple，方便后续作为固定尺寸使用。
    args.patch_size = tuple(args.patch_size)
    args.eval_crop_size = net.parse_eval_crop_size(args.eval_crop_size)
    if args.rgrecl_lambda < 0.0:
        raise ValueError("--rgrecl_lambda must be non-negative")
    if args.rgrecl_temperature <= 0.0:
        raise ValueError("--rgrecl_temperature must be positive")
    if not 0.0 <= args.rgrecl_hard_ratio <= 1.0:
        raise ValueError("--rgrecl_hard_ratio must be in [0, 1]")
    if args.rgrecl_feature_dim <= 0:
        raise ValueError("--rgrecl_feature_dim must be positive")
    if args.rgrecl_num_patches <= 0:
        raise ValueError("--rgrecl_num_patches must be positive")
    if args.rgrecl_negative_pool <= 0:
        raise ValueError("--rgrecl_negative_pool must be positive")



    # ==================== 训练前初始化配置 ====================
    # 固定随机种子：影响随机裁剪、随机翻转、dropout mask、网络初始化等随机过程。
    set_seed(args.seed)
    # 创建输出目录：
    #   checkpoints/ 保存 best.pth 和周期断点；
    #   log.csv 保存每轮 train/val 指标；
    #   figures/test 保存默认 Moment 测试图；
    #   uncertainty/ 仅在 --save_uncertainty 时保存 nii 方差图。
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    ensure_dir(save_dir / "checkpoints")

    # 自动判断使用显卡还是 CPU。所有输入张量、模型参数、损失函数都必须放到同一个 device。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.amp:
        print("[警告] 当前实际设备不是CUDA，已关闭AMP")
    # AMP只在实际CUDA设备上开启；兼容 cuda、cuda:0 等设备写法。
    args.amp = bool(args.amp and device.type == "cuda")
    # 打印当前硬件配置，方便之后核对实验日志。
    print(f"训练设备: {device}")
    print(f"混合精度状态: {args.amp}")
    print(f"数据集路径: {args.data_dir}")
    print(f"RG-ReCL 状态: {args.rgrecl}")
    print("方法版本: E0 + Region-aware PatchNCE")
    print(
        "RG-ReCL 参数: "
        f"lambda={args.rgrecl_lambda}, "
        f"temperature={args.rgrecl_temperature}, "
        f"hard_ratio={args.rgrecl_hard_ratio}, "
        f"feature_dim={args.rgrecl_feature_dim}, "
        f"patches/negative_pool={args.rgrecl_num_patches}/"
        f"{args.rgrecl_negative_pool}"
    )
    print("Moment propagation不参与训练，仅保留为训练后不确定性评估路径")
    print(f"验证/测试裁剪: {args.eval_crop_size if args.eval_crop_size is not None else 'full'}")


    # ==================== 加载训练、验证数据集 ====================
    # discover_cases 会扫描每个 split 目录，收集配对好的 CT/MRI 病例路径。
    train_cases = discover_cases(Path(args.data_dir) / "train")
    val_cases = discover_cases(Path(args.data_dir) / "val")
    test_dir = Path(args.data_dir) / "test"
    test_cases = discover_cases(test_dir) if test_dir.exists() else []
    split_cases = {"train": train_cases, "val": val_cases}
    if test_cases:
        split_cases["test"] = test_cases
    validate_case_splits(split_cases)
    print(f"训练集病例总数: {len(train_cases)}")
    print(f"验证集病例总数: {len(val_cases)}")
    if test_cases:
        print(f"测试集病例总数: {len(test_cases)}")
    print("CT/MRI配对检查: 病例ID、shape、affine和跨集合去重均通过")

    # 训练集 training=True：
    #   随机裁剪 patch、随机翻转、强度扰动等增强只在训练阶段使用。
    #   目的：增加样本变化，降低过拟合。
    train_ds = CTMRIDataset(
        train_cases, patch_size=args.patch_size, training=True,
        ct_norm=args.ct_norm, mri_norm=args.mri_norm,
    )
    # 验证集 training=False：
    #   关闭随机增强，用稳定输入评估模型泛化效果。
    #   验证结果用于选择 best.pth，所以不能加随机扰动。
    val_ds = CTMRIDataset(
        val_cases, patch_size=args.patch_size, training=False,
        ct_norm=args.ct_norm, mri_norm=args.mri_norm,
    )

    # 训练 DataLoader：
    #   shuffle=True：每轮打乱训练样本；
    #   drop_last=True：丢掉最后不足 batch_size 的小批次，避免某些归一化/统计不稳定；
    #   pin_memory=True：使用 GPU 时加速 CPU 到 GPU 的数据传输。
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    # 验证 DataLoader：batch_size=1，按病例顺序稳定评估，不打乱。
    val_loader = DataLoader(                                       
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )


    # ==================== 初始化网络、损失函数、优化器 ====================
    # 构建 CT->MRI 主模型。
    # forward() 用于训练/普通验证；forward_mu_var_cov() 用于测试阶段 Moment 方差图；
    # mc_dropout_inference() 用于可选 MC Dropout 对照。
    model = AttnResCTtoMRI(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels,
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
        final_activation=args.final_activation,
        use_rgrecl=args.rgrecl,
        rgrecl_feature_dim=args.rgrecl_feature_dim,
    ).to(device)
    # 统计可训练参数量，便于论文/实验记录模型规模。
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 重建损失：强度 + 多尺度结构 + 边缘 + 弱频域细节。
    criterion = ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        frequency_weight=args.frequency_weight,
        frequency_alpha=args.frequency_alpha,
    ).to(device)

    rgrecl_criterion = None
    if args.rgrecl:
        rgrecl_criterion = RegionPatchNCELoss(
            RegionPatchNCEConfig(
                temperature=args.rgrecl_temperature,
                hard_ratio=args.rgrecl_hard_ratio,
                feature_dim=args.rgrecl_feature_dim,
                num_patches=args.rgrecl_num_patches,
                negative_pool_size=args.rgrecl_negative_pool,
            )
        ).to(device)
    # AdamW 根据 loss.backward() 得到的梯度更新卷积权重、归一化参数、注意力模块、
    # RG-ReCL 投影头等所有 requires_grad=True 的参数。
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
        )
    # GradScaler 只在 CUDA + AMP 时需要；CPU 或关闭 AMP 时不用创建。
    scaler = GradScaler(device.type) if args.amp else None


    # ==================== 主训练循环 ====================
    # log.csv 记录每个 epoch 的训练/验证指标，后面画收敛曲线或写论文表格都从这里取。
    csv_path = save_dir / "log.csv"
    # 同时保存结构、像素和细节最优模型；best.pth 继续兼容原来的 SSIM 主模型。
    best_ssim = -999.0
    best_psnr = -999.0
    best_hfen = float("inf")
    # best.pth 是后续 test 和不确定性推理真正加载的权重。
    best_path = save_dir / "checkpoints" / "best.pth"
    best_ssim_path = save_dir / "checkpoints" / "best_ssim.pth"
    best_psnr_path = save_dir / "checkpoints" / "best_psnr.pth"
    best_hfen_path = save_dir / "checkpoints" / "best_hfen.pth"
    start_epoch = 1
    global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\n加载断点继续训练: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer_restored = False
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                optimizer_restored = True
            except ValueError as exc:
                print(
                    "[warning] checkpoint optimizer groups do not match the frozen EMA "
                    "teacher configuration; model weights were restored, but AdamW "
                    f"state was reinitialized ({exc})."
                )
        if scheduler is not None and optimizer_restored and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and optimizer_restored and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        rng_state = ckpt.get("rng_state")
        if isinstance(rng_state, dict):
            try:
                if "python" in rng_state:
                    net.random.setstate(rng_state["python"])
                if "numpy" in rng_state:
                    net.np.random.set_state(rng_state["numpy"])
                if "torch" in rng_state:
                    torch.set_rng_state(rng_state["torch"].cpu())
                if device.type == "cuda" and "cuda" in rng_state:
                    torch.cuda.set_rng_state_all(
                        [state.cpu() for state in rng_state["cuda"]]
                    )
            except Exception as exc:
                print(f"[warning] RNG state restore failed; continuing with current RNG state ({exc}).")
        saved_best_metrics = ckpt.get("best_metrics", {})
        best_ssim = float(saved_best_metrics.get("ssim", ckpt.get("best_metric", best_ssim)))
        best_psnr = float(saved_best_metrics.get("psnr", best_psnr))
        best_hfen = float(saved_best_metrics.get("hfen", best_hfen))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", (start_epoch - 1) * len(train_loader)))
        print(f"从 epoch {start_epoch} 开始，当前最佳验证SSIM: {best_ssim:.6f}")

    # 如果是从头训练，start_epoch=1；如果 resume，则从 checkpoint 的下一轮继续。
    print(f"\n开始训练，epoch {start_epoch} 到 {args.epochs}")
    t0 = time.time()
    total_steps = max(1, args.epochs * len(train_loader))

    # 每个 epoch 的顺序固定为：
    #   1. train_one_epoch：model.train()，dropout 开，计算 loss，backward，optimizer.step() 更新权重。
    #   2. validate_one_epoch：model.eval()，dropout 关，不更新权重，只算验证指标。
    #   3. 如果验证 SSIM 更好，保存 best.pth。
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.amp,
            rgrecl_criterion=rgrecl_criterion,
            lambda_rgrecl=args.rgrecl_lambda,
            start_step=global_step,
            total_steps=total_steps,
            max_grad_norm=args.max_grad_norm,
        )
        global_step = int(train_metrics.pop("next_step"))
        optimizer_steps = int(train_metrics.pop("optimizer_steps", 0))
        skipped_optimizer_steps = int(train_metrics.pop("skipped_optimizer_steps", 0))
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, args.amp, divisor=args.divisor,
            eval_crop_size=args.eval_crop_size,
        )

        # 把本轮所有指标打包写入 CSV。这里记录的 lr 是本轮训练实际使用的学习率。
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": f"{current_lr:.8f}",
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_mae": f"{train_metrics['mae']:.6f}",
            "train_psnr": f"{train_metrics['psnr']:.4f}",
            "train_ssim": f"{train_metrics['ssim']:.6f}",
            "train_optimizer_steps": optimizer_steps,
            "train_skipped_optimizer_steps": skipped_optimizer_steps,
            "train_grad_norm": f"{train_metrics['grad_norm']:.8f}",
            "train_amp_scale": f"{train_metrics['amp_scale']:.1f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_mae": f"{val_metrics['mae']:.6f}",
            "val_psnr": f"{val_metrics['psnr']:.4f}",
            "val_ssim": f"{val_metrics['ssim']:.6f}",
            "val_foreground_mae": f"{val_metrics['foreground_mae']:.6f}",
            "val_foreground_psnr": f"{val_metrics['foreground_psnr']:.4f}",
            "val_foreground_ssim": f"{val_metrics['foreground_ssim']:.6f}",
            "val_gradient_mae": f"{val_metrics['gradient_mae']:.8f}",
            "val_hfen": f"{val_metrics['hfen']:.8f}",
        }
        for component_name in (
            "l1",
            "ms_ssim",
            "edge",
            "frequency",
            "weighted_l1",
            "weighted_ms_ssim",
            "weighted_edge",
            "weighted_frequency",
        ):
            row[f"train_rec_{component_name}"] = (
                f"{train_metrics[f'rec_{component_name}']:.8f}"
            )
            row[f"val_rec_{component_name}"] = (
                f"{val_metrics[f'rec_{component_name}']:.8f}"
            )
        if args.rgrecl:
            row["train_rec_loss"] = f"{train_metrics['rec_loss']:.8f}"
            row["train_rgrecl_loss"] = f"{train_metrics['rgrecl_loss']:.8f}"
            row["train_rgrecl_weighted_loss"] = (
                f"{train_metrics['rgrecl_weighted_loss']:.8f}"
            )
            row["train_rgrecl_hard_region_fraction"] = (
                f"{train_metrics['rgrecl_hard_region_fraction']:.8f}"
            )
            row["train_rgrecl_sampled_hard_fraction"] = (
                f"{train_metrics['rgrecl_sampled_hard_fraction']:.8f}"
            )
            row["train_rgrecl_positive_similarity"] = (
                f"{train_metrics['rgrecl_positive_similarity']:.8f}"
            )
        write_log_row(csv_path, row)

        # 根据已完成 epoch 的平均耗时估算剩余训练时间，只用于控制台提示。
        elapsed = time.time() - t0
        completed_epochs = epoch - start_epoch + 1
        eta = (elapsed / max(1, completed_epochs)) * (args.epochs - epoch)

        # 控制台打印关键指标，便于训练时实时判断是否收敛、是否过拟合。
        if args.rgrecl:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"lr:{current_lr:.2e} "
                f"train L:{train_metrics['loss']:.6f} "
                f"Rec:{train_metrics['rec_loss']:.6f} "
                f"RG:{train_metrics['rgrecl_loss']:.6f} "
                f"RGW:{train_metrics['rgrecl_weighted_loss']:.6f} "
                f"Hard/Sampled:{train_metrics['rgrecl_hard_region_fraction']:.3f}/"
                f"{train_metrics['rgrecl_sampled_hard_fraction']:.3f} "
                f"PosSim:{train_metrics['rgrecl_positive_similarity']:.3f} "
                f"Opt:{optimizer_steps} Skip:{skipped_optimizer_steps} "
                f"Grad:{train_metrics['grad_norm']:.3f} "
                f"Scale:{train_metrics['amp_scale']:.0f} "
                f"MAE:{train_metrics['mae']:.4f} "
                f"PSNR:{train_metrics['psnr']:.1f} "
                f"SSIM:{train_metrics['ssim']:.4f} | "
                f"val L:{val_metrics['loss']:.4f} "
                f"PSNR:{val_metrics['psnr']:.1f} "
                f"SSIM:{val_metrics['ssim']:.4f} "
                f"HFEN:{val_metrics['hfen']:.4f} | "
                f"预估剩余时间:{eta/60:.0f}分钟"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"lr:{current_lr:.2e} "
                f"train L:{train_metrics['loss']:.4f} "
                f"Opt:{optimizer_steps} Skip:{skipped_optimizer_steps} "
                f"Grad:{train_metrics['grad_norm']:.3f} "
                f"Scale:{train_metrics['amp_scale']:.0f} "
                f"MAE:{train_metrics['mae']:.4f} "
                f"PSNR:{train_metrics['psnr']:.1f} "
                f"SSIM:{train_metrics['ssim']:.4f} | "
                f"val L:{val_metrics['loss']:.4f} "
                f"PSNR:{val_metrics['psnr']:.1f} "
                f"SSIM:{val_metrics['ssim']:.4f} | "
                f"预估剩余时间:{eta/60:.0f}分钟"
            )
        if scheduler is not None and optimizer_steps > 0:
            scheduler.step()

        # 只使用验证集选择 checkpoint，不允许测试集参与模型选择。
        improved_ssim = val_metrics["ssim"] > best_ssim
        improved_psnr = val_metrics["psnr"] > best_psnr
        improved_hfen = val_metrics["hfen"] < best_hfen
        if improved_ssim:
            best_ssim = val_metrics["ssim"]
        if improved_psnr:
            best_psnr = val_metrics["psnr"]
        if improved_hfen:
            best_hfen = val_metrics["hfen"]
        best_metrics = {
            "ssim": best_ssim,
            "psnr": best_psnr,
            "hfen": best_hfen,
        }
        if improved_ssim:
            save_checkpoint(
                best_path, model, optimizer, epoch, best_ssim, args,
                scheduler=scheduler, scaler=scaler, global_step=global_step,
                metric_name="ssim", best_metrics=best_metrics,
            )
            save_checkpoint(
                best_ssim_path, model, optimizer, epoch, best_ssim, args,
                scheduler=scheduler, scaler=scaler, global_step=global_step,
                metric_name="ssim", best_metrics=best_metrics,
            )
            print(f"  >> 刷新最优模型，验证SSIM：{best_ssim:.6f}")
        if improved_psnr:
            save_checkpoint(
                best_psnr_path, model, optimizer, epoch, best_psnr, args,
                scheduler=scheduler, scaler=scaler, global_step=global_step,
                metric_name="psnr", best_metrics=best_metrics,
            )
            print(f"  >> 刷新PSNR最优模型：{best_psnr:.4f} dB")
        if improved_hfen:
            save_checkpoint(
                best_hfen_path, model, optimizer, epoch, best_hfen, args,
                scheduler=scheduler, scaler=scaler, global_step=global_step,
                metric_name="hfen", best_metrics=best_metrics,
            )
            print(f"  >> 刷新HFEN最优模型：{best_hfen:.6f}")

        # 每一轮都覆盖保存 latest.pth，防止断电时只剩 log.csv 没有权重。
        latest_path = save_dir / "checkpoints" / "latest.pth"
        save_checkpoint(
            latest_path, model, optimizer, epoch, best_ssim, args,
            scheduler=scheduler, scaler=scaler, global_step=global_step,
            metric_name="ssim", best_metrics=best_metrics,
        )

        # 周期断点用于续训；不一定是最优模型，但能恢复训练状态。
        if epoch % args.save_every == 0:
            ckpt_path = save_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
            save_checkpoint(
                ckpt_path, model, optimizer, epoch, best_ssim, args,
                scheduler=scheduler, scaler=scaler, global_step=global_step,
                metric_name="ssim", best_metrics=best_metrics,
            )
            print(f"  >> 保存周期断点: {ckpt_path}")


    # ==================== 全部训练完成，测试集最终评估 ====================
    print(f"\n全部训练结束！最优验证SSIM：{best_ssim:.6f}")
    # test 集只在全部训练结束后使用，用于最终报告，不参与调参和保存 best。
    if test_dir.exists():
        print("\n加载最优模型，在独立测试集评估最终效果...")
        print(f"测试集病例总数: {len(test_cases)}")
        # 测试集必须关闭数据增强，保证每次评估输入完全一致。
        test_ds = CTMRIDataset(
            test_cases, patch_size=args.patch_size, training=False,
            ct_norm=args.ct_norm, mri_norm=args.mri_norm,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,num_workers=args.num_workers,
        )
        # 加载验证集选出的 best.pth，而不是最后一个 epoch 的权重。
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        moment_calibration_scale = 1.0
        if args.moment_calibration:
            calibration = fit_moment_variance_scale(
                model=model,
                loader=val_loader,
                device=device,
                amp=args.amp,
                divisor=args.divisor,
                eval_crop_size=args.eval_crop_size,
            )
            moment_calibration_scale = float(calibration["scale"])
            write_log_row(save_dir / "moment_calibration.csv", calibration)
            print(
                "Moment验证集校准 | "
                f"scale={moment_calibration_scale:.6g} | "
                f"raw MAE={calibration['raw_variance_to_squared_error_mae']:.6g} | "
                f"calibrated MAE={calibration['calibrated_variance_to_squared_error_mae']:.6g}"
            )

        # 普通测试指标：使用 model.eval() + 普通 forward，dropout 关闭。
        # 这组 MAE/PSNR/SSIM 是重建质量指标，不包含 MC Dropout。
        test_metrics = validate_one_epoch(
            model, test_loader, criterion, device, args.amp, divisor=args.divisor,
            eval_crop_size=args.eval_crop_size,
        )
        print(
            f"测试集最终指标 | "
            f"MAE: {test_metrics['mae']:.6f} | "
            f"PSNR: {test_metrics['psnr']:.2f} dB | "
            f"SSIM: {test_metrics['ssim']:.6f} | "
            f"前景PSNR: {test_metrics['foreground_psnr']:.2f} dB | "
            f"前景SSIM: {test_metrics['foreground_ssim']:.6f} | "
            f"Gradient MAE: {test_metrics['gradient_mae']:.6f} | "
            f"HFEN: {test_metrics['hfen']:.6f}"
        )
        # 保存最终测试集重建指标。
        test_row = {f"test_{k}": f"{v:.6f}" for k, v in test_metrics.items()}
        write_log_row(save_dir / "test_results.csv", test_row)

        # ==================== MC Dropout 方差图对照评估（默认开启） ====================
        if args.eval_mc_dropout_compare:
            print("\n" + "=" * 60)
            print("启动MC Dropout方差图对照评估（单次矩传播 vs MC Dropout）...")
            print(f"  MC Dropout 推理次数: {args.mc_passes}")
            print(f"  Dropout 率: {args.dropout}")
            if args.dropout <= 0.0:
                print("  [警告] dropout=0，不确定性将全为零！请使用 --dropout 0.2 重新训练")

            # --save_uncertainty 控制是否额外保存 nii.gz 体数据；
            # 不加该参数时仍会计算 CSV 对比指标，但不会写出 NIfTI 方差图。
            unc_save_dir = str(save_dir / "uncertainty") if args.save_uncertainty else None
            unc_csv = str(save_dir / "uncertainty_results.csv")
            mc_fig_dir = save_dir / "figures" / "test_mc_dropout_compare"
            figure_state = {"saved": 0}

            def save_uncertainty_case(**case_outputs):
                if not args.save_test_figures:
                    return
                if (
                    args.max_test_figures > 0
                    and figure_state["saved"] >= args.max_test_figures
                ):
                    return
                save_precomputed_uncertainty_comparison(
                    out_dir=mc_fig_dir,
                    moment_calibration_scale=moment_calibration_scale,
                    **case_outputs,
                )
                figure_state["saved"] += 1

            # validate_with_uncertainty 会同时计算：
            #   1. Moment variance：单次矩传播，本文方法；
            #   2. MC variance：多次 MC Dropout，参考对照；
            #   3. 两张方差图之间的 RMSD/MAE。
            unc_metrics = validate_with_uncertainty(
                model=model,
                loader=test_loader,
                device=device,
                amp=args.amp,
                divisor=args.divisor,
                mc_passes=args.mc_passes,
                save_uncertainty_dir=unc_save_dir,
                csv_log_path=unc_csv,
                eval_crop_size=args.eval_crop_size,
                case_callback=save_uncertainty_case,
                moment_calibration_scale=moment_calibration_scale,
            )
            print(
                f"不确定性方差图对比结果 | "
                f"RMSD vs MC{args.mc_passes} variance: {unc_metrics['unc_rmsd_vs_mc']:.6f} | "
                f"MAE brain vs MC{args.mc_passes} variance: {unc_metrics['unc_mae_brain_vs_mc']:.6f}"
            )
            print(
                f"Moment-vs-MC相关性 | "
                f"Pearson: {unc_metrics['unc_pearson_moment_vs_mc']:.4f} | "
                f"Spearman: {unc_metrics['unc_spearman_moment_vs_mc']:.4f}"
            )
            print(
                f"不确定性-vs-绝对误差相关性 | "
                f"Moment Pearson/Spearman: "
                f"{unc_metrics['moment_error_pearson']:.4f}/"
                f"{unc_metrics['moment_error_spearman']:.4f} | "
                f"MC Pearson/Spearman: "
                f"{unc_metrics['mc_error_pearson']:.4f}/"
                f"{unc_metrics['mc_error_spearman']:.4f}"
            )
            print(
                f"高不确定区域覆盖高误差区域(top 10%) | "
                f"Moment: {unc_metrics['moment_top10_error_overlap']:.4f} | "
                f"MC: {unc_metrics['mc_top10_error_overlap']:.4f}"
            )
            print(
                "Moment方差与真实平方误差的MAE | "
                f"raw: {unc_metrics['moment_raw_variance_error_mae']:.6f} | "
                f"validation-calibrated: "
                f"{unc_metrics['moment_calibrated_variance_error_mae']:.6f}"
            )
            print(
                f"平均推理时间 | Moment: "
                f"{unc_metrics['moment_inference_seconds']:.3f}s (1次) | "
                f"MC{args.mc_passes}: {unc_metrics['mc_inference_seconds']:.3f}s "
                f"({args.mc_passes}次) | "
                f"时间比: {unc_metrics['mc_over_moment_time_ratio']:.1f}x"
            )
            print(
                f"峰值额外显存 | Moment: "
                f"{unc_metrics['moment_peak_memory_mb']:.1f} MB | "
                f"MC{args.mc_passes}: {unc_metrics['mc_peak_memory_mb']:.1f} MB"
            )
            if args.save_test_figures:
                print(
                    f"同次推理的一一对应对比图保存至: {mc_fig_dir} "
                    f"({figure_state['saved']} cases)"
                )
            if args.save_uncertainty:
                print(f"不确定性方差nii文件保存至: {unc_save_dir}")
            print(f"不确定性方差CSV日志保存至: {unc_csv}")
            print("=" * 60)
        elif args.save_test_figures:
            # 关闭 MC 对照时，仍保存主方法 Moment 的普通测试图。
            fig_dir = save_dir / "figures" / "test"
            save_test_comparison_figures(
                model=model,
                loader=test_loader,
                device=device,
                amp=args.amp,
                divisor=args.divisor,
                out_dir=fig_dir,
                max_cases=args.max_test_figures,
                eval_crop_size=args.eval_crop_size,
            )
    print(f"\n所有模型、日志文件存放路径：{save_dir}")


# Program entry point.
if __name__ == "__main__":
    main()
