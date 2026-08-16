#!/usr/bin/env python3
"""
model_attnres3d_gtmedreclpp.py

3D CT -> MRI 重建、Geometry-Tolerant Med-ReCL++ 与不确定性估计核心实现。

当前代码包含三条清晰分工的前向路径：
1. forward()
   普通 CT->MRI 重建前向。
   训练阶段用它计算重建 loss 并更新网络权重；验证/普通测试阶段也用它计算 MAE、PSNR、SSIM。

2. forward_mu_var_cov()
   本实验的主方法：单次矩传播不确定性推理。
   它不更新权重，不做 backward，只把已经训练好的网络权重代入解析矩传播公式，
   一次前向同时输出重建 MRI 和 Moment propagation variance map。

3. mc_dropout_inference()
   参考对照方法：MC Dropout。
   它使用同一个 best.pth 权重，只在测试/分析阶段打开 dropout 随机性，多次前向后求方差。
   作用是和 Moment variance map 对比，不是默认临床推理必须步骤。

重要约定：
- uncertainty 变量默认表示 variance map（方差图），不是 standard deviation。
- 如果需要和图像强度同单位的 std，需要显式使用 sqrt(variance.clamp_min(0))。
- MomentDrop3d 使用 PyTorch 标准 inverted dropout 定义：训练时 x*mask/keep，eval 时直接 x。
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


# ======================================================================================================================================================
# Utilities
# ======================================================================================================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_eval_crop_size(text) -> Optional[Tuple[int, int, int]]:
    """Parse an evaluation crop size such as 150,150,150."""
    if text is None:
        return None
    if isinstance(text, (tuple, list)):
        if len(text) == 1:
            return parse_eval_crop_size(text[0])
        if len(text) != 3:
            raise ValueError("eval crop size must have three values")
        return tuple(int(v) for v in text)
    text = str(text).strip().lower()
    if text in {"", "none", "full", "false", "0"}:
        return None
    parts = [int(item.strip()) for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError("eval crop size must look like 150,150,150")
    return tuple(parts)


def crop_pair_around_foreground(
    source: torch.Tensor,
    target: torch.Tensor,
    crop_size: Optional[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop paired [B,C,D,H,W] CT/MRI tensors around the foreground center."""
    if crop_size is None:
        return source, target
    if source.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("Evaluation crop expects batch_size=1")

    _, _, d, h, w = source.shape
    cd, ch, cw = crop_size
    # Evaluation/test crops must be determined from CT only. Using target MRI
    # here would leak ground-truth information into the tested field of view.
    mask = source[0, 0] > 0.01
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() > 0:
        center = coords.float().mean(dim=0).round().long()
    else:
        center = torch.tensor([d // 2, h // 2, w // 2], device=source.device)

    shape = torch.tensor([d, h, w], device=source.device)
    crop = torch.tensor([cd, ch, cw], device=source.device)
    crop = torch.minimum(crop, shape)
    start = center - crop // 2
    start = torch.maximum(torch.zeros_like(start), torch.minimum(start, shape - crop))
    end = start + crop
    z0, y0, x0 = [int(v) for v in start.tolist()]
    z1, y1, x1 = [int(v) for v in end.tolist()]
    return source[:, :, z0:z1, y0:y1, x0:x1], target[:, :, z0:z1, y0:y1, x0:x1]


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as float32."""
    arr = nib.load(str(path)).get_fdata()
    return np.asarray(arr, dtype=np.float32)


def clip_percentile_norm(
    x: np.ndarray,
    low=0.5,
    high=99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Clip intensity outliers by percentile and normalize to [0, 1]."""
    lo = np.percentile(x, low)
    hi = np.percentile(x, high)
    x = np.clip(x, lo, hi)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x.astype(np.float32)


def zscore_nonzero(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score only nonzero voxels while keeping background unchanged."""
    mask = volume != 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    vals = volume[mask]
    mean = vals.mean()
    std = vals.std()
    out = volume.copy().astype(np.float32)
    out[mask] = (out[mask] - mean) / (std + eps)
    return out


def pad_to_shape(
    x: np.ndarray,
    target_shape: Tuple[int, int, int],
    value: float = 0.0,
) -> np.ndarray:
    """Symmetrically pad a 3D volume to at least target_shape."""
    d, h, w = x.shape
    td, th, tw = target_shape

    pd = max(0, td - d)
    ph = max(0, th - h)
    pw = max(0, tw - w)

    pad_before = (pd // 2, ph // 2, pw // 2)
    pad_after = (pd - pad_before[0], ph - pad_before[1], pw - pad_before[2])

    return np.pad(
        x,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode="constant",
        constant_values=value,
    )


def random_crop_pair(
    source: np.ndarray,
    target: np.ndarray,
    crop_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop paired CT/MRI tensors at the same spatial location."""
    _, d, h, w = source.shape
    cd, ch, cw = crop_shape

    if d < cd or h < ch or w < cw:
        pad_shape = (max(d, cd), max(h, ch), max(w, cw))
        source = np.stack([pad_to_shape(ch_i, pad_shape) for ch_i in source], axis=0)
        target = np.stack([pad_to_shape(ch_i, pad_shape) for ch_i in target], axis=0)
        _, d, h, w = source.shape

    sd = 0 if d == cd else random.randint(0, d - cd)
    sh = 0 if h == ch else random.randint(0, h - ch)
    sw = 0 if w == cw else random.randint(0, w - cw)

    source = source[:, sd:sd + cd, sh:sh + ch, sw:sw + cw]
    target = target[:, sd:sd + cd, sh:sh + ch, sw:sw + cw]
    return source, target


def random_flip_pair(
    source: np.ndarray,
    target: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synchronized random flips to paired CT/MRI tensors."""
    for axis in [1, 2, 3]:
        if random.random() < p:
            source = np.flip(source, axis=axis).copy()
            target = np.flip(target, axis=axis).copy()
    return source, target


def random_intensity_shift_scale(
    x: np.ndarray,
    shift_std: float = 0.05,
    scale_std: float = 0.05,
    p: float = 0.5,
) -> np.ndarray:
    """Apply mild random intensity shift/scale augmentation to the source image."""
    if random.random() >= p:
        return x
    x = x.copy()
    for c in range(x.shape[0]):
        shift = np.random.normal(0.0, shift_std)
        scale = np.random.normal(1.0, scale_std)
        x[c] = x[c] * scale + shift
    return x


def pad_tensor_to_divisible(
    x: torch.Tensor,
    divisor: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """Pad a 5D tensor so D/H/W are divisible by divisor."""
    _, _, d, h, w = x.shape
    pd = (divisor - d % divisor) % divisor
    ph = (divisor - h % divisor) % divisor         
    pw = (divisor - w % divisor) % divisor
    pad = (
        pw // 2, pw - pw // 2,
        ph // 2, ph - ph // 2,
        pd // 2, pd - pd // 2,
    )
    x = F.pad(x, pad, mode="constant", value=0.0)
    return x, pad


def unpad_tensor(x: torch.Tensor, pad: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """Remove padding produced by pad_tensor_to_divisible."""
    pwl, pwr, phl, phr, pdl, pdr = pad
    d_slice = slice(pdl, x.shape[2] - pdr if pdr > 0 else x.shape[2])
    h_slice = slice(phl, x.shape[3] - phr if phr > 0 else x.shape[3])
    w_slice = slice(pwl, x.shape[4] - pwr if pwr > 0 else x.shape[4])
    return x[:, :, d_slice, h_slice, w_slice]
# =======================================================================================================================================================
# Dataset
# =======================================================================================================================================================

#1. 数据结构定义
@dataclass           # dataclass装饰器：自动生成__init__、打印、赋值等基础方法，只用来存路径数据，不用手写构造函数
class CasePaths:     # 单个病例的路径存储类
    case_id: str     # 病人唯一ID，字符串类型，对应文件夹名称
    ct: Path         # Path对象，该病例CT nii.gz文件完整路径
    mri: Path        # Path对象，该病例MRI nii.gz文件完整路径


#2. 数据检索函数-自动找病人数据
def discover_cases(split_dir: str | Path) -> List[CasePaths]:      # 输入：训练集/验证集根文件夹（字符串/Path路径都行） #输出：列表，列表中每个元素是CasePaths，代表一个完整病人数据
    split_dir = Path(split_dir)                    # 统一转为Path对象，Windows/Linux路径自动兼容
    if not split_dir.exists():                     # 文件夹不存在直接抛错，避免后续读取崩溃
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    cases: List[CasePaths] = []                    # 初始化病例列表  # 空列表，用来存放所有有效病人
    for case_dir in sorted(split_dir.iterdir()):   # split_dir.iterdir()：遍历根目录下所有子文件/子文件夹。# sorted()：排序遍历，每次运行加载病人顺序固定，实验可复现。# case_dir：循环中「单个病人文件夹」的Path对象
        if not case_dir.is_dir():                  # 过滤非文件夹对象。如果不是文件夹，直接跳过  #跳过文件，只处理文件夹
            continue
        case_id = case_dir.name                    # 文件夹名字直接作为病人编号
        ct_candidates = (
            case_dir / f"{case_id}_ct.nii.gz",
            case_dir / "ct.nii.gz",
        )
        mri_candidates = (
            case_dir / f"{case_id}_mri.nii.gz",
            case_dir / "mri.nii.gz",
            case_dir / "mr.nii.gz",
        )
        ct = next((p for p in ct_candidates if p.exists()), ct_candidates[0])
        mri = next((p for p in mri_candidates if p.exists()), mri_candidates[0])

        # 核心逻辑：只有当CT和MRI文件同时存在时，该病例才被视为有效并加入列表
        if ct.exists() and mri.exists():
            cases.append(CasePaths(case_id=case_id, ct=ct, mri=mri))

    if len(cases) == 0:# 防错机制：若未发现有效病例则抛出异常
        raise RuntimeError(f"No valid cases found in {split_dir}")
    return cases


def validate_case_splits(split_cases: Dict[str, List[CasePaths]]) -> None:
    """
    Validate one-to-one CT/MRI pairing and patient independence across splits.

    This reads only NIfTI headers, not image voxels, so it does not expose test
    image content during training.
    """
    seen: Dict[str, str] = {}
    for split_name, cases in split_cases.items():
        local_ids = set()
        for case in cases:
            if case.case_id in local_ids:
                raise RuntimeError(
                    f"Duplicate case ID inside {split_name}: {case.case_id}"
                )
            local_ids.add(case.case_id)
            if case.case_id in seen:
                raise RuntimeError(
                    f"Patient leakage: {case.case_id} appears in both "
                    f"{seen[case.case_id]} and {split_name}"
                )
            seen[case.case_id] = split_name

            ct_image = nib.load(str(case.ct))
            mri_image = nib.load(str(case.mri))
            if ct_image.shape != mri_image.shape:
                raise RuntimeError(
                    f"CT/MRI shape mismatch for {case.case_id}: "
                    f"{ct_image.shape} vs {mri_image.shape}"
                )
            if not np.allclose(ct_image.affine, mri_image.affine, atol=1e-4):
                raise RuntimeError(
                    f"CT/MRI affine mismatch for {case.case_id}; paired voxels "
                    "would not represent the same physical locations."
                )


#3.负责把 CT 和 MRI 读出来、处理好、喂给模型训练（最重要的一部分啊）
class CTMRIDataset(Dataset):
    def __init__(                                  #参数设置
        self,
        cases: List[CasePaths],                    # 前面discover_cases返回的所有病例列表
        patch_size: Tuple[int, int, int],          # 训练时随机裁剪的3D小块尺寸 (D,H,W)
        training: bool = True,                     # 开关：True训练模式（做增强）；False推理/验证模式（不增强，只补边）
        ct_norm: str = "clip01",                   # CT图像归一化方案：clip01 / zscore_nonzero
        mri_norm: str = "clip01",                  # MRI图像归一化方案
    ) -> None: 
        self.cases = cases                         # 把入参全部存为实例变量，类内所有函数都能调用。
        self.patch_size = patch_size
        self.training = training
        self.ct_norm = ct_norm
        self.mri_norm = mri_norm

    def _normalize(self, x: np.ndarray, mode: str) -> np.ndarray:  #内部工具：根据mode选择对应的归一化函数。
        if mode == "clip01":
            return clip_percentile_norm(x)        # 通常指将 0.5% 至 99.5% 强度映射到 [0, 1]，消除极值影响
        if mode == "zscore_nonzero":
            return zscore_nonzero(x)              # 仅针对非零区域（非背景）计算均值和标准差进行归一化
        raise ValueError(f"Unknown normalization mode: {mode}")     # 传入不认识的归一化参数直接报错

    def __len__(self) -> int:                    
        return len(self.cases)                   # 返回数据集总病例数量，DataLoader取总样本要用

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:    #  输入索引idx，返回单条样本（一对CT/MRI）
        case = self.cases[idx]                                     # 【拿一个病人】根据索引检索对应的CasePaths对象
        ct = self._normalize(load_nifti(case.ct), self.ct_norm)    #  加载 NIfTI 原始文件并立即进行标准化处理
        mri= self._normalize(load_nifti(case.mri), self.mri_norm)
        source = ct[None, ...]                     # 在第0维增加通道轴(C, D, H, W)，满足PyTorch卷积层的4D/5D输入要求
        target = mri[None, ...]                    # (D,H,W)→(1,D,H,W)【PyTorch模型必须带通道，所以加一维】#none只是增加维度
        # ========== 分支1：训练模式，开启成对数据增强 ==============
        if self.training:        #如果是训练模式 → 做数据增强
            source, target = random_crop_pair(source, target, self.patch_size) # 同步随机裁剪：CT/MRI切完全相同位置的小块
            source, target = random_flip_pair(source, target, p=0.5)           # 同步随机翻转：三轴50%概率镜像，输入输出对齐
            source = random_intensity_shift_scale(source, p=0.5)               # 仅对CT输入做亮度对比度扰动，MRI真值不改动
        # ==========分支2：验证/推理模式，无增强，仅补边 =============
        else:                  
            d, h, w = source.shape[1:]            # 推理阶段：处理尺寸不匹配问题
            td, th, tw = self.patch_size         
            if d < td or h < th or w < tw:         # 如果原图尺寸比patch小块还小，对称补零到patch大小
                pad_shape = (max(d, td), max(h, th), max(w, tw))
                source = np.stack([pad_to_shape(ch, pad_shape) for ch in source], axis=0)   #分通道补边，再拼回4维(C,D,H,W)
                target = np.stack([pad_to_shape(ch, pad_shape) for ch in target], axis=0)
        # numpy数组转GPU可用torch.float32张量，打包字典返回
        return {                                          # 返回字典格式数据，并将Numpy数组转换为PyTorch FloatTensor
            "source": torch.from_numpy(source).float(),   # 把处理好的CT当输入，MRI当目标，返回给模型
            "target": torch.from_numpy(target).float(),
            "case_id": case.case_id,    # 病人ID
        }# torch.from_numpy(source)；source 是 (1,D,H,W) 的 float32 numpy 数组，执行转换：；numpy 数组 → torch 张量，默认继承原数组的数据类型；因为前面预处理返回的是np.float32，这里转完是 torch.float32 张量。#第二步：.float()：强制把张量转为浮点型张量（双重保险），网络训练必须用浮点，不能用整数。
        #（准备好数据之后要变成PyTorch张量）


# =======================================================================================================================================================
# Losses / Metrics    （3D 重建任务常用的损失函数与评价指标）
# =======================================================================================================================================================
def gaussian_kernel_3d(kernel_size: int = 7, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """
    构造3D高斯卷积核，用于SSIM3D中的局部统计计算。

    参数:
        kernel_size: 高斯核大小，通常取奇数，例如7(模糊窗口大小 7x7x7)
        sigma: 高斯分布标准差，控制平滑程度(模糊程度)
        channels: 输入通道数，用于生成适配多通道输入的卷积核(通道数)
    返回:
        shape = [channels, 1, kernel_size, kernel_size, kernel_size]的3D高斯核
    """
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel_3d = kernel_1d / kernel_1d.sum()
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    return kernel_3d.repeat(channels, 1, 1, 1, 1)


class SSIM3D(nn.Module):
    """
    3D 结构相似性指标（SSIM）的实现。
    常用于评估 3D 图像/体数据重建质量，例如 CT、MRI、体素重建等。
    """
    def __init__(self, channels: int = 1, kernel_size: int = 7, sigma: float = 1.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.register_buffer("window", gaussian_kernel_3d(kernel_size, sigma, channels))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        data_range: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算输入 x 和 y 的 3D SSIM 值。
        参数:
            x: 预测张量，shape 通常为 [B, C, D, H, W]
            y: 目标张量，shape 与 x 相同
            data_range: 数据动态范围，若输入已归一化到 [0, 1]，则取 1.0

        返回:
            整个 batch 的平均 SSIM 值，越接近 1 越相似                       
        """                                    # 代码固定 data_range=1.0，代表我们约定图像有效灰度区间是 0~1。如果预测像素出现负数、大于 1 的异常值，C1/C2 常数会匹配不上，SSIM 分数直接失真、乱飘，指标完全不准。
        x = x.float()
        y = y.to(device=x.device, dtype=x.dtype)
        c1 = (0.01 * data_range) ** 2          # SSIM 中的两个稳定常数，避免分母过小导致数值不稳定               # 高斯核同步到输入张量的显卡、数据类型
        c2 = (0.03 * data_range) ** 2          # 3.不做 padding 会发生什么？（对比直观理解），输入深度 D=64，kernel=7，padding=0，stride=1：Out=64+0−7+1=58，输出直接缩小成 58，图像边缘一圈像素被丢掉。而 SSIM 需要每个像素都有对应的局部窗口均值方差，丢边缘会计算出错，所以必须补边。，4. padding 到底在补什么？，padding=3：在图像上下、左右、前后（三维 D/H/W） 各填充 3 圈 0，把原图撑大一圈。卷积窗口滑到原图最边缘时，旁边有填充的像素兜底，不会滑出图像外，因此输出尺寸不变。
        padding = self.kernel_size // 2        # 为保持输出尺寸不变，使用same padding【因为后面做卷积时，如果不补边，图像尺寸会变小】【他是为了让输入数据卷积后输出的尺寸不变】
        window = self.window.to(dtype=x.dtype, device=x.device)   # 把保存好的高斯核self.window，变成和输入x一样的数据类型、一样的设备

        mu_x = F.conv3d(x, window, padding=padding, groups=self.channels)    #计算局部均值 μx, μy；groups=self.channels 表示每个通道独立做卷积【得到的是一整张"局部均值图"】
        mu_y = F.conv3d(y, window, padding=padding, groups=self.channels)    #F.conv3d(...)：表示做 3D 卷积。【因为这里的 window 不是一般卷积核，而是一个 归一化后的高斯核；所以它和某个局部区域做卷积时，本质上就是在算：这个局部区域的加权平均】


        mu_x2 = mu_x.pow(2)    # 均值平方与均值乘积
        mu_y2 = mu_y.pow(2)    # mu_x是x的局部平均值；mu_x2就是这个局部平均值再平方【(E[x])2】
        mu_xy = mu_x * mu_y

                                                                            
        sigma_x2 = F.conv3d(x * x, window, padding=padding, groups=self.channels) - mu_x2   #【F.conv3d(x * x, window, padding=padding, groups=self.channels)为-----E[x2]】计算局部方差 σx², σy² 和协方差 σxy    #σx2​=E[x2]−(E[x])2  /  σxy​=E[xy]−E[x]E[y]
        sigma_y2 = F.conv3d(y * y, window, padding=padding, groups=self.channels) - mu_y2   # 2. 局部方差、协方差公式：σ² = E[x²] - (E[x])²
        sigma_xy = F.conv3d(x * y, window, padding=padding, groups=self.channels) - mu_xy

        # SSIM 公式:
        # ((2μxμy + c1)(2σxy + c2)) / ((μx² + μy² + c1)(σx² + σy² + c2))+（1e-8）
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8                    #ssim_map：每个体素位置单独的SSIM分数，3D体图
        )
        if mask is not None:
            mask = mask.to(device=ssim_map.device, dtype=torch.bool)
            if mask.shape != ssim_map.shape:
                mask = mask.expand_as(ssim_map)
            if torch.any(mask):
                return ssim_map[mask].mean()
        return ssim_map.mean()            #返回整个张量的平均SSIM【算然他算的是整张图的最后一个结果，但是他是分局部进行计算的，所有最后求一个平均，就是它的值】假设一张图简化后只有 4 个像素，ssim_map = [0.9, 0.8, 0.95, 0.75]平均值 = (0.9 + 0.8 + 0.95 + 0.75) / 4 = 0.85     # 整张batch所有体素取平均，返回0~1之间标量


#所有 nn.Module 前向传播默认自动记录计算图、保存梯度，训练时用来 loss.backward() 更新网络权重。
#【一类是训练时真的拿来优化模型的"损失函数"】
class MultiScaleSSIM3D(nn.Module):
    """Memory-conscious three-scale SSIM for 3D reconstruction training."""

    def __init__(
        self,
        channels: int = 1,
        scale_weights: Tuple[float, ...] = (0.50, 0.30, 0.20),
    ):
        super().__init__()
        if not scale_weights:
            raise ValueError("scale_weights must contain at least one value")
        weights = torch.tensor(scale_weights, dtype=torch.float32)
        self.register_buffer("scale_weights", weights / weights.sum())
        self.ssim = SSIM3D(channels=channels)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        data_range: float = 1.0,
    ) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        x_scale = x
        y_scale = y
        for scale in range(self.scale_weights.numel()):
            scores.append(self.ssim(x_scale, y_scale, data_range=data_range))
            if scale + 1 < self.scale_weights.numel():
                if min(x_scale.shape[-3:]) < 4:
                    break
                x_scale = F.avg_pool3d(x_scale, kernel_size=2, stride=2)
                y_scale = F.avg_pool3d(y_scale, kernel_size=2, stride=2)

        weights = self.scale_weights[:len(scores)]
        weights = weights / weights.sum()
        return torch.sum(torch.stack(scores) * weights.to(scores[0]))


class ReconstructionLoss(nn.Module):      #总损失函数，继承自PyTorch的nn.Module。
    """
    重建主损失:
    L_rec = 0.45 * L1 + 0.30 * (1 - MS-SSIM3D)
            + 0.15 * GDL3D + 0.10 * FFL3D。

    L1 保持强度，多尺度 SSIM 保持跨尺度结构，梯度差保持组织边界，
    弱 Focal Frequency Loss 约束仍未恢复的频率细节。
    """
    def __init__(
        self,
        l1_weight: float = 0.45,
        ssim_weight: float = 0.30,
        edge_weight: float = 0.15,
        frequency_weight: float = 0.10,
        frequency_alpha: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.frequency_weight = frequency_weight
        self.frequency_alpha = frequency_alpha
        self.l1 = nn.L1Loss()               # L1公式：loss = 所有像素 |pred - target| 全部加起来 / 总像素数量（全局平均）
        self.ssim = MultiScaleSSIM3D(channels=1)
        self.last_components: Dict[str, float] = {}

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dz_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dz_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dy_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        dx_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dx_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        return (
            F.l1_loss(torch.abs(dz_pred), torch.abs(dz_target))
            + F.l1_loss(torch.abs(dy_pred), torch.abs(dy_target))
            + F.l1_loss(torch.abs(dx_pred), torch.abs(dx_target))
        ) / 3.0

    @staticmethod
    def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        dz = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return torch.sqrt(dz.pow(2) + dy.pow(2) + dx.pow(2) + 1e-12)

    def _focal_frequency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        3D Focal Frequency Loss。

        使用 rFFT 节省显存；难匹配频率获得更大权重。权重从当前频谱误差
        计算后停止梯度，避免网络通过改变权重本身降低损失。FFT 强制使用
        FP32，保证 AMP 下 96^3 等非 2 次幂尺寸也能在 CUDA 上稳定运行。
        """
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_freq = torch.fft.rfftn(
                pred.float(),
                dim=(-3, -2, -1),
                norm="ortho",
            )
            target_freq = torch.fft.rfftn(
                target.float(),
                dim=(-3, -2, -1),
                norm="ortho",
            )
            distance_sq = torch.abs(pred_freq - target_freq).square()

            focal_weight = distance_sq.detach().clamp_min(1e-12)
            focal_weight = focal_weight.pow(0.5 * self.frequency_alpha)
            max_weight = focal_weight.amax(dim=(-3, -2, -1), keepdim=True)
            focal_weight = focal_weight / max_weight.clamp_min(1e-12)

            # DC 只描述整幅平均灰度，已经由 L1 负责；频域项聚焦结构细节。
            dc_mask = torch.ones_like(focal_weight)
            dc_mask[..., 0, 0, 0] = 0.0
            focal_weight = focal_weight * dc_mask
            return torch.mean(focal_weight * distance_sq)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:   #定义这个损失函数真正计算的过程
        """
        计算重建损失。

        参数:
            pred: 模型预测输出
            target: 真实目标

        返回:
            加权后的总重建损失
        """
        pred = torch.clamp(pred.float(), 0.0, 1.0)               # 将输入裁剪到 [0, 1]，防止越界影响损失计算。clamp 的意思可以理解成：裁剪/卡住（小于 0.0 的，变成 0.0;大于 1.0 的，变成 1.0，中间的值保持不变）
        target = torch.clamp(target.to(device=pred.device, dtype=pred.dtype), 0.0, 1.0)           # torch.clamp 只改像素值，图像尺寸不变   
        l1 = self.l1(pred, target)                       # 计算 L1 损失
        # 3D SSIM contains several local variance subtractions. Keep this
        # numerically sensitive part in FP32 even when the backbone uses AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            ssim_loss = 1.0 - self.ssim(pred.float(), target.float())
        edge_loss = self._edge_loss(pred, target)
        frequency_loss = (
            self._focal_frequency_loss(pred, target)
            if self.frequency_weight > 0.0
            else pred.new_zeros(())
        )
        weighted_l1 = self.l1_weight * l1
        weighted_ssim = self.ssim_weight * ssim_loss
        weighted_edge = self.edge_weight * edge_loss
        weighted_frequency = self.frequency_weight * frequency_loss
        total = (
            weighted_l1
            + weighted_ssim
            + weighted_edge
            + weighted_frequency
        )
        self.last_components = {
            "l1": float(l1.detach().item()),
            "ms_ssim": float(ssim_loss.detach().item()),
            "edge": float(edge_loss.detach().item()),
            "frequency": float(frequency_loss.detach().item()),
            "weighted_l1": float(weighted_l1.detach().item()),
            "weighted_ms_ssim": float(weighted_ssim.detach().item()),
            "weighted_edge": float(weighted_edge.detach().item()),
            "weighted_frequency": float(weighted_frequency.detach().item()),
            "total": float(total.detach().item()),
        }
        return total


@dataclass
class MedReCLConfig:
    """Geometry-Tolerant Structure-Appearance Med-ReCL++ parameters."""
    enabled: bool = True
    proj_dim: int = 64
    appearance_start_ratio: float = 0.50
    appearance_ramp_end_ratio: float = 0.85
    lambda_x: float = 0.0
    level_weights: Tuple[float, float, float] = (0.25, 0.5, 1.0)
    appearance_level_weights: Tuple[float, float, float] = (1.0, 0.5, 0.25)
    tau0: float = 0.20
    tau_min: float = 0.10
    tau_max: float = 0.30
    kappa_g: float = 0.5
    kappa_r: float = 0.5
    beta_g: float = 1.0
    beta_r: float = 1.0
    rho_s: float = 0.5
    rho_g: float = 0.5
    rho_r: float = 0.5
    lambda_B: float = 1.0
    lambda_R: float = 0.0
    align_weight: float = 0.2
    lambda_structure_max: float = 0.01
    lambda_appearance_max: float = 0.004
    recon_only_ratio: float = 0.20
    structure_ramp_end_ratio: float = 0.60
    contrast_max_ratio: float = 0.15
    structure_cap_ratio: float = 0.08
    appearance_cap_ratio: float = 0.06
    appearance_context_samples: int = 128
    appearance_context_bandwidth: float = 0.5
    appearance_context_radii: Tuple[int, int, int] = (4, 3, 2)
    appearance_amplitude_weight: float = 0.60
    appearance_contextual_weight: float = 0.30
    appearance_stat_weight: float = 0.10
    appearance_stat_std_weight: float = 0.5
    teacher_momentum: float = 0.99
    # The data are LPS with the first tensor axis left-right. Registration is
    # exact in shape/affine, so only the two in-plane axes receive one-voxel
    # tolerance at the highest-resolution decoder level. Coarser levels stay
    # coordinate-exact to avoid anatomy-crossing matches.
    soft_positive_radii: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
    )
    soft_positive_center_prior: float = 0.50
    soft_positive_spatial_eta: float = 0.10
    false_negative_threshold: float = 0.95
    false_negative_weight: float = 0.10
    invariance_weight: float = 0.05
    invariance_level_weights: Tuple[float, float, float] = (0.25, 0.5, 1.0)
    invariance_samples: int = 256
    invariance_temperature: float = 0.10
    contrast_gamma_min: float = 0.85
    contrast_gamma_max: float = 1.15
    contrast_scale_min: float = 0.90
    contrast_scale_max: float = 1.10
    contrast_bias: float = 0.05
    contrast_noise_std: float = 0.01
    gradient_balance_interval: int = 10
    lambda_easy: float = 0.1
    lambda_A: float = 0.1
    gamma_max: float = 1.0
    gamma_min: float = 0.3
    delta_mu: float = 0.06
    # Only near-identical pairs are fully ignored. Far-away highly similar
    # tissue pairs are retained with a small denominator weight by FN suppression.
    ignore_distance: float = 0.02
    hard_distance: float = 0.35
    error_hard_distance: float = 0.45
    normal_distance: float = 0.90
    background_mu_threshold: float = 0.02
    background_grad_threshold: float = 0.05
    boundary_quantile: float = 0.80
    error_quantile: float = 0.80
    anchor_samples: int = 1024
    normal_negative_samples: int = 256
    hard_negative_samples: int = 64
    easy_background_samples: int = 64
    candidate_pool_size: int = 4096
    exclusion_radii: Tuple[int, int, int] = (3, 2, 1)
    stats_kernel: int = 3
    eps: float = 1e-8


class ProjectionHead3D(nn.Module):
    """将多尺度 3D 特征映射到统一对比空间。"""

    def __init__(self, in_channels: int, proj_dim: int):
        super().__init__()
        hidden = max(in_channels, proj_dim)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden, proj_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AppearanceProjectionHead3D(nn.Module):
    """Preserve MRI amplitude information without InstanceNorm or L2 normalization."""

    def __init__(self, in_channels: int, proj_dim: int):
        super().__init__()
        hidden = max(in_channels, proj_dim)
        self.conv1 = nn.Conv3d(in_channels, hidden, kernel_size=1, bias=True)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv3d(hidden, proj_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.activation(self.conv1(x)))

    def forward_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same 1x1x1 head to sampled [N,C] vectors in FP32."""
        weight1 = self.conv1.weight[:, :, 0, 0, 0]
        weight2 = self.conv2.weight[:, :, 0, 0, 0]
        x = F.linear(x.float(), weight1.float(), self.conv1.bias.float())
        x = self.activation(x)
        return F.linear(x, weight2.float(), self.conv2.bias.float())


class MedReCLTargetEncoder(nn.Module):
    """训练阶段使用的轻量 MRI target encoder。"""

    def __init__(self, in_channels: int, c1: int, c2: int, c3: int):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, c1, dropout=0.0)
        self.enc2 = DownBlock3D(c1, c2, dropout=0.0)
        self.enc3 = DownBlock3D(c2, c3, dropout=0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return [f1, f2, f3]


class MedReCLLoss(nn.Module):
    """Geometry-tolerant structure contrast plus MRI appearance alignment."""

    def __init__(self, config: Optional[MedReCLConfig] = None):
        super().__init__()
        self.config = config or MedReCLConfig()
        self._last_balance_step = -1
        self._structure_gradient_scale = 1.0
        self._appearance_gradient_scale = 1.0
        self._structure_gradient_ratio = 0.0
        self._appearance_gradient_ratio = 0.0
        self._last_appearance_amplitude_loss = 0.0
        self._last_appearance_contextual_loss = 0.0
        self._last_appearance_stat_loss = 0.0
        self._last_soft_positive_similarity = 0.0
        self._last_false_negative_ratio = 0.0
        self._last_invariance_loss = 0.0
        self._soft_positive_values: List[float] = []
        self._false_negative_values: List[float] = []

    def progressive_weights(self, current_step: int, total_steps: int) -> Tuple[float, float]:
        if not self.config.enabled:
            return 0.0, 0.0
        cfg = self.config
        progress = float(current_step) / float(max(total_steps, 1))
        progress = max(0.0, min(1.0, progress))
        recon_only = max(0.0, min(cfg.recon_only_ratio, 0.95))
        structure_end = max(recon_only + cfg.eps, min(cfg.structure_ramp_end_ratio, 0.98))
        appearance_start = max(recon_only, min(cfg.appearance_start_ratio, 0.98))
        appearance_end = max(
            appearance_start + cfg.eps,
            min(cfg.appearance_ramp_end_ratio, 1.0),
        )

        if progress <= recon_only:
            structure_weight = 0.0
        elif progress < structure_end:
            structure_progress = (progress - recon_only) / max(structure_end - recon_only, cfg.eps)
            structure_weight = cfg.lambda_structure_max * (
                0.5 * (1.0 - math.cos(math.pi * structure_progress))
            )
        else:
            structure_weight = cfg.lambda_structure_max

        if progress <= appearance_start:
            appearance_weight = 0.0
        elif progress < appearance_end:
            appearance_progress = (progress - appearance_start) / max(
                appearance_end - appearance_start,
                cfg.eps,
            )
            appearance_weight = cfg.lambda_appearance_max * (
                0.5 * (1.0 - math.cos(math.pi * appearance_progress))
            )
        else:
            appearance_weight = cfg.lambda_appearance_max
        return float(structure_weight), float(appearance_weight)

    def contrast_ratio_cap(self, current_step: int, total_steps: int) -> float:
        if not self.config.enabled:
            return 0.0
        cfg = self.config
        progress = float(current_step) / float(max(total_steps, 1))
        progress = max(0.0, min(1.0, progress))
        if progress <= cfg.recon_only_ratio:
            return 0.0
        return float(cfg.contrast_max_ratio)

    def forward_components(
        self,
        model: "AttnResCTtoMRI",
        source: torch.Tensor,
        target: torch.Tensor,
        pred: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
        current_step: int,
        total_steps: int,
        compute_structure: bool = True,
        compute_appearance: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.enabled:
            zero = pred.new_zeros(())
            return zero, zero

        target_feats = model.extract_target_features(target, use_teacher=False)
        teacher_target_feats = model.extract_target_features(target, use_teacher=True)
        projected = model.project_medrecl_features(
            feature_dict["dec"],
            feature_dict["enc"],
            target_feats,
            teacher_target_feats=teacher_target_feats,
        )

        with torch.no_grad():
            source_aux = self._minmax_normalize(source.detach())
            target_aux = self._minmax_normalize(target.detach())
            pred_aux = torch.clamp(pred.detach(), 0.0, 1.0)
            grad_map = self._minmax_normalize(self._gradient_magnitude(target_aux))
            error_map = self._minmax_normalize(torch.abs(pred_aux - target_aux))

        level_losses: List[torch.Tensor] = []
        level_contexts: List[Dict[str, torch.Tensor]] = []
        level_weights = pred.new_tensor(self.config.level_weights, dtype=pred.dtype)
        appearance_level_weights = pred.new_tensor(
            self.config.appearance_level_weights,
            dtype=pred.dtype,
        )

        self._soft_positive_values = []
        self._false_negative_values = []
        for level, (zg, zx, zy) in enumerate(
            zip(projected["gen"], projected["ct"], projected["mri_teacher"])
        ):
            level_context = self._build_level_context(
                source_aux=source_aux,
                target_aux=target_aux,
                grad_map=grad_map,
                error_map=error_map,
                feature_shape=zg.shape[-3:],
            )
            level_contexts.append(level_context)
            if compute_structure:
                level_losses.append(
                    self._level_loss(
                        zg=zg,
                        zx=zx,
                        zy=zy,
                        context=level_context,
                        level=level,
                        gamma_t=self._gamma_t(current_step=current_step, total_steps=total_steps),
                    )
                )

        structure_loss = (
            torch.sum(level_weights * torch.stack(level_losses)) / torch.sum(level_weights)
            if compute_structure
            else pred.new_zeros(())
        )

        if compute_structure and self.config.invariance_weight > 0.0:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                contrast_target = self._contrast_augment_mri(target.float())
                contrast_feats = model.extract_target_features(
                    contrast_target,
                    use_teacher=False,
                )
                contrast_projected = model.project_medrecl_target_features(
                    contrast_feats,
                    use_teacher=False,
                )
                invariance_loss = self._contrast_invariance_loss(
                    contrast_projected,
                    projected["mri_teacher"],
                )
            structure_loss = structure_loss + self.config.invariance_weight * invariance_loss
        else:
            invariance_loss = pred.new_zeros(())
        self._last_invariance_loss = float(invariance_loss.detach().item())
        self._last_soft_positive_similarity = (
            float(sum(self._soft_positive_values) / len(self._soft_positive_values))
            if self._soft_positive_values
            else 0.0
        )
        self._last_false_negative_ratio = (
            float(sum(self._false_negative_values) / len(self._false_negative_values))
            if self._false_negative_values
            else 0.0
        )

        if compute_appearance:
            # Local statistics and contextual distances are sensitive to FP16
            # rounding. Keep the backbone under AMP, but evaluate this small,
            # sampled auxiliary branch completely in FP32.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                appearance_loss = self._appearance_alignment_loss(
                    model=model,
                    projected_gen=[feature.float() for feature in projected["raw_gen"]],
                    projected_mri=[feature.float() for feature in projected["raw_mri_teacher"]],
                    level_contexts=level_contexts,
                    level_weights=appearance_level_weights.float(),
                    pred_image=pred.float(),
                    target_image=target.float(),
                )
        else:
            appearance_loss = pred.new_zeros(())
        if not compute_appearance:
            self._last_appearance_amplitude_loss = 0.0
            self._last_appearance_contextual_loss = 0.0
            self._last_appearance_stat_loss = 0.0

        return structure_loss, appearance_loss

    def _contrast_augment_mri(self, target: torch.Tensor) -> torch.Tensor:
        """Intensity-only MRI augmentation; spatial anatomy is unchanged."""
        cfg = self.config
        mask = (target > cfg.background_mu_threshold).float()
        batch = target.shape[0]
        shape = (batch, 1, 1, 1, 1)
        gamma = torch.empty(shape, device=target.device).uniform_(
            cfg.contrast_gamma_min,
            cfg.contrast_gamma_max,
        )
        scale = torch.empty(shape, device=target.device).uniform_(
            cfg.contrast_scale_min,
            cfg.contrast_scale_max,
        )
        bias = torch.empty(shape, device=target.device).uniform_(
            -cfg.contrast_bias,
            cfg.contrast_bias,
        )
        low_res_shape = tuple(max(2, size // 32) for size in target.shape[-3:])
        bias_field = torch.randn(
            (batch, 1, *low_res_shape),
            device=target.device,
            dtype=target.dtype,
        )
        bias_field = F.interpolate(
            bias_field,
            size=target.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        bias_field = bias_field / bias_field.flatten(1).std(dim=1).view(shape).clamp_min(1e-4)
        noise = torch.randn_like(target) * cfg.contrast_noise_std
        augmented = target.clamp(0.0, 1.0).pow(gamma)
        augmented = augmented * scale + bias + 0.03 * bias_field + noise
        return augmented.clamp(0.0, 1.0) * mask

    def _contrast_invariance_loss(
        self,
        online_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        weights = online_features[0].new_tensor(
            self.config.invariance_level_weights,
            dtype=torch.float32,
        )
        losses = []
        for online, teacher in zip(online_features, teacher_features):
            batch_losses = []
            for batch_index in range(online.shape[0]):
                online_flat = online[batch_index].float().flatten(1).transpose(0, 1)
                teacher_flat = teacher[batch_index].detach().float().flatten(1).transpose(0, 1)
                count = min(self.config.invariance_samples, online_flat.shape[0])
                if count <= 1:
                    continue
                indices = torch.randperm(online_flat.shape[0], device=online.device)[:count]
                query = online_flat[indices]
                key = teacher_flat[indices]
                logits = query @ key.transpose(0, 1)
                logits = logits / max(self.config.invariance_temperature, self.config.eps)
                labels = torch.arange(count, device=online.device)
                contrastive = F.cross_entropy(logits, labels)
                cosine = 1.0 - torch.sum(query * key, dim=1).mean()
                batch_losses.append(contrastive + 0.2 * cosine)
            losses.append(
                torch.stack(batch_losses).mean()
                if batch_losses
                else online.new_zeros((), dtype=torch.float32)
            )
        return torch.sum(weights * torch.stack(losses)) / weights.sum().clamp_min(self.config.eps)

    def _appearance_alignment_loss(
        self,
        model: "AttnResCTtoMRI",
        projected_gen: List[torch.Tensor],
        projected_mri: List[torch.Tensor],
        level_contexts: List[Dict[str, torch.Tensor]],
        level_weights: torch.Tensor,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        amplitude_losses: List[torch.Tensor] = []
        contextual_losses: List[torch.Tensor] = []
        stat_losses: List[torch.Tensor] = []
        for level, (gen_feat, mri_feat, context) in enumerate(zip(
            projected_gen,
            projected_mri,
            level_contexts,
        )):
            batch_losses: List[torch.Tensor] = []
            batch_amplitude_losses: List[torch.Tensor] = []
            batch_contextual_losses: List[torch.Tensor] = []
            batch_stat_losses: List[torch.Tensor] = []
            feature_shape = gen_feat.shape[-3:]
            pred_level = F.interpolate(
                pred_image.float(),
                size=feature_shape,
                mode="trilinear",
                align_corners=False,
            )
            target_level = F.interpolate(
                target_image.detach().float(),
                size=feature_shape,
                mode="trilinear",
                align_corners=False,
            )
            pred_mean, pred_std = self._local_mean_std(
                pred_level,
                kernel_size=self.config.stats_kernel,
            )
            target_mean, target_std = self._local_mean_std(
                target_level,
                kernel_size=self.config.stats_kernel,
            )
            for batch_index in range(gen_feat.shape[0]):
                indices = self._sample_appearance_indices(
                    context={k: v[batch_index] for k, v in context.items()},
                    max_samples=self.config.appearance_context_samples,
                )
                if indices.numel() == 0:
                    continue

                channels = gen_feat.shape[1]
                gen_vectors = gen_feat[batch_index].reshape(channels, -1).transpose(0, 1)
                mri_vectors = mri_feat[batch_index].reshape(channels, -1).transpose(0, 1)
                gen_vectors = model.project_medrecl_appearance_vectors(
                    gen_vectors[indices],
                    level=level,
                    use_teacher=False,
                )
                target_vectors = model.project_medrecl_appearance_vectors(
                    mri_vectors[indices].detach(),
                    level=level,
                    use_teacher=True,
                )

                amplitude_loss = F.smooth_l1_loss(gen_vectors, target_vectors)
                coordinates = self._unravel_indices(indices, feature_shape)
                contextual_loss = self._local_contextual_feature_loss(
                    gen_vectors,
                    target_vectors,
                    coordinates=coordinates,
                    radius=self.config.appearance_context_radii[level],
                    bandwidth=self.config.appearance_context_bandwidth,
                )

                pred_mean_vectors = (
                    pred_mean[batch_index].reshape(pred_mean.shape[1], -1).transpose(0, 1)[indices]
                )
                target_mean_vectors = (
                    target_mean[batch_index].reshape(target_mean.shape[1], -1).transpose(0, 1)[indices]
                )
                pred_std_vectors = (
                    pred_std[batch_index].reshape(pred_std.shape[1], -1).transpose(0, 1)[indices]
                )
                target_std_vectors = (
                    target_std[batch_index].reshape(target_std.shape[1], -1).transpose(0, 1)[indices]
                )
                stat_loss = F.smooth_l1_loss(
                    pred_mean_vectors,
                    target_mean_vectors,
                ) + self.config.appearance_stat_std_weight * F.smooth_l1_loss(
                    pred_std_vectors,
                    target_std_vectors,
                )
                batch_amplitude_losses.append(amplitude_loss)
                batch_contextual_losses.append(contextual_loss)
                batch_stat_losses.append(stat_loss)
                batch_losses.append(
                    self.config.appearance_amplitude_weight * amplitude_loss
                    + self.config.appearance_contextual_weight * contextual_loss
                    + self.config.appearance_stat_weight * stat_loss
                )

            if batch_losses:
                losses.append(torch.stack(batch_losses).mean())
                amplitude_losses.append(torch.stack(batch_amplitude_losses).mean())
                contextual_losses.append(torch.stack(batch_contextual_losses).mean())
                stat_losses.append(torch.stack(batch_stat_losses).mean())
            else:
                losses.append(gen_feat.new_zeros(()))
                amplitude_losses.append(gen_feat.new_zeros(()))
                contextual_losses.append(gen_feat.new_zeros(()))
                stat_losses.append(gen_feat.new_zeros(()))
        if not losses:
            self._last_appearance_amplitude_loss = 0.0
            self._last_appearance_contextual_loss = 0.0
            self._last_appearance_stat_loss = 0.0
            return level_weights.new_zeros(())
        weight_sum = torch.sum(level_weights)
        appearance_loss = torch.sum(level_weights * torch.stack(losses)) / weight_sum
        amplitude_loss = torch.sum(level_weights * torch.stack(amplitude_losses)) / weight_sum
        contextual_loss = torch.sum(level_weights * torch.stack(contextual_losses)) / weight_sum
        stat_loss = torch.sum(level_weights * torch.stack(stat_losses)) / weight_sum
        self._last_appearance_amplitude_loss = float(amplitude_loss.detach().item())
        self._last_appearance_contextual_loss = float(contextual_loss.detach().item())
        self._last_appearance_stat_loss = float(stat_loss.detach().item())
        return appearance_loss

    def _sample_appearance_indices(
        self,
        context: Dict[str, torch.Tensor],
        max_samples: int,
    ) -> torch.Tensor:
        foreground = ~context["background"].reshape(-1).bool()
        candidates = torch.nonzero(foreground, as_tuple=False).flatten()
        if candidates.numel() <= max_samples:
            return candidates

        grad = context["grad"].reshape(-1)
        error = context["error"].reshape(-1)
        boundary_count = max(1, int(round(max_samples * 0.30)))
        error_count = max(1, int(round(max_samples * 0.20)))
        random_count = max_samples - boundary_count - error_count

        boundary_order = torch.argsort(grad[candidates], descending=True)
        boundary = candidates[boundary_order[:boundary_count]]

        remaining_mask = ~torch.isin(candidates, boundary)
        remaining = candidates[remaining_mask]
        error_order = torch.argsort(error[remaining], descending=True)
        error_hard = remaining[error_order[:error_count]]

        remaining = remaining[~torch.isin(remaining, error_hard)]
        if random_count > 0 and remaining.numel() > 0:
            choice = torch.randperm(remaining.numel(), device=remaining.device)[
                : min(random_count, remaining.numel())
            ]
            random_foreground = remaining[choice]
        else:
            random_foreground = remaining[:0]
        selected = torch.cat([boundary, error_hard, random_foreground], dim=0)
        if selected.numel() < max_samples:
            selected = self._merge_unique_indices(selected, candidates, max_samples)
        return selected[:max_samples]

    def _local_contextual_feature_loss(
        self,
        pred_vectors: torch.Tensor,
        target_vectors: torch.Tensor,
        coordinates: torch.Tensor,
        radius: int,
        bandwidth: float,
    ) -> torch.Tensor:
        eps = self.config.eps
        target_center = target_vectors.mean(dim=0, keepdim=True)
        pred_norm = F.normalize(pred_vectors - target_center, dim=1, eps=eps)
        target_norm = F.normalize(target_vectors - target_center, dim=1, eps=eps)
        distance = torch.clamp(1.0 - pred_norm @ target_norm.transpose(0, 1), min=0.0)
        spatial_distance = torch.max(
            torch.abs(coordinates[:, None, :] - coordinates[None, :, :]),
            dim=-1,
        ).values
        local_mask = spatial_distance <= max(0, int(radius))
        masked_distance = distance.masked_fill(~local_mask, float("inf"))
        # Stop gradients through the adaptive distance scale. In homogeneous
        # regions the true minimum can be almost zero; differentiating through
        # that denominator creates very large gradients under AMP.
        row_min = masked_distance.amin(dim=1, keepdim=True).detach()
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.ones_like(row_min))
        row_min = row_min.clamp_min(max(eps, 1e-4))
        relative_distance = distance / row_min
        logits = (1.0 - relative_distance) / max(bandwidth, eps)
        logits = logits.masked_fill(~local_mask, torch.finfo(logits.dtype).min)
        contextual = torch.softmax(logits, dim=1)
        row_score = contextual.amax(dim=1).mean()
        column_score = contextual.amax(dim=0).mean()
        return -0.5 * (
            torch.log(row_score.clamp_min(eps))
            + torch.log(column_score.clamp_min(eps))
        )

    @staticmethod
    def _loss_gradient_norm(
        loss: torch.Tensor,
        parameters: List[torch.Tensor],
    ) -> float:
        if not loss.requires_grad or float(loss.detach().abs().item()) == 0.0:
            return 0.0
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        squared_norms = [
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not squared_norms:
            return 0.0
        total_norm = torch.sqrt(torch.stack(squared_norms).sum())
        if not torch.isfinite(total_norm):
            return float("inf")
        return float(total_norm.item())

    def _update_gradient_scales(
        self,
        model: "AttnResCTtoMRI",
        rec_loss: torch.Tensor,
        structure_term: torch.Tensor,
        appearance_term: torch.Tensor,
        current_step: int,
    ) -> None:
        interval = max(1, int(self.config.gradient_balance_interval))
        should_update = (
            self._last_balance_step < 0
            or current_step - self._last_balance_step >= interval
        )
        if not should_update or not torch.is_grad_enabled() or not model.training:
            return

        balance_parameters = [
            next(parameter for parameter in block.parameters() if parameter.requires_grad)
            for block in (model.dec1, model.dec2, model.dec3)
        ]
        rec_norm = self._loss_gradient_norm(rec_loss, balance_parameters)
        structure_norm = self._loss_gradient_norm(structure_term, balance_parameters)
        appearance_norm = self._loss_gradient_norm(appearance_term, balance_parameters)
        eps = self.config.eps

        if not math.isfinite(rec_norm) or rec_norm <= eps:
            self._structure_gradient_scale = 0.0
            self._appearance_gradient_scale = 0.0
            self._structure_gradient_ratio = float("inf")
            self._appearance_gradient_ratio = float("inf")
            self._last_balance_step = current_step
            return

        structure_ratio = structure_norm / rec_norm
        appearance_ratio = appearance_norm / rec_norm
        structure_ratio = structure_ratio if math.isfinite(structure_ratio) else float("inf")
        appearance_ratio = appearance_ratio if math.isfinite(appearance_ratio) else float("inf")
        structure_scale = min(
            1.0,
            self.config.structure_cap_ratio / max(structure_ratio, eps)
            if math.isfinite(structure_ratio)
            else 0.0,
        )
        appearance_scale = min(
            1.0,
            self.config.appearance_cap_ratio / max(appearance_ratio, eps)
            if math.isfinite(appearance_ratio)
            else 0.0,
        )
        bounded_total = (
            structure_ratio * structure_scale
            + appearance_ratio * appearance_scale
        )
        if bounded_total > self.config.contrast_max_ratio:
            common_scale = self.config.contrast_max_ratio / max(bounded_total, eps)
            structure_scale *= common_scale
            appearance_scale *= common_scale

        self._structure_gradient_scale = structure_scale
        self._appearance_gradient_scale = appearance_scale
        self._structure_gradient_ratio = structure_ratio
        self._appearance_gradient_ratio = appearance_ratio
        self._last_balance_step = current_step

    def weighted_loss(
        self,
        model: "AttnResCTtoMRI",
        source: torch.Tensor,
        target: torch.Tensor,
        pred: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
        rec_loss: torch.Tensor,
        current_step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        structure_weight, appearance_weight = self.progressive_weights(
            current_step=current_step,
            total_steps=total_steps,
        )
        structure_loss, appearance_loss = self.forward_components(
            model=model,
            source=source,
            target=target,
            pred=pred,
            feature_dict=feature_dict,
            current_step=current_step,
            total_steps=total_steps,
            compute_structure=structure_weight > 0.0,
            compute_appearance=appearance_weight > 0.0,
        )
        structure_term = structure_loss * structure_weight
        appearance_term = appearance_loss * appearance_weight
        self._update_gradient_scales(
            model=model,
            rec_loss=rec_loss,
            structure_term=structure_term,
            appearance_term=appearance_term,
            current_step=current_step,
        )
        structure_term = structure_term * self._structure_gradient_scale
        appearance_term = appearance_term * self._appearance_gradient_scale
        weighted = structure_term + appearance_term
        cap_ratio = self.contrast_ratio_cap(
            current_step=current_step,
            total_steps=total_steps,
        )
        raw_loss = structure_loss + appearance_loss
        effective_weight = float((weighted.detach() / raw_loss.detach().clamp_min(self.config.eps)).item())
        metrics = {
            "raw_loss": float(raw_loss.detach().item()),
            "structure_loss": float(structure_loss.detach().item()),
            "appearance_loss": float(appearance_loss.detach().item()),
            "appearance_amplitude_loss": self._last_appearance_amplitude_loss,
            "appearance_contextual_loss": self._last_appearance_contextual_loss,
            "appearance_stat_loss": self._last_appearance_stat_loss,
            "soft_positive_similarity": self._last_soft_positive_similarity,
            "false_negative_ratio": self._last_false_negative_ratio,
            "invariance_loss": self._last_invariance_loss,
            "structure_weight": structure_weight,
            "appearance_weight": appearance_weight,
            "effective_weight": effective_weight,
            "weighted_loss": float(weighted.detach().item()),
            "structure_weighted_loss": float(structure_term.detach().item()),
            "appearance_weighted_loss": float(appearance_term.detach().item()),
            "cap_ratio": cap_ratio,
            "structure_gradient_scale": self._structure_gradient_scale,
            "appearance_gradient_scale": self._appearance_gradient_scale,
            "structure_gradient_ratio": self._structure_gradient_ratio,
            "appearance_gradient_ratio": self._appearance_gradient_ratio,
        }
        return weighted, metrics, raw_loss

    def forward(
        self,
        model: "AttnResCTtoMRI",
        source: torch.Tensor,
        target: torch.Tensor,
        pred: torch.Tensor,
        feature_dict: Dict[str, List[torch.Tensor]],
        current_step: int,
        total_steps: int,
    ) -> torch.Tensor:
        structure_loss, appearance_loss = self.forward_components(
            model=model,
            source=source,
            target=target,
            pred=pred,
            feature_dict=feature_dict,
            current_step=current_step,
            total_steps=total_steps,
        )
        return structure_loss + appearance_loss

    def _level_loss(
        self,
        zg: torch.Tensor,
        zx: torch.Tensor,
        zy: torch.Tensor,
        context: Dict[str, torch.Tensor],
        level: int,
        gamma_t: float,
    ) -> torch.Tensor:
        batch_losses: List[torch.Tensor] = []
        radius = self.config.exclusion_radii[level]

        for b in range(zg.shape[0]):
            batch_losses.append(
                self._single_case_level_loss(
                    zg=zg[b],
                    zx=zx[b],
                    zy=zy[b],
                    context={k: v[b] for k, v in context.items()},
                    level=level,
                    radius=radius,
                    gamma_t=gamma_t,
                )
            )

        return torch.stack(batch_losses).mean()

    def _single_case_level_loss(
        self,
        zg: torch.Tensor,
        zx: torch.Tensor,
        zy: torch.Tensor,
        context: Dict[str, torch.Tensor],
        level: int,
        radius: int,
        gamma_t: float,
    ) -> torch.Tensor:
        cfg = self.config
        c, d, h, w = zg.shape
        num_positions = d * h * w

        zg_flat = zg.reshape(c, num_positions).transpose(0, 1)
        zx_flat = zx.reshape(c, num_positions).transpose(0, 1)
        zy_flat = zy.reshape(c, num_positions).transpose(0, 1)

        grad_flat = context["grad"].reshape(-1)
        err_flat = context["error"].reshape(-1)
        mu_y_flat = context["mu_y"].reshape(-1)
        bg_flat = context["background"].reshape(-1).bool()
        grad_q80 = context["grad_q80"].reshape(()).to(zg_flat.dtype)
        err_q80 = context["error_q80"].reshape(()).to(zg_flat.dtype)

        anchor_idx = self._sample_anchor_indices(
            score_flat=grad_flat + err_flat,
            background_flat=bg_flat,
            num_positions=num_positions,
            max_samples=cfg.anchor_samples,
        )
        if anchor_idx.numel() == 0:
            return zg_flat.new_zeros(())

        candidate_idx = self._sample_candidate_pool(
            background_flat=bg_flat,
            num_positions=num_positions,
            max_samples=min(cfg.candidate_pool_size, num_positions),
        )
        if candidate_idx.numel() == 0:
            return zg_flat.new_zeros(())

        # Similarity logits and exponentials are sensitive to FP16 overflow.
        # Sampling keeps this FP32 matrix bounded in memory.
        anchor_feat = zg_flat[anchor_idx].float()
        cand_feat = zy_flat[candidate_idx].float()
        sim_matrix = anchor_feat @ cand_feat.transpose(0, 1)
        dist_matrix = 1.0 - sim_matrix

        anchor_coords = self._unravel_indices(anchor_idx, (d, h, w))
        cand_coords = self._unravel_indices(candidate_idx, (d, h, w))
        chebyshev = torch.max(
            torch.abs(anchor_coords[:, None, :] - cand_coords[None, :, :]),
            dim=-1,
        ).values
        valid_mask = chebyshev > radius

        anchor_grad = grad_flat[anchor_idx]
        anchor_err = err_flat[anchor_idx]
        cand_grad = grad_flat[candidate_idx]
        cand_err = err_flat[candidate_idx]
        cand_mu_y = mu_y_flat[candidate_idx]
        cand_bg = bg_flat[candidate_idx]

        tau = torch.clamp(
            cfg.tau0 / (1.0 + cfg.kappa_g * anchor_grad + cfg.kappa_r * anchor_err),
            min=cfg.tau_min,
            max=cfg.tau_max,
        )
        omega = torch.clamp(
            1.0 + cfg.beta_g * anchor_grad + cfg.beta_r * anchor_err,
            min=1.0,
            max=3.0,
        )

        sim_pos_y = self._local_soft_positive_similarity(
            anchor_feat=anchor_feat,
            target_flat=zy_flat.float(),
            anchor_coords=anchor_coords,
            feature_shape=(d, h, w),
            level=level,
        )
        sim_pos_x = torch.sum(anchor_feat * zx_flat[anchor_idx].float(), dim=-1)
        ct_positive_weight = cfg.lambda_x if level == len(cfg.level_weights) - 1 else 0.0
        pos_y_logit = sim_pos_y / tau
        pos_x_logit = sim_pos_x / tau
        if ct_positive_weight > 0.0:
            pos_logits = torch.stack(
                [
                    pos_y_logit,
                    pos_x_logit + math.log(max(float(ct_positive_weight), cfg.eps)),
                ],
                dim=1,
            )
            log_positive = torch.logsumexp(pos_logits, dim=1)
        else:
            log_positive = pos_y_logit

        hard_base = valid_mask & (dist_matrix > cfg.ignore_distance) & (dist_matrix <= cfg.hard_distance)
        relaxed_hard_base = (
            valid_mask
            & (dist_matrix > cfg.ignore_distance)
            & (dist_matrix <= cfg.error_hard_distance)
        )
        normal_mask = valid_mask & (dist_matrix > cfg.hard_distance) & (dist_matrix <= cfg.normal_distance)
        easy_mask = valid_mask & (dist_matrix > cfg.normal_distance)
        background_mask = valid_mask & cand_bg.unsqueeze(0) & (dist_matrix > cfg.ignore_distance)

        mu_diff = torch.abs(mu_y_flat[anchor_idx][:, None] - cand_mu_y[None, :])
        boundary_mask = (
            hard_base
            & (cand_grad.unsqueeze(0) > grad_q80)
            & (mu_diff < cfg.delta_mu)
        )
        boundary_relaxed = (
            relaxed_hard_base
            & (cand_grad.unsqueeze(0) > grad_q80)
            & (mu_diff < cfg.delta_mu)
        )
        error_mask = relaxed_hard_base & (cand_err.unsqueeze(0) > err_q80)

        alpha_matrix = torch.clamp(
            1.0
            + cfg.rho_s * torch.clamp_min(sim_matrix, 0.0)
            + cfg.rho_g * (anchor_grad[:, None] + cand_grad[None, :])
            + cfg.rho_r * (anchor_err[:, None] + cand_err[None, :]),
            min=1.0,
            max=4.0,
        )
        false_negative_mask = valid_mask & (
            sim_matrix > cfg.false_negative_threshold
        )
        valid_count = valid_mask.float().sum().clamp_min(1.0)
        false_negative_ratio = false_negative_mask.float().sum() / valid_count
        self._soft_positive_values.append(float(sim_pos_y.detach().mean().item()))
        self._false_negative_values.append(float(false_negative_ratio.detach().item()))

        losses: List[torch.Tensor] = []
        for row in range(anchor_idx.numel()):
            sim_row = sim_matrix[row]
            alpha_row = alpha_matrix[row]
            row_neg_logits: List[torch.Tensor] = []

            def make_negative_logits(
                indices: torch.Tensor,
                base_weight: float,
                adaptive_weight: Optional[torch.Tensor] = None,
            ) -> Optional[torch.Tensor]:
                if indices.numel() == 0:
                    return None
                weights = torch.full_like(
                    sim_row[indices],
                    fill_value=max(float(base_weight), cfg.eps),
                    dtype=torch.float32,
                )
                if adaptive_weight is not None:
                    weights = weights * adaptive_weight[indices].float().clamp_min(cfg.eps)
                fn = false_negative_mask[row, indices]
                weights = torch.where(
                    fn,
                    weights * cfg.false_negative_weight,
                    weights,
                )
                return sim_row[indices] / tau[row] + torch.log(weights.clamp_min(cfg.eps))

            normal_idx = self._sample_random_mask(
                normal_mask[row], cfg.normal_negative_samples
            )
            easy_idx = self._sample_random_mask(
                easy_mask[row], cfg.easy_background_samples
            )
            background_idx = self._sample_random_mask(
                background_mask[row], cfg.easy_background_samples
            )

            boundary_idx = self._sample_hard_indices(
                primary_mask=boundary_mask[row],
                relaxed_mask=boundary_relaxed[row],
                normal_mask=normal_mask[row],
                sim_row=sim_row,
                target_count=cfg.hard_negative_samples,
            )
            if cfg.lambda_R > 0.0:
                error_idx = self._sample_hard_indices(
                    primary_mask=error_mask[row],
                    relaxed_mask=error_mask[row],
                    normal_mask=normal_mask[row],
                    sim_row=sim_row,
                    target_count=cfg.hard_negative_samples,
                )
            else:
                error_idx = anchor_idx.new_empty((0,))

            if normal_idx.numel() > 0:
                logits = make_negative_logits(normal_idx, gamma_t)
                if logits is not None:
                    row_neg_logits.append(logits)
            if easy_idx.numel() > 0:
                logits = make_negative_logits(easy_idx, cfg.lambda_easy)
                if logits is not None:
                    row_neg_logits.append(logits)
            if background_idx.numel() > 0:
                logits = make_negative_logits(background_idx, cfg.lambda_A)
                if logits is not None:
                    row_neg_logits.append(logits)
            if boundary_idx.numel() > 0:
                logits = make_negative_logits(boundary_idx, cfg.lambda_B, alpha_row)
                if logits is not None:
                    row_neg_logits.append(logits)
            if error_idx.numel() > 0:
                logits = make_negative_logits(error_idx, cfg.lambda_R, alpha_row)
                if logits is not None:
                    row_neg_logits.append(logits)

            if row_neg_logits:
                log_denominator = torch.logsumexp(
                    torch.cat([log_positive[row].view(1), *row_neg_logits], dim=0),
                    dim=0,
                )
            else:
                log_denominator = log_positive[row]
            losses.append(omega[row] * (log_denominator - log_positive[row]))

        contrast_loss = torch.stack(losses).mean()
        align_loss = torch.mean(omega * (1.0 - sim_pos_y))
        return contrast_loss + cfg.align_weight * align_loss

    def _local_soft_positive_similarity(
        self,
        anchor_feat: torch.Tensor,
        target_flat: torch.Tensor,
        anchor_coords: torch.Tensor,
        feature_shape: Tuple[int, int, int],
        level: int,
    ) -> torch.Tensor:
        """Match locally while retaining the registered same-voxel correspondence."""
        cfg = self.config
        radius = cfg.soft_positive_radii[min(level, len(cfg.soft_positive_radii) - 1)]
        rd, rh, rw = (max(0, int(value)) for value in radius)
        ranges = [
            torch.arange(-radius, radius + 1, device=anchor_feat.device)
            for radius in (rd, rh, rw)
        ]
        offsets = torch.cartesian_prod(*ranges)
        if offsets.ndim == 1:
            offsets = offsets.view(-1, 3)
        positive_coords = anchor_coords[:, None, :] + offsets[None, :, :]
        shape = anchor_coords.new_tensor(feature_shape)
        valid = ((positive_coords >= 0) & (positive_coords < shape)).all(dim=-1)
        safe_coords = torch.minimum(
            torch.maximum(positive_coords, torch.zeros_like(positive_coords)),
            shape.view(1, 1, 3) - 1,
        )
        flat_indices = (
            safe_coords[..., 0] * feature_shape[1] * feature_shape[2]
            + safe_coords[..., 1] * feature_shape[2]
            + safe_coords[..., 2]
        ).long()
        positive_feat = target_flat[flat_indices]
        similarities = torch.sum(anchor_feat[:, None, :] * positive_feat, dim=-1)
        spatial_distance = offsets.float().abs().amax(dim=1)
        scores = similarities - cfg.soft_positive_spatial_eta * spatial_distance[None, :]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        center_mask = (offsets == 0).all(dim=1)
        center_prior = max(0.0, min(1.0, float(cfg.soft_positive_center_prior)))
        if center_prior > 0.0 and bool(center_mask.any()) and offsets.shape[0] > 1:
            weights = (1.0 - center_prior) * weights
            weights[:, center_mask] = weights[:, center_mask] + center_prior
        return torch.sum(weights * similarities, dim=1)

    def _build_level_context(
        self,
        source_aux: torch.Tensor,
        target_aux: torch.Tensor,
        grad_map: torch.Tensor,
        error_map: torch.Tensor,
        feature_shape: Tuple[int, int, int],
    ) -> Dict[str, torch.Tensor]:
        mode = "trilinear"
        target_l = F.interpolate(target_aux, size=feature_shape, mode=mode, align_corners=False)
        source_l = F.interpolate(source_aux, size=feature_shape, mode=mode, align_corners=False)
        grad_l = F.interpolate(grad_map, size=feature_shape, mode=mode, align_corners=False)
        error_l = F.interpolate(error_map, size=feature_shape, mode=mode, align_corners=False)

        mu_y, _ = self._local_mean_std(target_l, kernel_size=self.config.stats_kernel)
        mu_x, _ = self._local_mean_std(source_l, kernel_size=self.config.stats_kernel)
        background = (
            (mu_y < self.config.background_mu_threshold)
            & (mu_x < self.config.background_mu_threshold)
            & (grad_l < self.config.background_grad_threshold)
        )
        grad_q80 = torch.quantile(
            grad_l.flatten(1),
            q=self.config.boundary_quantile,
            dim=1,
        )
        error_q80 = torch.quantile(
            error_l.flatten(1),
            q=self.config.error_quantile,
            dim=1,
        )
        return {
            "grad": grad_l,
            "error": error_l,
            "mu_y": mu_y,
            "background": background,
            "grad_q80": grad_q80,
            "error_q80": error_q80,
        }

    def _gamma_t(self, current_step: int, total_steps: int) -> float:
        progress = min(max(float(current_step) / float(max(total_steps, 1)), 0.0), 1.0)
        return self.config.gamma_min + (self.config.gamma_max - self.config.gamma_min) * ((1.0 - progress) ** 2)

    @staticmethod
    def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        dz = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return torch.sqrt(dz.pow(2) + dy.pow(2) + dx.pow(2) + 1e-12)

    @staticmethod
    def _minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.amin(dim=(2, 3, 4), keepdim=True)
        x_max = x.amax(dim=(2, 3, 4), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    @staticmethod
    def _local_mean_std(
        x: torch.Tensor,
        kernel_size: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding = kernel_size // 2
        mean = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool3d(x * x, kernel_size=kernel_size, stride=1, padding=padding)
        variance = torch.clamp(sq_mean - mean * mean, min=0.0)
        # sqrt'(0) is infinite. A small variance floor keeps gradients finite in
        # flat background/tissue regions without changing relative statistics.
        std = torch.sqrt(variance + 1e-6)
        return mean, std

    @staticmethod
    def _unravel_indices(indices: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        h, w = shape[1], shape[2]
        d_idx = torch.div(indices, h * w, rounding_mode="floor")
        rem = torch.remainder(indices, h * w)
        h_idx = torch.div(rem, w, rounding_mode="floor")
        w_idx = torch.remainder(rem, w)
        return torch.stack([d_idx, h_idx, w_idx], dim=-1)

    @staticmethod
    def _sample_random_indices(indices: torch.Tensor, max_samples: int) -> torch.Tensor:
        if indices.numel() <= max_samples:
            return indices
        choice = torch.randperm(indices.numel(), device=indices.device)[:max_samples]
        return indices[choice]

    @staticmethod
    def _sample_top_indices(
        indices: torch.Tensor,
        values: torch.Tensor,
        max_samples: int,
    ) -> torch.Tensor:
        if indices.numel() <= max_samples:
            return indices
        order = torch.argsort(values[indices], descending=True)
        return indices[order[:max_samples]]

    def _sample_random_mask(self, mask_row: torch.Tensor, max_samples: int) -> torch.Tensor:
        indices = torch.nonzero(mask_row, as_tuple=False).flatten()
        if indices.numel() == 0:
            return indices
        return self._sample_random_indices(indices, max_samples)

    def _sample_top_mask(
        self,
        mask_row: torch.Tensor,
        values: torch.Tensor,
        max_samples: int,
    ) -> torch.Tensor:
        indices = torch.nonzero(mask_row, as_tuple=False).flatten()
        if indices.numel() == 0:
            return indices
        return self._sample_top_indices(indices, values, max_samples)

    @staticmethod
    def _merge_unique_indices(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        max_samples: int,
    ) -> torch.Tensor:
        if primary.numel() >= max_samples:
            return primary[:max_samples]
        if secondary.numel() == 0:
            return primary
        if primary.numel() == 0:
            return secondary[:max_samples]

        mask = ~torch.isin(secondary, primary)
        merged = torch.cat([primary, secondary[mask]], dim=0)
        return merged[:max_samples]

    def _sample_hard_indices(
        self,
        primary_mask: torch.Tensor,
        relaxed_mask: torch.Tensor,
        normal_mask: torch.Tensor,
        sim_row: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        selected = self._sample_top_mask(primary_mask, sim_row, target_count)
        if selected.numel() < target_count:
            relaxed = self._sample_top_mask(relaxed_mask, sim_row, target_count)
            selected = self._merge_unique_indices(selected, relaxed, target_count)
        if selected.numel() < target_count:
            normal = self._sample_top_mask(normal_mask, sim_row, target_count)
            selected = self._merge_unique_indices(selected, normal, target_count)
        return selected

    def _sample_anchor_indices(
        self,
        score_flat: torch.Tensor,
        background_flat: torch.Tensor,
        num_positions: int,
        max_samples: int,
    ) -> torch.Tensor:
        valid = torch.nonzero(~background_flat, as_tuple=False).flatten()
        if valid.numel() == 0:
            valid = torch.arange(num_positions, device=score_flat.device)
        if valid.numel() <= max_samples:
            return valid

        top_count = min(max_samples // 2, valid.numel())
        top_idx = self._sample_top_indices(valid, score_flat, top_count)
        remaining_mask = torch.ones(valid.numel(), device=valid.device, dtype=torch.bool)
        if top_idx.numel() > 0:
            remaining_mask &= ~torch.isin(valid, top_idx)
        remaining = valid[remaining_mask]
        rand_count = max_samples - top_idx.numel()
        rand_idx = self._sample_random_indices(remaining, rand_count)
        if rand_idx.numel() < rand_count:
            supplement = self._sample_random_indices(valid, rand_count)
            rand_idx = self._merge_unique_indices(rand_idx, supplement, rand_count)
        return torch.cat([top_idx, rand_idx], dim=0)[:max_samples]

    def _sample_candidate_pool(
        self,
        background_flat: torch.Tensor,
        num_positions: int,
        max_samples: int,
    ) -> torch.Tensor:
        all_indices = torch.arange(num_positions, device=background_flat.device)
        if num_positions <= max_samples:
            return all_indices

        non_bg = torch.nonzero(~background_flat, as_tuple=False).flatten()
        bg = torch.nonzero(background_flat, as_tuple=False).flatten()
        non_bg_quota = min(non_bg.numel(), int(max_samples * 0.75))
        bg_quota = min(bg.numel(), max_samples - non_bg_quota)

        selected = []
        if non_bg_quota > 0:
            selected.append(self._sample_random_indices(non_bg, non_bg_quota))
        if bg_quota > 0:
            selected.append(self._sample_random_indices(bg, bg_quota))

        candidate_idx = torch.cat(selected, dim=0) if selected else all_indices[:0]
        if candidate_idx.numel() < max_samples:
            supplement = self._sample_random_indices(all_indices, max_samples)
            candidate_idx = self._merge_unique_indices(candidate_idx, supplement, max_samples)
        return candidate_idx[:max_samples]

#两个函数都加了 @torch.no_grad()：关闭梯度计算，不占用显存、不更新网络，只在验证集 / 测试集算效果看模型好坏。
#特征 2：没有 @torch.no_grad() 装饰器。整个类没有加这个装饰，前向过程每一步运算都会追踪梯度。
#【另一类是训练/验证时拿来"看效果"的评价指标】
@torch.no_grad()     #告诉 PyTorch：下面这个函数只是拿来"算结果"的，不需要记录梯度
def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算 MAE（Mean Absolute Error，平均绝对误差）。

    参数:
        pred: 预测值
        target: 真实值

    返回:
        标量张量，表示平均绝对误差
    """
    return torch.mean(torch.abs(pred - target))


@torch.no_grad()    #这个函数只是评估用，不需要梯度【明确提示它们是"评估用"】
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:  # 先计算均方误差 MSE
    """
    计算 PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）。

    参数:
        pred: 预测值
        target: 真实值
        data_range: 数据范围，若数据在 [0,1]，则取 1.0
    返回:
        PSNR 值，单位 dB，越大表示重建质量越好
    """
    # 情况1：两张图几乎完全一样，误差极小
    mse = torch.mean((pred - target) ** 2)             #MSE（均方误差）
    if mse <= 1e-12:                                   #若MSE 极小，说明几乎完全一致，直接返回一个较大的PSNR值。  #两张图几乎完全一致，避免log10(0)无穷大报错。
        return torch.tensor(99.0, device=pred.device)  # 直接返回满分99，避开无穷大bug  
    # 情况2：存在明显误差，正常套PSNR公式计算分数
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse) # PSNR 公式: # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)


@torch.no_grad()
def compute_masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MAE only inside a foreground mask."""
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if not torch.any(mask):
        return compute_mae(pred, target)
    return torch.mean(torch.abs(pred[mask] - target[mask]))


@torch.no_grad()
def compute_masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute PSNR only inside a foreground mask."""
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if not torch.any(mask):
        return compute_psnr(pred, target, data_range=data_range)
    mse = torch.mean((pred[mask] - target[mask]).square())
    if mse <= 1e-12:
        return pred.new_tensor(99.0)
    return (
        20.0 * torch.log10(pred.new_tensor(float(data_range)))
        - 10.0 * torch.log10(mse)
    )


def _gradient_magnitude_3d(x: torch.Tensor) -> torch.Tensor:
    """Finite-difference 3D gradient magnitude with shape preserved."""
    dz = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    return torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-12)


@torch.no_grad()
def compute_gradient_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MAE between 3D gradient magnitudes; lower means sharper matching detail."""
    difference = torch.abs(
        _gradient_magnitude_3d(pred.float())
        - _gradient_magnitude_3d(target.float())
    )
    if mask is not None:
        mask = mask.to(device=difference.device, dtype=torch.bool)
        if torch.any(mask):
            return difference[mask].mean()
    return difference.mean()


def _gaussian_blur_3d_separable(
    x: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Memory-efficient separable 3D Gaussian blur used by HFEN."""
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords = coords - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    channels = x.shape[1]
    padding = kernel_size // 2
    kernels = (
        kernel.view(1, 1, kernel_size, 1, 1),
        kernel.view(1, 1, 1, kernel_size, 1),
        kernel.view(1, 1, 1, 1, kernel_size),
    )
    out = x
    paddings = (
        (padding, 0, 0),
        (0, padding, 0),
        (0, 0, padding),
    )
    for axis_kernel, pad in zip(kernels, paddings):
        weight = axis_kernel.repeat(channels, 1, 1, 1, 1)
        out = F.conv3d(out, weight, padding=pad, groups=channels)
    return out


def _laplacian_3d(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    kernel = x.new_zeros((channels, 1, 3, 3, 3))
    kernel[:, 0, 1, 1, 1] = -6.0
    kernel[:, 0, 0, 1, 1] = 1.0
    kernel[:, 0, 2, 1, 1] = 1.0
    kernel[:, 0, 1, 0, 1] = 1.0
    kernel[:, 0, 1, 2, 1] = 1.0
    kernel[:, 0, 1, 1, 0] = 1.0
    kernel[:, 0, 1, 1, 2] = 1.0
    return F.conv3d(x, kernel, padding=1, groups=channels)


@torch.no_grad()
def compute_hfen(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Normalized high-frequency error norm based on a 3D LoG response.

    Lower is better. The normalization makes values comparable across cases with
    different MRI signal amplitudes.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred_log = _laplacian_3d(_gaussian_blur_3d_separable(pred.float()))
        target_log = _laplacian_3d(_gaussian_blur_3d_separable(target.float()))
        difference_sq = (pred_log - target_log).square()
        target_sq = target_log.square()
        if mask is not None:
            mask = mask.to(device=pred.device, dtype=torch.bool)
            if torch.any(mask):
                difference_sq = difference_sq[mask]
                target_sq = target_sq[mask]
        numerator = torch.sqrt(difference_sq.sum().clamp_min(0.0))
        denominator = torch.sqrt(target_sq.sum().clamp_min(1e-12))
        return numerator / denominator.clamp_min(1e-12)


@torch.no_grad()
def compute_pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pearson correlation with a finite zero fallback for constant maps."""
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator <= 1e-12:
        return x.new_tensor(0.0)
    return torch.clamp(torch.dot(x, y) / denominator, -1.0, 1.0)


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """NumPy rankdata equivalent using average ranks for tied values."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    average_ranks = (starts + cumulative - 1).astype(np.float64) / 2.0
    return average_ranks[inverse]


@torch.no_grad()
def compute_spearman_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int = 200_000,
) -> torch.Tensor:
    """Spearman correlation with deterministic sampling for large 3D maps."""
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)
    if x.numel() > max_samples:
        indices = torch.linspace(
            0,
            x.numel() - 1,
            steps=max_samples,
            device=x.device,
        ).round().long()
        x = x[indices]
        y = y[indices]
    x_rank = torch.from_numpy(_average_tie_ranks(x.cpu().numpy())).to(
        device=x.device,
        dtype=torch.float32,
    )
    y_rank = torch.from_numpy(_average_tie_ranks(y.cpu().numpy())).to(
        device=y.device,
        dtype=torch.float32,
    )
    return compute_pearson_correlation(x_rank, y_rank)


@torch.no_grad()
def compute_top_quantile_overlap(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    quantile: float = 0.90,
) -> torch.Tensor:
    """Fractional overlap between equally sized high-uncertainty/high-error sets."""
    uncertainty = uncertainty.detach().float().reshape(-1)
    error = error.detach().float().reshape(-1)
    if uncertainty.numel() == 0 or error.numel() == 0:
        return uncertainty.new_tensor(0.0)
    if uncertainty.std(unbiased=False) <= 1e-12 or error.std(unbiased=False) <= 1e-12:
        return uncertainty.new_tensor(0.0)
    count = max(1, int(round((1.0 - quantile) * uncertainty.numel())))
    uncertainty_idx = torch.topk(uncertainty, k=count, sorted=False).indices
    error_idx = torch.topk(error, k=count, sorted=False).indices
    selected = torch.zeros(
        uncertainty.numel(),
        device=uncertainty.device,
        dtype=torch.bool,
    )
    selected[uncertainty_idx] = True
    intersection = selected[error_idx].sum()
    return intersection.float() / float(count)


# =======================================================================================================================================================
# Moment Propagation Utilities —— 模块 1：基础矩传播工具函数
# 实现论文 Table 1 中所有层的精确矩传播公式
# 包含：GELU一阶泰勒导数、Softmax二阶泰勒展开、Dropout矩生成
# =======================================================================================================================================================
#类（nn.Module）：带缓存、带梯度、可当网络 / 损失、能存参数、参与训练；独立 def 函数：无缓存、无梯度、纯临时计算、只做验证评估、用完就销毁。


def gelu_with_derivative(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:   # 输入张量 x，同时输出 GELU 激活值 + GELU 一阶导；矩传播、梯度反向传播、海森矩阵计算时需要激活函数导数。
    """
    GELU激活函数及其一阶导数（精确形式，基于erf）。
    GELU(x) = x * Φ(x)    Φ是标准正态累积分布CDF
    GELU'(x) = Φ(x) + x * φ(x)    φ是标准正态概率密度PDF
    参数: x: 任意形状输入张量
    返回: (gelu_val激活输出, dgelu一阶导数)，和x同shape
    """
    sqrt2 = math.sqrt(2.0)             #根号下2                                    # 普通nn.GELU只能得到输出，拿不到解析导数；矩传播计算方差 / 协方差必须用到激活函数一阶导，所以单独封装同时返回值和导数。
    # 1. 计算正态累积分布 Φ(x) = 0.5 * (1 + erf(x/√2))
    phi = 0.5 * (1.0 + torch.erf(x / sqrt2))
    # torch.erf：误差函数，用来构造正态分布积分CDF
    # 2. 标准正态概率密度 φ(x) = 1/√(2π) * exp(-x²/2)
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    # exp(-0.5x²) 正态分布核心，除以归一化常数
    # 3. 计算GELU前向输出
    gelu_val = x * phi
    # 4. GELU解析一阶导数
    dgelu = phi + x * pdf
    return gelu_val, dgelu


def softmax_2nd_order_moments(                        #输入 logits 的均值mu_z、协方差矩阵Sigma_z，通过二阶泰勒展开，输出经过softmax 后的分布均值（带二阶修正）、一阶传播协方差。用于分类 / 多通道特征不确定性传播，解决 softmax 非线性无法直接线性传递协方差的问题。
    mu_z: torch.Tensor,
    Sigma_z: torch.Tensor,                            #普通一阶矩传播只使用雅可比，均值有偏差；加入二阶泰勒修正，分布均值估计更精准，适合不确定性量化任务。
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax二阶泰勒展开矩传播。
    对 scores = softmax(z) 二阶泰勒近似：
      - 均值 μ_s 包含一阶+二阶修正项（更精准）
      - 协方差 Σ_s 使用一阶雅可比矩阵线性传播
    数学：
      雅可比 J_{i,j} = s_i * (δ_{ij} - s_j)
      二阶均值修正 Δμ_i = 0.5 * s_i * [Σ_{ii} - 2(Σ@s)_i + 2*s^T@Σ@s - Σ_j s_j*Σ_{jj}]
      协方差传播公式:  Σ_s = J @ Σ_z @ J^T
    参数:
        mu_z: [B, N] 每个batch的logits均值，N通道数
        Sigma_z: [B, N, N] 每个batch logits的协方差矩阵
    返回:
        mu_s: [B, N] 二阶修正后的softmax均值
        Sigma_s: [B, N, N] 雅可比传播后的softmax协方差    
    """
    B, N = mu_z.shape     # 解包batch、通道数量

    # 1. 一阶基础均值：先把均值logits直接过softmax得到s
    s = F.softmax(mu_z, dim=-1)  # [B, N] #dim=-1指的是N对通道进行  

    # ---------------------- 第一步：雅可比矩阵，传播协方差 ----------------------
    # J_{i,j} = s_i * (δ_ij - s_j)
    # δ_ij 单位对角矩阵：i=j为1，其余0
    eye = torch.eye(N, device=mu_z.device, dtype=mu_z.dtype)  # [N, N]  #eye = torch.eye(N)生成 N 阶单位对角矩阵，对应克罗内克函数δij ：对角线全 1，其余 0，shape[N,N]。整句 dtype=mu_z.dtype 作用：同理，如果不指定dtype，torch.eye()默认生成float32。假如你的网络用半精度float16训练，mu_z是 float16，而eye是 float32，不同精度张量无法做四则运算，直接报错。
    # 广播构造批量雅可比矩阵 [B,N,N]
    # s.unsqueeze(-1) → [B,N,1]；eye.unsqueeze(0)-s.unsqueeze(-2) → [B,N,N]
    J = s.unsqueeze(-1) * (eye.unsqueeze(0) - s.unsqueeze(-2))  # [B, N, N]  #4. 张量广播规则（关键！）两个张量做加减乘除时：只要某一维长度是 1，会自动复制扩充，匹配对方的长度第 0 维：1 ↔ B，[1,N,N] 自动复制 B 份，变成 [B,N,N]第 1 维：N ↔ 1，[B,1,N] 中间一维复制 N 份，变成 [B,N,N]第 2 维：N ↔ N，不用变两个张量都会自动扩成统一形状 [B,N,N]，才能逐位置相减。
    # 协方差线性传播公式 Σ_s = J Σ_z J^T
    Sigma_s = J @ Sigma_z @ J.transpose(-1, -2)

    # ---------------------- 第二步：计算各项二阶修正所需中间量 ----------------------
    # s^T @ Σ_z  向量乘法：[B,N]
    sTSigma = torch.einsum('bi,bij->bj', s, Sigma_z)
    # einsum解释：bi × bij → 对i求和，得到每个bj的结果
    # s^T @ Σ_z @ s 标量，每个batch一个数值 [B]
    sTSigmas = torch.einsum('bi,bij,bj->b', s, Sigma_z, s)
    # 提取协方差矩阵对角线 Σ_ii，即各通道自身方差 [B,N]
    Sigma_z_diag = torch.diagonal(Sigma_z, dim1=-2, dim2=-1)     #2. torch.diagonal 功能。作用：提取矩阵对角线上所有数字。dim1=-2, dim2=-1 固定搭配：dim1=-2：倒数第 2 维，矩阵的「行」。dim2=-1：倒数第 1 维，矩阵的「列」。也就是：对每一张图的 N×N 矩阵，取主对角线元素
    # 迹项 Σ_j s_j * Σ_jj，每个batch标量 [B]
    trace_term = torch.sum(s * Sigma_z_diag, dim=-1)

    # ---------------------- 三阶：代入二阶修正公式，修正均值 ----------------------
    correction = 0.5 * s * (
        Sigma_z_diag                                                # [B, N]
        - 2.0 * sTSigma                                             # [B, N] 
        + (2.0 * sTSigmas.unsqueeze(-1) - trace_term.unsqueeze(-1)) # [B,1] → 广播 [B,N]
    )  # [B, N]
    # 原始一阶softmax输出 + 二阶泰勒修正，得到更精准的均值
    mu_s = s + correction                                 #全部项广播统一为[B,N]，再和s[B,N]相乘得到correction[B,N]。
    return mu_s, Sigma_s


def spatial_global_avg_pool_moments(
    mu: torch.Tensor,                                     #进网络卷积层之后：CT原图经过多层3D 卷积、贝叶斯网络，不再是CT灰度，变成多通道特征分布
    var: torch.Tensor,                                    #卷积算出特征均值mu、方差var
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对空间特征做全局平均池化，传播均值和方差。

    对 [B, C, D, H, W] → [B, C] 做平均:
      μ_avg = mean(μ, dim=(2,3,4))
      var_avg = mean(var, dim=(2,3,4)) / N   (独立假设下)

    返回初始对角协方差 Σ_avg = diag(var_avg)，shape [B, C, C]

    参数:
        mu: [B, C, D, H, W]
        var: [B, C, D, H, W]
    返回:
        mu_avg: [B, C]  空间均值
        var_avg: [B, C]  空间平均方差
        Sigma_avg: [B, C, C]  对角协方差
    """
    _, _, d, h, w = mu.shape
    N = d * h * w                                    # N=d×h×w：一个通道里一共有多少个空间体素。

    mu_avg = mu.mean(dim=(2, 3, 4))  # [B, C]        # μavg,c​=1/N*∑d,h,w（μ c,d,h,w）含义：第 c 通道所有空间像素均值，就是池化后的 logit 均值。 
    var_avg = var.mean(dim=(2, 3, 4)) / max(N, 1)    # [B, C]方差除以N（独立假设）   #除以N：独立随机变量均值的方差公式；max(N,1)：防止图片尺寸为 0 除 0 报错；   #var avg,c= 1/N*1/N*∑d,h,wvar(c,d,h,w)


    # 构造对角协方差矩阵 [B, C, C]
    B, C = mu_avg.shape
    Sigma_avg = torch.diag_embed(var_avg)    #[B, C, C]     #输入一维数组[B,C]，生成批量对角矩阵[B,C,C]：矩阵对角线 = 对应通道的var_avg；矩阵所有非对角线位置=0；
                                                            # 假设不同通道之间相互独立、无关联，所以协方差非对角线为 0；后面经过线性层、softmax 层运算，才会产生非对角关联项。
    return mu_avg, var_avg, Sigma_avg


def linear_moments(
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear层的精确矩传播。

    μ_out = W @ μ_in + b
    Σ_out = W @ Σ_in @ W^T

    参数:
        mu: [B, C_in]  每张图片输入通道的分布均值                       # 来自池化函数输出mu_avg
        Sigma: [B, C_in, C_in]   每张图输入通道之间的协方差矩阵         # 来自池化函数输出Sigma_avg
        weight: [C_out, C_in]    全连接权重，输出通道*输入通道
        bias: [C_out] or None    全连接偏置（可选）                    # 是你网络里Linear层本身自带的参数，网络训练时学习出来的权重、偏置。
    返回:
        mu_out: [B, C_out]             线性层输出通道均值
        Sigma_out: [B, C_out, C_out]   线性层输出通道协方差矩阵
    """
    # μ_out = W @ μ_in + b
    # [C_out, C_in] @ [B, C_in]^T = [B, C_out]
    mu_out = torch.einsum('oi,bi->bo', weight, mu)                    # 1. einsum ('oi,bi->bo') 详解；oi：对应weight，维度含义 o=C_out，i=C_in；bi：对应mu，维度含义 b=batch，i=C_in->bo：输出维度 b（批次）、o（输出通道）
    if bias is not None:
        mu_out = mu_out + bias.unsqueeze(0)                           # bias原始形状[C_out]，前面加维度变成[1, C_out]广播后和[B,C_out]逐行相加，每张图共用同一套偏置，对应公式 +b

    # Σ_out = W @ Σ_in @ W^T
    # Batched matmul with broadcasting: [1, C_out, C_in] @ [B, C_in, C_in] @ [1, C_in, C_out]
    # = [B, C_out, C_in] @ [B, C_in, C_out]  (after broadcast) = [B, C_out, C_out]
    W_batch = weight.unsqueeze(0)        # [1, C_out, C_in]
    WT_batch = weight.T.unsqueeze(0)    # [1, C_in, C_out]
    WS = W_batch @ Sigma                # [B, C_out, C_in]
    Sigma_out = WS @ WT_batch           # [B, C_out, C_out]

    return mu_out, Sigma_out


# ==========================================================================================
# Module 1: MomentDrop3d —— 标准 inverted dropout 的矩传播
# ==========================================================================================
class MomentDrop3d(nn.Module):
    """
    同时支持普通 forward 和矩传播 forward_mu_var 的 3D Dropout 层。

    这里使用的是 PyTorch 标准 inverted dropout，不是论文中“测试时再乘 keep”
    的原始 non-inverted 写法。

    普通 forward 行为：
    - 训练态 self.training=True：
        y = δ * x / keep，δ ~ Bernoulli(keep)
        也就是随机把一部分中间特征置 0，并把保留下来的特征除以 keep。
        这样训练时输出期望保持不变：E[y] = x。
    - 评估态 self.training=False：
        直接返回 x，不再随机丢特征，也不再额外乘 keep。

    矩传播 forward_mu_var 行为：
    - 输入不是一次具体采样值，而是当前层特征分布的均值 mu 和方差 var。
    - 使用同一个 inverted dropout 定义解析计算输出均值/方差。
    - 公式为：
        μ_y = μ_x
        σ²_y = σ²_x / keep + (p / keep) * μ_x²

    这能让三条链保持一致：
    1. 训练 forward：标准 inverted dropout；
    2. MC Dropout：测试时只打开 dropout，多次采样；
    3. Moment propagation：不用采样，直接用上面的方差公式算 dropout 引入的随机性。

    参数：
        p: dropout 丢弃概率。例如 p=0.2 表示每次训练随机丢弃 20% 中间特征。
        keep_prob: 保留概率，keep_prob = 1 - p。
    """
    def __init__(self, p: float = 0.2):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}.")
        self.p = p                    # 丢弃概率；p=0.2 表示随机置零 20% 中间特征
        self.keep_prob = 1.0 - p      # 保留概率；inverted dropout 的缩放因子为 1 / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        普通张量前向。

        训练阶段：
            随机生成 mask，对中间特征做 x * mask / keep_prob。
            这里的随机性是训练正则化，也是之后 MC Dropout 的采样来源。

        验证/普通测试阶段：
            self.training=False，直接返回 x。
            注意：这里不随机丢弃，也不乘 keep，因为 inverted dropout 已经在训练时完成期望对齐。
        """
        if (not self.training) or self.p <= 0.0:
            return x
        mask = torch.empty_like(x).bernoulli_(self.keep_prob)
        return x * mask / self.keep_prob

    def forward_mu_var(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dropout 层的解析矩传播。

        输入的 mu/var 代表“当前层特征在随机 dropout 影响下的分布”，而不是最终 MRI 图像。
        本函数只负责把 dropout 这一层带来的随机性合并进方差流。

        参数:
            mu: [B, C, D, H, W] 输入特征每个体素的分布均值
            var: [B, C, D, H, W] 输入特征当前已经累计的方差
        返回:
            mu_out: [B, C, D, H, W] inverted dropout 下均值保持不变
            var_out: [B, C, D, H, W] 叠加 dropout 随机开关后的总方差
        """
        p = self.p
        keep = self.keep_prob

        # inverted dropout 的关键点：训练时除以 keep，所以输出均值不被压小。
        mu_out = mu

        # 方差由两部分组成：
        # 1. var / keep：输入已有方差经过随机 mask 和 1/keep 缩放后的结果；
        # 2. p / keep * μ²：dropout 开/关本身给确定均值特征新增的随机方差。
        var_out = var / keep + (p / keep) * (mu * mu)

        return mu_out, var_out


# ==========================================================================================
# Model —— 修改后的网络层，各层均新增 forward_mu_var / forward_mu_var_cov
# ==========================================================================================

#【把输入的3D图像/特征，经过两次卷积处理，提取更有用的局部特征。】
class ConvBlock3D(nn.Module):
    """
    基础3D卷积模块，适配3D CT医学重建任务，同时支持标准训练前向 + 论文矩传播不确定性计算双分支
    模块内部固定流水线：两层3D卷积 + 实例归一化InstanceNorm3d + GELU激活，末尾可选挂载矩传播专用Dropout3d

    对应论文矩传播规则说明：
      1. forward：常规训练/推理主通路，和普通网络一致，用于计算重建loss、更新权重
      2. forward_mu_var：矩传播专用分支，输入每层特征分布均值mu、方差var，逐层解析传播一二阶矩
      3. Conv3D：无偏置线性运算，均值直接卷积；方差使用权重平方卷积（论文表1精确公式，无近似）
      4. InstanceNorm3d：严格遵循论文表1归一化均值、方差传播闭式解，精确计算
      5. GELU非线性激活：使用一阶泰勒展开近似传播均值方差
      6. MomentDrop3d：按标准 inverted dropout 定义生成额外不确定性方差
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(out_ch)

        # dropout>0 时挂载 MomentDrop3d：
        #   普通 forward 走标准 inverted dropout；
        #   forward_mu_var 走同一定义下的解析方差公式。
        # dropout=0 时置空，表示该卷积块不引入 dropout 随机性。
        self.moment_drop: Optional[MomentDrop3d] = (
            MomentDrop3d(dropout) if dropout > 0 else None
        )
        self.drop: nn.Module = self.moment_drop if self.moment_drop is not None else nn.Identity()

        # 构建标准训练前向的串行网络层，兼容常规训练流程
        # 包含两层卷积、归一化、GELU激活，末尾挂载标准 inverted Bernoulli dropout
        layers = [
            self.conv1, self.in1, nn.GELU(),
            self.conv2, self.in2, nn.GELU(),
            self.drop,
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                   
        """
        标准训练前向传播函数，网络训练、常规推理时自动调用
        功能：输入原始3D特征图，串行执行卷积、归一化、激活、dropout，输出提取后的特征
        训练通路说明：
          - model.train() 时 MomentDrop3d 随机失活中间特征；
          - model.eval() 时 MomentDrop3d 关闭随机性；
          - MC Dropout 对照时只把 MomentDrop3d 临时切回 train，其余层保持 eval。
        参数:
            x: [B, in_ch, D, H, W] 输入3D CT特征，批次、输入通道、深度、高度、宽度
        返回:
            处理完成的3D特征 [B, out_ch, D, H, W]
        """
        return self.block(x)

    def forward_mu_var(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        矩传播专用前向函数，仅在单趟不确定性量化时手动调用，不参与训练权重更新
        完整流水线：Conv3D_1 → InstanceNorm3D_1 → GELU → Conv3D_2 → InstanceNorm3D_2 → GELU → MomentDrop3d
        逐层代入论文解析公式，同步更新特征分布的均值mu、方差var，全程无多次采样，单次计算不确定性

        参数:
            mu: [B, in_ch, D, H, W] 输入特征每个体素的分布均值
            var: [B, in_ch, D, H, W] 输入特征每个体素固有分布方差
        返回:
            mu_out: [B, out_ch, D, H, W] 经过本卷积块全部变换后的特征均值
            var_out: [B, out_ch, D, H, W] 叠加所有层变换、dropout新增波动后的总方差（总不确定性）
        """
        # -------------------------- 第1层 3D卷积 线性矩传播（无偏置，论文Table1精确线性公式） --------------------------

        mu = self.conv1(mu)                                                   # 均值传播公式：μ_y = ∑ w_t * μ_{x,t} + b，本卷积bias=False，b=0  # 均值直接使用卷积权重做标准3D卷积变换

        var = F.conv3d(var, self.conv1.weight ** 2, bias=None,                # 方差传播公式（线性变换方差规则）：σ_y² = ∑ w_t² * σ_{x,t}²    Var(y)=w2⋅Var(x)
                    stride=self.conv1.stride, padding=self.conv1.padding)     # 线性层方差 = 权重平方卷积输入方差，无偏置项（常数bias方差为0，不参与方差计算），严格匹配论文表1卷积方差公式

        # -------------------------- 第一层实例归一化 矩传播（论文Table1归一化公式） --------------------------
        # 归一化均值：μ_y = γ * (μ_x − E[μ_x]) / √(V(μ_x)+ε) + β
        # 归一化方差：σ_y² = γ² * σ_x² / (V(μ_x)+ε)
        mu, var = self._instance_norm_moments(mu, var, self.in1)

        # -------------------------- GELU激活 一阶泰勒近似矩传播 --------------------------
        # 一阶泰勒近似公式：
        # μ_y ≈ GELU(μ_x)
        # σ_y² ≈ [GELU′(μ_x)]² · σ_x²
        mu, var = self._gelu_moments(mu, var)

        # -------------------------- 第2层 3D卷积 线性矩传播（同论文卷积公式） --------------------------
        # 均值：μ_y = ∑ w_t * μ_{x,t}，无bias
        mu = self.conv2(mu)
        # 方差：σ_y² = ∑ w_t² * σ_{x,t}²，权重平方卷积输入方差
        var = F.conv3d(var, self.conv2.weight ** 2, bias=None,
                       stride=self.conv2.stride, padding=self.conv2.padding)

        # -------------------------- 第二层实例归一化 矩传播（复用Table1归一化公式） --------------------------
        # μ_y = γ * (μ_x − E[μ_x]) / √(V(μ_x)+ε) + β
        # σ_y² = γ² * σ_x² / (V(μ_x)+ε)
        mu, var = self._instance_norm_moments(mu, var, self.in2)

        # -------------------------- GELU激活 一阶泰勒近似矩传播 --------------------------
        # μ_y ≈ GELU(μ_x)
        # σ_y² ≈ [GELU′(μ_x)]² · σ_x²
        mu, var = self._gelu_moments(mu, var)

        # -------------------------- Dropout层解析计算新增方差 --------------------------
        # 标准 inverted dropout 的解析矩公式：
        #   μ_y = μ_x
        #   σ²_y = σ²_x / keep + p / keep * μ_x²
        # σ_y² = σ_x² / keep + p / keep · μ_x²
        # 存在dropout层时，调用MomentDrop3d专用矩传播函数更新均值、方差
        if self.moment_drop is not None:
            mu, var = self.moment_drop.forward_mu_var(mu, var)

        # 返回经过整个卷积块全部层变换后的均值、方差，传入下一层网络继续逐层矩传播
        return mu, var

    @staticmethod
    def _instance_norm_moments(
        mu: torch.Tensor, var: torch.Tensor, in_layer: nn.InstanceNorm3d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        静态工具函数：InstanceNorm3d 归一化层的精确矩传播实现，完全对应论文Table1归一化公式
        归一化均值公式：μ_y = γ * (μ_x - E[μ_x]) / √(Var[μ_x] + ε) + β
        归一化方差公式：σ²_y = γ² * σ²_x / (Var[μ_x] + ε)
        其中 E[μ_x]、Var[μ_x] 是单张CT样本内部空间维度(D,H,W)计算的均值、方差

        参数:
            mu: [B, C, D, H, W] 归一化前特征均值
            var: [B, C, D, H, W] 归一化前特征方差
            in_layer: 实例归一化层实例，读取内部eps、缩放γ、偏移β参数
        返回:
            mu_out: 归一化后特征均值
            var_out: 归一化缩放后的特征方差
        """
        # 读取归一化防止分母为0的极小值
        eps = in_layer.eps
        # 获取通道数量C
        C = mu.shape[1]

        # 安全读取归一化可学习参数gamma(缩放γ)、beta(偏移β)
        # 若关闭affine无参数，则默认gamma全1、beta全0
        gamma = in_layer.weight if in_layer.weight is not None else torch.ones(C, device=mu.device, dtype=mu.dtype)   #device=mu.device：张量和mu放在同一张显卡/CPU，避免设备不匹配报错； dtype=mu.dtype：数据类型和mu保持一致（float32/float16等） #torch.ones(C,...)：全 1 数组，作为归一化默认缩放 γ
        beta = in_layer.bias if in_layer.bias is not None else torch.zeros(C, device=mu.device, dtype=mu.dtype)       #torch.zeros(C,...)：全 0 数组，作为归一化默认偏移 β

        # 步骤1：在空间维度D,H,W计算单实例均值 E[μ_x]，keepdim保留维度方便广播 [B, C, 1, 1, 1]
        E_mu = mu.mean(dim=(2, 3, 4), keepdim=True)                                                                   #dim=(2,3,4) = 在 ** 空间三维（D、H、W）** 求平均，只对单个体的空间体素做均值，不会跨批次、跨通道；#keepdim=True 关键作用:加 keepdim=True;输出 E_mu 形状 [B, C]（2 维）;加 keepdim=True：输出 E_mu 形状 [B, C, 1, 1, 1]（依旧 5 维）
        # 步骤2：空间维度计算单实例方差 Var[μ_x]，无偏估计关闭unbiased=False [B, C, 1, 1, 1]
        V_mu = mu.var(dim=(2, 3, 4), keepdim=True, unbiased=False)

        # 原始特征标准化：(原始均值 - 实例均值) / 实例标准差
        inv_std = torch.rsqrt(V_mu + eps)                                   # torch.rsqrt(x) = 平方根的倒数
        mu_norm = (mu - E_mu) * inv_std

        # 将一维gamma、beta扩展为5维张量，匹配[B,C,D,H,W]维度广播运算
        gamma_5d = gamma.view(1, -1, 1, 1, 1)                               # gamma、beta 是 InstanceNorm 的仿射参数，一维张量：[C]
        beta_5d = beta.view(1, -1, 1, 1, 1)                                 # 它的可学习参数 weight(gamma) / bias(beta) 长度 = 输出通道数 out_ch，一个通道对应一组独立的 γ、β。每个通道特征分布不一样，需要单独缩放、单独平移；
        # 完成归一化缩放+偏移，得到输出均值
        mu_out = gamma_5d * mu_norm + beta_5d                                #按通道独立计算缩放与偏移，空间上全部体素共用该通道的一套 gamma/beta。

        # 方差传播：按照论文公式，方差乘以gamma平方，除以实例方差+eps
        gamma_sq = (gamma ** 2).view(1, -1, 1, 1, 1)
        var_out = gamma_sq * var / (V_mu + eps)                           

        return mu_out, var_out

    @staticmethod
    def _gelu_moments(
        mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        静态工具函数：GELU非线性激活的矩传播，采用一阶泰勒展开近似
        近似公式：
            输出均值 μ_y ≈ GELU(μ_x)
            输出方差 σ²_y ≈ [GELU'(μ_x)]² * σ²_x
        原理：在均值点做一阶泰勒展开，非线性方差仅由激活函数导数平方缩放原有方差

        参数:
            mu: [B, C, D, H, W] 激活前特征均值
            var: [B, C, D, H, W] 激活前特征方差
        返回:
            mu_out: GELU激活后均值
            var_out: 激活缩放后的方差
        """
        # 同时计算GELU输出值 + GELU在mu处的一阶导数值
        gelu_val, dgelu = gelu_with_derivative(mu)
        # 均值直接取激活函数计算结果
        mu_out = gelu_val
        # 方差 = 导数平方 × 输入方差，一阶泰勒近似
        var_out = (dgelu ** 2) * var
        return mu_out, var_out


class DownBlock3D(nn.Module):
    """
    3D UNet下采样模块，适配论文单趟矩传播不确定性量化双分支逻辑
    整体执行流水线：MaxPool3d下采样压缩空间尺寸→ConvBlock3D双层卷积提取深层特征

    对应论文Table1各层矩传播规则说明：
      1. MaxPool3d最大池化：依据特征均值mu选出每个池化窗口最大值的索引，方差var同步取用该索引位置的值（无近似，精确传播）
      2. ConvBlock3D卷积块：复用已实现的forward_mu_var完整链路，逐层传播均值与方差
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2, return_indices=True)
        self.conv = ConvBlock3D(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准训练/常规重建推理前向函数
        流程：3D最大池化下采样 → 卷积块提取特征
        参数:
            x: [B, in_ch, D, H, W] 输入原始3D特征
        返回:
            下采样并提取后的深层3D特征 [B, out_ch, D//2, H//2, W//2]
        """
        y, _ = self.pool(x)                                # 池化返回两个值：下采样后的特征y、最大值索引indices；训练通路不需要索引，用下划线丢弃
        return self.conv(y)                                # 送入卷积块完成特征提取，返回最终输出特征

    def forward_mu_var(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        矩传播专用前向函数，仅单趟计算认知不确定性时调用
        完整流水线：均值mu做MaxPool获取最大值位置索引 → 方差var按相同索引路由匹配对应数值 → 送入卷积块继续传播矩
        MaxPool论文精确矩传播公式（Table1）:
          T = argmax(μ_x)    # T为每个2×2×2池化窗口内均值最大值的空间位置索引
          μ_y = μ_{x,T}      # 池化输出均值 = 窗口内均值最大位置对应的原始均值
          σ²_y = σ²_{x,T}    # 池化输出方差 = 窗口内均值最大位置对应的原始方差
        参数:
            mu: [B, in_ch, D, H, W] 输入特征分布均值
            var: [B, in_ch, D, H, W] 输入特征分布方差
        返回:
            mu_out: [B, out_ch, D//2, H//2, W//2] 下采样+卷积后特征均值
            var_out: [B, out_ch, D//2, H//2, W//2] 同步变换后的总不确定性方差
        """
        # -------------------------- 第一步：对均值mu执行3D最大池化，获取最大值索引 --------------------------
        # mu_pooled：均值池化结果，空间尺寸长宽高全部减半 [B, C, D', H', W']，D'=D//2, H'=H//2, W'=W//2
        # indices：保存每个2×2×2窗口内均值最大值所在的空间位置索引，形状与mu_pooled完全一致
        mu_pooled, indices = self.pool(mu)

        # -------------------------- 第二步：利用均值得到的索引，同步路由匹配方差var（核心池化方差传播逻辑） --------------------------
        # 取出输入方差的5维尺寸：B批次、C通道、D深度、H高度、W宽度
        B, C, D, H, W = var.shape
        # 将方差var的空间D/H/W三维展平为一维，新形状 [B, C, D*H*W]，方便按索引取值
        var_flat = var.flatten(2)                                                       # tensor.flatten(start_dim)：start_dim=2：从第 2 个维度开始，把后面所有维度压平合并成 1 维
        # 同步把索引indices的空间维度展平，形状变为 [B, C, D'*H'*W']，和展平后方差维度对应
        idx_flat = indices.flatten(2)
        # torch.gather：按照展平后的索引，从展平方差中取出每个窗口最大值位置对应的方差值
        var_pooled_flat = torch.gather(var_flat, 2, idx_flat)
        # 将提取完成的一维池化方差恢复为5维，尺寸和池化后的均值mu_pooled完全一致
        var_pooled = var_pooled_flat.view_as(mu_pooled)

        # -------------------------- 第三步：下采样后的均值、方差送入卷积块，继续逐层矩传播 --------------------------
        mu_out, var_out = self.conv.forward_mu_var(mu_pooled, var_pooled)
        # 返回本下采样模块完整变换后的均值、方差，送入下一层网络
        return mu_out, var_out


class UpBlock3D(nn.Module):
    """
    3D UNet解码器上采样模块，双分支设计：标准训练前向 + 论文单趟矩传播前向
    整体流程：转置卷积2倍上采样恢复空间分辨率 → 对齐跳跃连接特征尺寸 → 通道维度拼接skip特征 → ConvBlock3D融合双通道信息

    矩传播对应论文Table1规则说明：
      1. ConvTranspose3d转置卷积：均值使用原始权重计算，方差使用权重平方做转置卷积（精确线性变换，和普通Conv3d方差逻辑一致）
      2. Skip跳跃拼接：均值、方差各自在通道维度直接拼接，无额外运算 μ_cat = cat(μ_up, μ_skip)，var_cat = cat(var_up, var_skip)
      3. ConvBlock3D卷积块：复用已实现forward_mu_var完整链路，逐层传递均值、方差
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()                                                       # 3D转置卷积，kernel=2，stride=2，将特征D/H/W空间尺寸整体放大2倍（上采样）
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)     # 拼接后通道数 = 上采样输出通道out_ch + 编码器跳跃通道skip_ch，送入卷积块融合特征。#拿卷积权重把小特征图放大 2 倍，生成高分辨率特征
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        标准训练/常规重建推理前向分支
        参数:
            x: [B, in_ch, D, H, W] 解码器浅层输入特征
            skip: [B, skip_ch, D_skip, H_skip, W_skip] 编码器对应层跳跃连接特征
        返回:
            融合跳跃特征后的输出特征 [B, out_ch, D_skip, H_skip, W_skip]
        """
        x = self.up(x)                            # 转置卷积上采样，空间尺寸放大2倍
        if x.shape[2:] != skip.shape[2:]:         # 防止转置卷积尺寸偏移，对齐跳跃特征的深度、高度、宽度
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)         # 要缩放的张量：上采样后的特征图 [B,C,D,H,W] # 目标尺寸，拿skip的D、H、W作为标准大小 # 插值模式：3D三线性插值，专门给三维体数据用 # 像素对齐开关，医学图像重建通用False，数值更稳定
        x = torch.cat([x, skip], dim=1)           # dim=1代表通道维度，拼接上采样特征与编码器skip特征
        return self.conv(x)                       # 卷积块融合拼接后的特征，得到本层输出

    def forward_mu_var(
        self, mu: torch.Tensor, var: torch.Tensor,
        mu_skip: torch.Tensor, var_skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        矩传播专用前向分支，论文One-shot单趟不确定性量化核心逻辑
        完整流程：转置卷积传播均值方差 → 对齐skip空间尺寸 → 通道拼接均值与方差 → 卷积块继续矩传播
        转置卷积矩传播公式（线性变换精确推导）：
        μ_up = ConvTranspose3d(W, μ)
        σ²_up = ConvTranspose3d(W², σ²)
        参数:
            mu: [B, in_ch, D, H, W] 解码器输入特征均值
            var: [B, in_ch, D, H, W] 解码器输入特征方差
            mu_skip: [B, skip_ch, D_skip, H_skip, W_skip] 编码器跳跃连接均值
            var_skip: [B, skip_ch, D_skip, H_skip, W_skip] 编码器跳跃连接方差
        返回:
            mu_out: [B, out_ch, D_skip, H_skip, W_skip] 融合后输出均值（重建图像）
            var_out: [B, out_ch, D_skip, H_skip, W_skip] 融合后输出认知不确定性方差图
        """
        # -------------------------- 第一步：转置卷积传播均值（使用原始权重） --------------------------
        mu_up = self.up(mu)

        # -------------------------- 第二步：转置卷积传播方差（使用权重平方，论文线性方差规则） --------------------------
        # 转置卷积权重形状：[in_ch, out_ch, kD, kH, kW]
        # 线性变换方差公式 Var(W·x) = W²·Var(x)，方差必须使用权重平方运算，无偏置（bias方差为0不参与）
        var_up = F.conv_transpose3d(
            var, self.up.weight ** 2, bias=None,
            stride=self.up.stride, padding=self.up.padding,
            output_padding=self.up.output_padding
        )

        # -------------------------- 第三步：对齐上采样后特征与skip特征的空间尺寸D/H/W --------------------------
        if mu_up.shape[2:] != mu_skip.shape[2:]:
            # 三线性插值缩放均值，匹配跳跃特征尺寸
            mu_up = F.interpolate(mu_up, size=mu_skip.shape[2:], mode="trilinear", align_corners=False)
            # 同步缩放方差，保证mu与var空间形状完全一致
            var_up = F.interpolate(var_up, size=var_skip.shape[2:], mode="trilinear", align_corners=False)

        # -------------------------- 第四步：通道维度拼接上采样特征与编码器跳跃特征 --------------------------
        # dim=1 通道维度，分别拼接均值、方差，通道数叠加
        mu_cat = torch.cat([mu_up, mu_skip], dim=1)
        var_cat = torch.cat([var_up, var_skip], dim=1)

        # -------------------------- 第五步：拼接后的均值方差送入卷积块，继续逐层矩传播 --------------------------
        mu_out, var_out = self.conv.forward_mu_var(mu_cat, var_cat)
        # 返回本上采样模块处理完成的均值、方差，传入下一层解码器
        return mu_out, var_out



#先升维做复杂变换，再降维输出结果。（1个像素就是一个token）
#【把特征当成一串 token，再用全连接层去混合这些token的通道信息】
class DepthwiseTokenMixer(nn.Module):
    """
    Token 混合模块，用于Transformer风格特征通道交互；双分支设计：标准训练前向 + 论文矩传播前向
    标准前向完整流程：LayerNorm层归一化 → 升维全连接fc1 → GELU非线性激活 → Dropout随机失活 → 降维全连接fc2 → Dropout随机失活
    矩传播各层对应规则说明：
      1. LayerNorm层：矩传播精确计算，无近似
      2. Linear全连接层：均值用原始权重，方差使用权重平方运算，线性变换精确推导
      3. GELU激活：一阶泰勒近似传递均值、方差，单通道独立运算
      4. Dropout随机丢弃：概率论精确矩公式计算均值与方差变化
    补充说明：GELU属于单变量逐元素非线性，每个token通道独立计算，仅需传递均值μ、方差σ²，不需要协方差矩阵，降低计算量
    """
    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion     # 升维隐藏通道数 = 原始通道 × 扩张系数，实现先升维丰富特征
        self.channels = channels          # 保存输入通道、隐藏通道、dropout丢弃概率，供矩传播函数调用
        self.hidden = hidden
        self.dropout_rate = dropout

        self.norm = nn.LayerNorm(channels)           # 层归一化，作用于最后一维通道维度[C]，每个token独立归一化
        self.fc1 = nn.Linear(channels, hidden)       # 全连接1：通道升维 C → hidden
        self.fc2 = nn.Linear(hidden, channels)       # 全连接2：通道降维 hidden → C，还原原始通道数量
        # 丢弃率大于0时，标准前向与矩传播共用 MomentDrop3d：
        # 普通训练随机失活，矩传播解析更新方差；dropout=0时不挂载该层。
        self.moment_drop: Optional[MomentDrop3d] = (
            MomentDrop3d(dropout) if dropout > 0 else None
        )
        self.drop: nn.Module = self.moment_drop if self.moment_drop is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准训练/推理前向传播函数
        张量形状规范：[B 批次, N token总数, C 通道]，输入输出形状保持不变 [B, N, C]
        参数:
            x: 输入token特征张量 [B, N, C]
        返回:
            通道混合、激活、归一化后的输出token特征 [B, N, C]
        """
        # 1. 对每个token做层归一化标准化
        y = self.norm(x)
        # 2. 全连接升维，通道从C扩张至hidden
        y = self.fc1(y)
        # 3. GELU非线性激活函数，引入非线性表达能力（这是正常的不需要拿导数的）
        y = F.gelu(y)                    
        # 4. 随机Dropout，抑制过拟合
        y = self.drop(y)
        # 5. 全连接降维，通道还原为原始C
        y = self.fc2(y)
        # 6. 第二次随机Dropout
        y = self.drop(y)
        return y

    def forward_mu_var(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        矩传播专用前向函数，单趟推理不确定性量化使用
        完整流水线：LayerNorm归一化矩传播 → fc1升维全连接矩传播 → GELU一阶泰勒近似 → Dropout矩更新 → fc2降维全连接矩传播 → Dropout矩更新
        所有运算均逐token、逐通道独立执行，仅传递一阶均值μ、二阶方差σ²，无需复杂协方差矩阵
        参数:
            mu: 输入token特征均值 [B, N, C_in]
            var: 输入token特征方差 [B, N, C_in]
        返回:
            mu_out: 经过完整模块变换后的输出均值 [B, N, C_out]（C_out=C_in）
            var_out: 同步变换后的输出总不确定性方差 [B, N, C_out]
        """
        # -------------------------- 第一层：LayerNorm层归一化 精确矩传播公式 --------------------------
        # LayerNorm标准化数学公式：y = (x − E[x]) / √(Var[x] + ε) * γ + β
        # 矩传播对应公式：
        # 均值：μ_y = γ * (μ_x − E[μ_x]) / √(Var_x + ε) + β
        # 方差：σ²_y = γ² * σ²_x / (Var_x + ε)
        eps = self.norm.eps          # 归一化防除0极小常数
        gamma = self.norm.weight     # 归一化通道缩放参数，一维张量[C]
        beta = self.norm.bias        # 归一化通道偏移参数，一维张量[C]

        # 对每个token的全部通道求均值、方差，dim=-1代表最后一维通道C；keepdim=True保留维度用于广播
        E_mu = mu.mean(dim=-1, keepdim=True)  # 输出形状 [B, N, 1]
        V_mu = mu.var(dim=-1, keepdim=True, unbiased=False)  # 无偏估计关闭，匹配LayerNorm内部计算逻辑 [B, N, 1]

        # 计算逆标准差 1 / sqrt(V_mu + eps)
        inv_std = torch.rsqrt(V_mu + eps)
        # 完成归一化均值变换，gamma自动广播适配[B,N,C]形状
        mu = gamma * (mu - E_mu) * inv_std + beta
        # 完成归一化方差变换，γ²缩放方差，除以方差项(V_mu+ε)
        var = (gamma ** 2) * var * (inv_std ** 2)

        # -------------------------- 第二层：fc1升维全连接 线性精确矩传播 --------------------------
        # 调用静态工具函数，输入当前mu、var、fc1层，输出升维后的新均值、方差
        mu, var = self._linear_moments_2d(mu, var, self.fc1)

        # -------------------------- 第三层：GELU激活 一阶泰勒近似矩传播 --------------------------
        # gelu_val = GELU(μ) 激活均值输出
        # dgelu = GELU导数，用于方差缩放：σ²_out = (GELU’(μ))² · σ²_in
        gelu_val, dgelu = gelu_with_derivative(mu)
        mu = gelu_val
        var = (dgelu ** 2) * var

        # -------------------------- 第四层：Dropout随机失活 概率论精确矩更新 --------------------------
        if self.moment_drop is not None:
            mu, var = self.moment_drop.forward_mu_var(mu, var)

        # -------------------------- 第五层：fc2降维全连接 线性精确矩传播 --------------------------
        mu, var = self._linear_moments_2d(mu, var, self.fc2)

        # -------------------------- 第六层：第二次Dropout随机失活，同步更新均值方差 --------------------------
        if self.moment_drop is not None:
            mu, var = self.moment_drop.forward_mu_var(mu, var)

        # 返回本TokenMixer模块完整变换后的均值、方差，送入下游网络层
        return mu, var

    @staticmethod
    def _linear_moments_2d(
        mu: torch.Tensor, var: torch.Tensor, layer: nn.Linear
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2维特征[B,N,C]专用全连接层矩传播静态工具函数，线性变换无近似、精确计算
        全连接线性变换数学公式：Y = X @ W.T + b
        矩传播对应公式：
        均值：μ_out = μ_in × W转置 + b （逐token独立运算）
        方差：σ²_out = σ²_in × (W平方)转置 （常数偏置b方差为0，不参与方差计算）
        参数:
            mu: 输入特征均值 [B, N, C_in]
            var: 输入特征方差 [B, N, C_in]
            layer: Linear全连接层实例，读取权重W、偏置b
        返回:
            mu_out: 全连接变换后均值 [B, N, C_out]
            var_out: 全连接变换后方差 [B, N, C_out]
        """
        W = layer.weight  # 权重张量形状 [C_out, C_in]
        b = layer.bias    # 偏置张量形状 [C_out]，无偏置时为None

        # 均值传播：einsum批量矩阵乘法 'bnc,oc->bno'                      #第一个参数 mu 维度标记 b n c：[B, N, C_in]；b = batch 批次；n = token 序号；c = 输入通道 C_in；第二个参数 W 维度标记 o c：[C_out, C_in]；o = 输出通道 C_out；c = 输入通道 C_in（和 mu 的 c 对齐相乘）；输出 b n o：[B, N, C_out]
        # bn:批次+token，c输入通道，o输出通道；等价批量X@W.T                #对每一个 batch 里的每一个 token，单独做矩阵乘法：输入通道 C_in 和 权重矩阵 W 相乘，输出 C_out 通道。
        mu_out = torch.einsum('bnc,oc->bno', mu, W)
        # 存在偏置时，给每个token统一叠加通道偏置b
        if b is not None:
            mu_out = mu_out + b

        # 方差传播规则：权重全部平方后再做矩阵乘法
        W_sq = W ** 2
        var_out = torch.einsum('bnc,oc->bno', var, W_sq)

        return mu_out, var_out


#【把"前面几层留下来的历史特征"拿出来，看当前特征更像谁，然后按相似程度，把这些历史特征加权求和。】
class AttnResidualAggregator(nn.Module):
    """
    Transformer风格注意力历史残差聚合模块，双分支设计：标准训练前向 + 论文高阶矩传播前向
    功能说明：以当前层特征作为Query，所有编码器历史特征作为Key，通过注意力相似度计算权重，加权融合全部历史残差特征，弥补3D UNet深层细节丢失问题

    矩传播核心特殊说明（区别前面所有模块）：
      普通卷积/全连接/GELU仅传递单通道均值μ、对角方差σ²；
      注意力QK点积+Softmax属于多变量耦合运算，通道之间存在相关性，对角方差假设完全失效，
      必须引入**完整协方差矩阵Σ**，使用二阶泰勒展开做精确矩传播，否则不确定性会严重低估
    矩传播完整执行流程：
      1. 3D特征全局平均池化：压缩空间D/H/W，输出通道维度均值μ_avg、对角方差var_avg、初始对角协方差矩阵Σ_avg
      2. Query线性层变换：计算Q的均值μ_Q、协方差Σ_Q
      3. 每个历史Key线性层变换：单独计算每组K的均值μ_K_l、协方差Σ_K_l
      4. QK缩放点积打分：计算相似度分数均值μ_scores、分数间协方差矩阵Σ_scores（考虑变量耦合）
      5. Softmax归一化权重：二阶泰勒展开近似，输出权重均值μ_w、权重协方差Σ_w
      6. 加权融合重建：均值直接权重求和；方差分为两部分：①固定权重下历史特征固有方差 ②权重自身不确定性带来的额外方差
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels                          # 保存特征通道数量C
        self.query = nn.Linear(channels, channels)        # Query全连接层：当前特征生成查询向量Q
        self.key = nn.Linear(channels, channels)          # Key全连接层：历史特征生成键向量K（只作用最后一层C）
        self.scale = channels ** -0.5                     # 注意力缩放因子 1/√C，缩放点积防止数值过大梯度爆炸

    def forward(self, current: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        """
        标准训练/常规重建推理前向分支
        参数:
            current: [B, C, D, H, W] 当前层输入3D特征
            history: 列表，每个元素 [B, C, D, H, W] 编码器各层历史跳跃残差特征
        返回:
            注意力加权融合后的聚合特征，形状与current完全一致 [B, C, D, H, W]
        """
        if len(history) == 0:                                                      # 无历史特征时直接返回全零张量
            return torch.zeros_like(current)                                       #生成和输入张量形状、数据类型、设备完全一模一样的全 0 张量。

        q = current.mean(dim=(2, 3, 4))                                            # 对当前特征做全局空间平均池化，压缩D/H/W，得到每个样本通道描述符 [B, C]。    #dim=(2,3,4)：只压缩空间，得到 [B,C]，每个通道一个均值。
        hist_desc = torch.stack([h.mean(dim=(2, 3, 4)) for h in history], dim=1)   # 遍历所有历史特征，全局平均池化后堆叠，维度1为历史序号L [B, L, C]。         #torch.stack（）：把一组相同形状的张量，在指定的新维度上堆叠，会新增一个维度。

        q = self.query(q)                                                        # 线性层生成Query向量 [B, C]。
        k = self.key(hist_desc)                                                  # 线性层批量生成所有历史Key向量 [B, L, C]。      #L表示历史特征。

        scores = torch.einsum("bc,blc->bl", q, k) * self.scale                   # 缩放点积计算相似度分数：每个样本、每个历史层得到打分 [B, L]。      #einsum 含义："bc,blc->bl"规则：相同字母做内积求和，不同字母保留；这里公共字母是c（通道），所以会对c全部相加；剩下独立字母b、l保留，得到输出bl。     #score（b,l）= 1/根号下C*∑【c=1，C】q（b,c）⋅k（b,l,c）
        weights = torch.softmax(scores, dim=1)                                   # Softmax归一化，得到各历史特征的注意力权重，所有权重之和=1         #dim=1：沿着 **L（历史层）** 这一维做 softmax

        out = torch.zeros_like(current)                                          # 初始化输出全零特征
        for i, h in enumerate(history):                                          # 遍历每一个历史特征，用对应注意力权重缩放后累加融合
            out = out + weights[:, i].view(-1, 1, 1, 1, 1) * h                   # weights[:,i]形状[B]，view扩展5维广播 [B,1,1,1,1]。每一个下相应的h去乘对应的权重。             #weights形状是[B=2, L=3]，weights[:,i] 表示:取出所有样本对应第i张历史图的权重，得到一个一维向量，形状[B=2]。
        return out

    def forward_mu_var_cov(
        self,
        mu_curr: torch.Tensor,
        var_curr: torch.Tensor,
        history_mu: List[torch.Tensor],
        history_var: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        高阶协方差矩传播专用分支，仅单趟不确定性量化推理时调用
        核心约束：注意力多变量耦合，必须维护通道间完整协方差矩阵Σ，二阶泰勒展开近似
        优化方案：协方差仅在全局池化后的一维通道描述符[B,C]上计算，不保留空间维度协方差，大幅节省GPU显存
        参数:
            mu_curr: [B, C, D, H, W] 当前输入特征均值
            var_curr: [B, C, D, H, W] 当前输入特征对角方差
            history_mu: 列表，每个元素[B,C,D,H,W] 编码器历史层特征均值
            history_var: 列表，每个元素[B,C,D,H,W] 编码器历史层特征对角方差
        返回:
            mu_out: [B, C, D, H, W] 注意力加权聚合后的输出特征均值（重建体数据）
            var_out: [B, C, D, H, W] 融合双重不确定性的总输出方差（量化重建误差）
        """
        L = len(history_mu)                          # L = 历史特征层数
        if L == 0:                                   # 无历史特征，直接返回零均值、零方差
            return torch.zeros_like(mu_curr), torch.zeros_like(var_curr)

        C = self.channels                            # 读取通道数、计算设备、数据类型、注意力缩放系数
        device = mu_curr.device
        dtype = mu_curr.dtype
        scale = self.scale

        # -------------------------- Step 1：全局平均池化，压缩空间维度，生成通道级矩与初始对角协方差 --------------------------
        # 自定义池化矩工具函数：输入5维均值方差，输出池化后[B,C]均值、[B,C]对角方差、[B,C,C]对角协方差矩阵
        mu_avg, var_avg, Sigma_avg = spatial_global_avg_pool_moments(mu_curr, var_curr)
        # mu_avg: [B, C] 池化后通道均值
        # Sigma_avg: [B, C, C] 初始协方差矩阵，仅对角线有值（各通道初始独立无相关性）

        # 遍历所有历史特征，批量做全局池化，保存每层的均值、方差、协方差
        hist_mu_avg_list = []
        hist_var_avg_list = []
        hist_Sigma_list = []
        for mu_h, var_h in zip(history_mu, history_var):                     #zip():把多个长度相同的可迭代对象（列表/张量列表）按位置配对打包，一一对应组合。
            m, v, S = spatial_global_avg_pool_moments(mu_h, var_h)
            hist_mu_avg_list.append(m)
            hist_var_avg_list.append(v)
            hist_Sigma_list.append(S)

        # -------------------------- Step 2：线性层分别传播Query(Q)、各历史Key(K)的均值与协方差 --------------------------
        # 2.1 对当前池化特征做Query线性变换，同步更新均值与协方差
        # 线性协方差传播公式：μ_out = Wμ+b，Σ_out = W·Σ_in·W^T
        mu_Q, Sigma_Q = linear_moments(mu_avg, Sigma_avg, self.query.weight, self.query.bias)

        # 2.2 循环处理每一层历史池化特征，生成对应Key的均值、协方差
        mu_K_list = []
        Sigma_K_list = []
        W_k = self.key.weight  # Key线性层权重 [C, C]
        b_k = self.key.bias    # Key线性层偏置 [C]
        for m_ha, S_ha in zip(hist_mu_avg_list, hist_Sigma_list):
            m_k, S_k = linear_moments(m_ha, S_ha, W_k, b_k)
            mu_K_list.append(m_k)
            Sigma_K_list.append(S_k)

        # 堆叠所有历史Key均值，维度1为历史序号L [B, L, C]
        mu_K = torch.stack(mu_K_list, dim=1)
        # 堆叠所有历史Key协方差，维度1为历史序号L [B, L, C, C]
        Sigma_K = torch.stack(Sigma_K_list, dim=1)

        # -------------------------- Step 3：QK缩放点积，计算相似度分数的均值、分数间协方差矩阵 --------------------------
        # 分数计算公式：scores[b,l] = (1/√C) * Σ_{c=0}^C Q[b,c] * K[b,l,c]
        # 近似假设：当前特征Q与历史特征K相互独立；不同历史层的K之间相互独立，简化协方差计算
        # 分数均值：μ_scores[b,l] = (1/√C) × Q均值 点乘 第l层K均值
        mu_scores = torch.einsum('bc,blc->bl', mu_Q, mu_K) * scale  # [B, L]

        # 初始化分数协方差矩阵 Σ_scores: [B, L, L]，存储任意两层历史分数之间的相关性
        B = mu_Q.shape[0]
        Sigma_scores = torch.zeros(B, L, L, device=device, dtype=dtype)

        # 双重循环填充协方差矩阵，分对角线(自身方差)、非对角线(跨层相关性)
        for l in range(L):
            # ========== 对角线项 l=m：第l个分数自身的方差 ==========
            # 项1：tr(Σ_Q @ Σ_K_l) 两个协方差矩阵乘积的迹，代表通道独立方差贡献
            tr_term = torch.einsum('bij,bji->b', Sigma_Q, Sigma_K[:, l])  # [B]                #[:, l] =固定第l层历史K,丢掉L维度;结果形状：[B, C, C];含义：第l层Key向量的通道协方差矩阵，每个样本一张C×C矩阵。   #b = 批次，i = 行通道，j = 列通道。
            # 项2：μ_Q^T @ Σ_K_l @ μ_Q Q均值作用于K协方差
            a_term = torch.einsum('bi,bij,bj->b', mu_Q, Sigma_K[:, l], mu_Q)  # [B]
            # 项3：μ_K_l^T @ Σ_Q @ μ_K_l K均值作用于Q协方差
            b_term = torch.einsum('bi,bij,bj->b', mu_K[:, l], Sigma_Q, mu_K[:, l])  # [B]
            # 合并三项 ×缩放系数平方，存入对角线位置
            Sigma_scores[:, l, l] = (tr_term + a_term + b_term) * (scale ** 2)               #下标 [:, l, l]：第一个:：全部样本 0~B-1 都取到；第二个l、第三个l：固定行 = 第 l 层、列 = 第 l 层 → 协方差矩阵对角线

            # ========== 非对角线项 l≠m：第l、m两个分数之间的协方差 ==========
            for m in range(l + 1, L):
                # 仅Q的协方差产生跨层相关性，K之间独立无交叉项
                cross_term = torch.einsum('bi,bij,bj->b', mu_K[:, l], Sigma_Q, mu_K[:, m])
                cross_term = cross_term * (scale ** 2)
                Sigma_scores[:, l, m] = cross_term
                # 协方差矩阵对称，(l,m)=(m,l)，同步赋值
                Sigma_scores[:, m, l] = Sigma_scores[:, l, m]

        # -------------------------- Step 4：Softmax非线性二阶泰勒近似，计算注意力权重均值、权重协方差 --------------------------
        # 输入分数均值μ_scores、分数协方差Σ_scores，二阶泰勒展开得到归一化权重的矩
        mu_weights, Sigma_weights = softmax_2nd_order_moments(mu_scores, Sigma_scores)
        # mu_weights: [B, L] 每层历史特征注意力权重均值
        # Sigma_weights: [B, L, L] 权重之间完整协方差矩阵（权重自身存在不确定性）

        # -------------------------- Step 5：利用注意力权重加权聚合所有历史3D特征，计算输出均值与总方差 --------------------------
        # 5.1 聚合输出均值：μ_out = Σ_{l=0}^L w_l * μ_hist_l，逐通道逐体素加权求和
        mu_out = torch.zeros_like(mu_curr)
        for l in range(L):
            # 权重[B]扩展5维广播 [B,1,1,1,1]，匹配3D特征维度
            w_l = mu_weights[:, l].view(-1, 1, 1, 1, 1)
            mu_out = mu_out + w_l * history_mu[l]

        # 5.2 聚合输出总方差，分为两部分叠加，完整公式：
        # var_out = Σ_l w_l²·var_hist_l  +  Σ_{l,m} Cov(w_l,w_m)·μ_hist_l·μ_hist_m
        var_out = torch.zeros_like(var_curr)

        # Part (a) 第一部分：权重视为固定常数时，历史特征自带的固有方差贡献
        for l in range(L):
            w_l_sq = (mu_weights[:, l] ** 2).view(-1, 1, 1, 1, 1)
            var_out = var_out + w_l_sq * history_var[l]

        # Part (b) 第二部分：权重本身存在不确定性（协方差不为0），带来额外方差贡献
        # 任意两层历史特征的均值两两相乘，乘以对应权重协方差后累加
        for l in range(L):
            for m in range(L):
                cov_w_lm = Sigma_weights[:, l, m].view(-1, 1, 1, 1, 1)
                var_out = var_out + cov_w_lm * (history_mu[l] * history_mu[m])

        # 返回聚合完成的重建均值、总不确定性方差
        return mu_out, var_out


class AttnResBottleneckBlock(nn.Module):
    """
    带注意力残差聚合的bottleneck块 + 矩传播扩展（模块4）。
    这是3D医学重建网络的基础残差模块，同时支持标准训练前向、不确定性矩传播双分支。

    双流前向流程（标准forward训练流程 & forward_mu_var_cov矩传播流程完全对齐）:
      1. 局部ConvBlock: forward_mu_var（精确矩，仅对角方差，不计算通道协方差）
      2. 第一次Attn聚合: forward_mu_var_cov（引入完整通道协方差，Softmax二阶泰勒近似，也就是你前面看懂的AttnResidualAggregator）
      3. TokenMixer全局特征混合: forward_mu_var（一阶泰勒近似GELU激活，仅对角方差，无协方差）
      4. InstanceNorm + 1×1 Conv投影层: 线性层、归一化，使用精确解析矩传播
      5. 第二次Attn聚合: forward_mu_var_cov，融新增本层中间特征作为历史记忆，合更多细节
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.channels = channels                                                        # 特征通道数

        self.conv_local = ConvBlock3D(channels, channels, dropout=dropout)              # 1. 局部3D卷积块，提取局部空间细节，支持矩传播
        self.token_mixer = DepthwiseTokenMixer(channels, expansion=2, dropout=dropout)  # 2. Token混合模块：深度卷积全局建模，带GELU激活，仅对角方差矩传播
        self.res_agg1 = AttnResidualAggregator(channels)                                # 第一次历史注意力聚合器：融合网络前面所有层历史残差特征
        self.res_agg2 = AttnResidualAggregator(channels)                                # 第二次注意力聚合器：融合旧历史 + 当前层中间特征
        self.norm = nn.InstanceNorm3d(channels)                                         # 实例归一化，逐样本标准化特征分布
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)            # 1×1 3D卷积，通道维度线性变换，无偏置

    def forward(self, x: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        """标准前向传播（训练用，只输出确定特征值，不计算不确定性方差）
        Args:
            x: 输入3D特征 [B, C, D, H, W]
            history: 列表，存储前面网络层输出的历史残差特征，每层一个[B,C,D,H,W]张量
        Returns:
            x2: 经过bottleneck块处理后的输出3D特征
        """
        # Step1 局部卷积分支
        local = self.conv_local(x)
        x1 = x + local  # 残差连接：原始输入 + 局部卷积提取的局部特征

        # 第一次注意力聚合：融合全部历史特征，叠加到主分支
        x1 = x1 + self.res_agg1(x1, history)

        # Step2 TokenMixer全局特征混合
        b, c, d, h, w = x1.shape
        # flatten空间维度D,H,W合并为token序列N，转置为 [B, N, C] 适配Transformer结构
        tokens = x1.flatten(2).transpose(1, 2)
        mixed = self.token_mixer(tokens)
        # 还原回原始3D体数据形状 [B, C, D, H, W]
        mixed = mixed.transpose(1, 2).reshape(b, c, d, h, w)

        x2 = x1 + mixed  # 残差连接：叠加全局混合后的特征
        x2 = self.norm(x2)  # 实例归一化
        x2 = self.proj(x2)  # 1×1卷积通道投影                      #kernel_size=1，卷积核只有 1×1×1，不改变空间尺寸 D、H、W，只在通道维度做线性变换。

        # 第二次注意力聚合：历史列表新增当前中间特征x1，融合更多细节
        x2 = x2 + self.res_agg2(x2, history + [x1])
        return x2

    def forward_mu_var_cov(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        history_mu: List[torch.Tensor],
        history_var: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整双流矩传播（模块4，不确定性量化推理专用）：
        执行步骤与forward标准前向完全一一对应，同步传递特征均值μ、对角方差var。
        优化设计：通道协方差仅在AttnResidualAggregator内部临时计算，运算结束销毁，节省显存。

        Args:
            mu: [B, C, D, H, W] 输入特征均值（对应标准forward里的x）
            var: [B, C, D, H, W] 输入特征对角方差（像素固有不确定性）
            history_mu: 列表，每层历史特征的均值张量
            history_var: 列表，每层历史特征的对角方差张量
        Returns:
            mu_out: [B, C, D, H, W] 块输出特征均值
            var_out: [B, C, D, H, W] 块输出总不确定性方差
        """
        C = self.channels

        # ===================== Step 1 局部卷积分支（仅对角方差，无协方差） =====================
        # 调用卷积块矩传播函数，输出卷积后特征均值、方差
        mu_local, var_local = self.conv_local.forward_mu_var(mu, var)
        # 残差相加：均值直接相加
        mu1 = mu + mu_local
        # 概率独立随机变量方差可直接相加，残差分支方差叠加
        var1 = var + var_local

        # 第一次历史注意力聚合（内部会生成、计算完整通道协方差矩阵）
        mu_agg1, var_agg1 = self.res_agg1.forward_mu_var_cov(mu1, var1, history_mu, history_var)
        # 残差叠加聚合后的融合特征
        mu1 = mu1 + mu_agg1
        var1 = var1 + var_agg1  # 两部分独立，方差叠加

        # ===================== Step 2 TokenMixer全局特征混合（一阶泰勒GELU，无协方差） =====================
        b, c, d, h, w = mu1.shape
        # 空间维度展平，转换token格式 [B, N, C]，均值、方差同步变换
        mu_tokens = mu1.flatten(2).transpose(1, 2)  # N = D*H*W
        var_tokens = var1.flatten(2).transpose(1, 2)

        # Token混合模块矩传播，GELU用一阶泰勒近似，仅输出对角方差
        mu_mixed, var_mixed = self.token_mixer.forward_mu_var(mu_tokens, var_tokens)

        # 还原回原始3D体数据尺寸，均值、方差同步reshape
        mu_mixed = mu_mixed.transpose(1, 2).reshape(b, c, d, h, w)
        var_mixed = var_mixed.transpose(1, 2).reshape(b, c, d, h, w)

        # 残差连接，叠加全局混合特征
        mu2 = mu1 + mu_mixed
        var2 = var1 + var_mixed

        # ===================== Step3 InstanceNorm3d 精确解析矩传播 =====================
        # 归一化自带均值、方差解析更新，无近似，精确计算标准化后不确定性
        mu2, var2 = ConvBlock3D._instance_norm_moments(mu2, var2, self.norm)

        # ===================== Step4 1×1 线性卷积投影（无偏置，精确矩） =====================
        # 均值直接正常卷积前向
        mu2 = self.proj(mu2)
        # 线性层方差传播公式：Var(Wx) = W² ⊙ Var(x)，无偏置无需加常数方差
        var2 = F.conv3d(var2, self.proj.weight ** 2, bias=None,
                        stride=self.proj.stride, padding=self.proj.padding)

        # ===================== Step5 第二次注意力聚合：扩充历史特征列表 =====================
        # 本次中间特征mu1、var1存入历史，下一层可以复用当前层细节信息
        updated_history_mu = history_mu + [mu1]
        updated_history_var = history_var + [var1]
        # 第二次注意力加权融合，内部再次计算协方差做高阶矩传播
        mu_agg2, var_agg2 = self.res_agg2.forward_mu_var_cov(
            mu2, var2, updated_history_mu, updated_history_var
        )
        # 残差叠加聚合输出
        mu2 = mu2 + mu_agg2
        var2 = var2 + var_agg2
        # 返回整个bottleneck块最终的特征均值、总不确定性方差
        return mu2, var2


# 用于 3D CT -> MRI 重建/翻译的 U-Net 风格网络主干。
class AttnResCTtoMRI(nn.Module):
    """
    CT -> MRI 主模型，并额外提供不确定性估计接口。

    网络主体：
        3D U-Net 编码器/解码器 + AttnRes bottleneck + 可选 Med-ReCL 训练约束。

    三个前向接口的分工：
    1. forward(x)
        普通重建前向，输出一张预测 MRI。
        训练阶段用它计算 loss 并更新权重；验证和普通测试也用它计算重建指标。

    2. forward_mu_var_cov(x)
        本实验主方法：单次矩传播。
        输入同一张 CT，只跑一次网络的解析矩传播分支，输出：
            pred: 合成 MRI 均值图；
            uncertainty: Moment propagation variance map。
        这个函数不训练参数，只使用已经学好的权重套公式。

    3. mc_dropout_inference(x, num_passes)
        参考对照方法：MC Dropout。
        在测试/分析阶段重复 forward 多次，统计预测方差。
        目的是检查 Moment variance map 是否接近 MC variance map。
    """
    def __init__(
        self,
        in_channels: int = 1,                # 输入CT单通道
        out_channels: int = 1,               # 输出MRI单通道
        base_channels: int = 32,             # 基础通道数，编码器逐层翻倍
        bottleneck_blocks: int = 6,          # 瓶颈层堆叠的注意力残差块数量
        dropout: float = 0.0,                # dropout丢弃概率；用于训练正则化、Moment方差公式和MC对照
        final_activation: str = "sigmoid",   # 最后一层激活，sigmoid把输出约束0~1
        use_medrecl: bool = True,
        medrecl_proj_dim: int = 64,
    ):
        super().__init__()
        self.dropout_rate = dropout
        self.use_medrecl = use_medrecl

        # 编码器通道逐级翻倍
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # 编码器：4层下采样提取多尺度CT特征
        self.enc1 = ConvBlock3D(in_channels, c1, dropout=dropout)   # 首层不做下采样
        self.enc2 = DownBlock3D(c1, c2, dropout=dropout)            # 下采样块：卷积+步长下采样
        self.enc3 = DownBlock3D(c2, c3, dropout=dropout)
        self.enc4 = DownBlock3D(c3, c4, dropout=dropout)

        # 瓶颈层：堆叠多个带历史注意力聚合的残差bottleneck块
        self.bottleneck = nn.ModuleList([
            AttnResBottleneckBlock(c4, dropout=dropout)              # 循环创建 bottleneck_blocks 个 AttnResBottleneckBlock 
            for _ in range(bottleneck_blocks)
        ])

        # 解码器：3层上采样，跳跃连接融合编码器浅层细节
        self.dec3 = UpBlock3D(c4, c3, c3, dropout=dropout)
        self.dec2 = UpBlock3D(c3, c2, c2, dropout=dropout)
        self.dec1 = UpBlock3D(c2, c1, c1, dropout=dropout)

        # 输出头：1×1 3D卷积，将通道压缩为1通道MRI图像
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)
        self.final_activation = final_activation                    #右边 final_activation：是模型__init__函数的输入参数，创建模型时传进来的字符串；左边 self.final_activation：把这个字符串保存为模型实例的成员变量，整个类里任何函数（forward、forward_mu_var_cov）都能读取使用。
        self.medrecl_level_channels = (c1, c2, c3)

        if self.use_medrecl:
            self.medrecl_target_encoder = MedReCLTargetEncoder(
                in_channels=in_channels,
                c1=c1,
                c2=c2,
                c3=c3,
            )
            self.medrecl_proj_g = nn.ModuleList(
                [ProjectionHead3D(ch, medrecl_proj_dim) for ch in self.medrecl_level_channels]
            )
            self.medrecl_proj_x = nn.ModuleList(
                [ProjectionHead3D(ch, medrecl_proj_dim) for ch in self.medrecl_level_channels]
            )
            self.medrecl_proj_y = nn.ModuleList(
                [ProjectionHead3D(ch, medrecl_proj_dim) for ch in self.medrecl_level_channels]
            )
            # Appearance uses its own projection space. Unlike the structure
            # heads, these heads intentionally contain no InstanceNorm and
            # their outputs are not L2-normalized, so MRI amplitude and local
            # contrast are not erased before appearance alignment.
            self.medrecl_app_proj_g = nn.ModuleList(
                [AppearanceProjectionHead3D(ch, medrecl_proj_dim) for ch in self.medrecl_level_channels]
            )
            self.medrecl_app_proj_y_ema = copy.deepcopy(self.medrecl_app_proj_g)
            self.medrecl_target_encoder_ema = copy.deepcopy(
                self.medrecl_target_encoder
            )
            self.medrecl_proj_y_ema = copy.deepcopy(self.medrecl_proj_y)
            self._initialize_medrecl_teacher()
        else:
            self.medrecl_target_encoder = None
            self.medrecl_target_encoder_ema = None
            self.medrecl_proj_g = nn.ModuleList()
            self.medrecl_proj_x = nn.ModuleList()
            self.medrecl_proj_y = nn.ModuleList()
            self.medrecl_proj_y_ema = nn.ModuleList()
            self.medrecl_app_proj_g = nn.ModuleList()
            self.medrecl_app_proj_y_ema = nn.ModuleList()

    @staticmethod
    @torch.no_grad()
    def _copy_module_state(target_module: nn.Module, source_module: nn.Module) -> None:
        target_module.load_state_dict(source_module.state_dict())

    @staticmethod
    @torch.no_grad()
    def _ema_module_state(
        target_module: nn.Module,
        source_module: nn.Module,
        momentum: float,
    ) -> None:
        source_parameters = dict(source_module.named_parameters())
        for name, target_parameter in target_module.named_parameters():
            source_parameter = source_parameters[name]
            target_parameter.mul_(momentum).add_(
                source_parameter.detach(),
                alpha=1.0 - momentum,
            )
        source_buffers = dict(source_module.named_buffers())
        for name, target_buffer in target_module.named_buffers():
            if name in source_buffers:
                target_buffer.copy_(source_buffers[name].detach())

    @torch.no_grad()
    def _initialize_medrecl_teacher(self) -> None:
        """Initialize the frozen MRI teacher from the online MRI target branch."""
        self._copy_module_state(
            self.medrecl_target_encoder_ema,
            self.medrecl_target_encoder,
        )
        self.medrecl_target_encoder_ema.requires_grad_(False)
        for teacher_projection, online_projection in zip(
            self.medrecl_proj_y_ema,
            self.medrecl_proj_y,
        ):
            self._copy_module_state(teacher_projection, online_projection)
            teacher_projection.requires_grad_(False)
        for teacher_projection, online_projection in zip(
            self.medrecl_app_proj_y_ema,
            self.medrecl_app_proj_g,
        ):
            self._copy_module_state(teacher_projection, online_projection)
            teacher_projection.requires_grad_(False)
        self.medrecl_target_encoder_ema.eval()
        self.medrecl_proj_y_ema.eval()
        self.medrecl_app_proj_y_ema.eval()

    @torch.no_grad()
    def update_medrecl_teacher(self, momentum: float = 0.99) -> None:
        """EMA-update the frozen MRI target encoder after a successful optimizer step."""
        if not self.use_medrecl or self.medrecl_target_encoder_ema is None:
            return
        momentum = float(max(0.0, min(0.99999, momentum)))
        self._ema_module_state(
            self.medrecl_target_encoder_ema,
            self.medrecl_target_encoder,
            momentum,
        )
        for teacher_projection, online_projection in zip(
            self.medrecl_proj_y_ema,
            self.medrecl_proj_y,
        ):
            self._ema_module_state(teacher_projection, online_projection, momentum)
        for teacher_projection, online_projection in zip(
            self.medrecl_app_proj_y_ema,
            self.medrecl_app_proj_g,
        ):
            self._ema_module_state(teacher_projection, online_projection, momentum)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load old checkpoints by seeding newly added EMA keys from online MRI keys."""
        if self.use_medrecl:
            for key in self.state_dict().keys():
                if key in state_dict:
                    continue
                if key.startswith("medrecl_target_encoder_ema."):
                    source_key = key.replace(
                        "medrecl_target_encoder_ema.",
                        "medrecl_target_encoder.",
                        1,
                    )
                elif key.startswith("medrecl_proj_y_ema."):
                    source_key = key.replace(
                        "medrecl_proj_y_ema.",
                        "medrecl_proj_y.",
                        1,
                    )
                elif key.startswith("medrecl_app_proj_y_ema."):
                    source_key = key.replace(
                        "medrecl_app_proj_y_ema.",
                        "medrecl_app_proj_g.",
                        1,
                    )
                else:
                    continue
                if source_key in state_dict:
                    state_dict[key] = state_dict[source_key].detach().clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_medrecl and self.medrecl_target_encoder_ema is not None:
            self.medrecl_target_encoder_ema.eval()
            self.medrecl_proj_y_ema.eval()
            self.medrecl_app_proj_y_ema.eval()
        return self

    def _forward_backbone(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
        """共享主干：普通前向只返回预测，训练期可选返回多尺度特征。"""
        # 编码器逐层提取特征，保存每层跳跃连接特征s1~s3
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)

        # 瓶颈层循环，维护历史特征列表，每层计算完存入history供后续注意力聚合
        history: List[torch.Tensor] = []
        for blk in self.bottleneck:
            x = blk(x, history)   # 残差注意力块前向，融合全部历史特征
            history.append(x)     # 将当前层输出加入历史池，给下一层使用

        # 解码器上采样 + 跳跃连接拼接浅层特征
        d3 = self.dec3(x, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # 输出头映射到单通道MRI
        pred = self.head(d1)

        # 激活函数归一化到0~1
        if self.final_activation == "sigmoid":
            pred = torch.sigmoid(pred)

        if not return_features:
            return pred, None

        features = {"enc": [s1, s2, s3], "dec": [d1, d2, d3]}
        return pred, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播（训练/验证/测试默认入口）"""
        pred, _ = self._forward_backbone(x, return_features=False)
        return pred

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """训练阶段额外返回 Med-ReCL 所需的编码器/解码器多尺度特征。"""
        pred, features = self._forward_backbone(x, return_features=True)
        assert features is not None
        return pred, features

    def extract_target_features(
        self,
        target: torch.Tensor,
        use_teacher: bool = False,
    ) -> List[torch.Tensor]:
        """Extract online or stop-gradient EMA MRI target features."""
        if not self.use_medrecl or self.medrecl_target_encoder is None:
            raise RuntimeError("Med-ReCL target encoder is disabled.")
        if use_teacher:
            with torch.no_grad():
                return self.medrecl_target_encoder_ema(target.float())
        return self.medrecl_target_encoder(target)

    def project_medrecl_features(
        self,
        dec_feats: List[torch.Tensor],
        ct_feats: List[torch.Tensor],
        target_feats: List[torch.Tensor],
        teacher_target_feats: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Return raw appearance embeddings and normalized structure embeddings."""
        if not self.use_medrecl:
            raise RuntimeError("Med-ReCL projection heads are disabled.")

        projected = {
            "gen": [],
            "ct": [],
            "mri": [],
            "mri_teacher": [],
            "raw_gen": [],
            "raw_ct": [],
            "raw_mri": [],
            "raw_mri_teacher": [],
        }
        if teacher_target_feats is None:
            teacher_target_feats = target_feats
        for idx, (g_feat, x_feat, y_feat, y_teacher_feat) in enumerate(
            zip(dec_feats, ct_feats, target_feats, teacher_target_feats)
        ):
            raw_gen = self.medrecl_proj_g[idx](g_feat)
            raw_ct = self.medrecl_proj_x[idx](x_feat)
            raw_mri = self.medrecl_proj_y[idx](y_feat)
            with torch.no_grad():
                raw_mri_teacher = self.medrecl_proj_y_ema[idx](y_teacher_feat.float())
            # Appearance projection is delayed until after spatial sampling.
            # A 1x1x1 convolution is pointwise, so projecting sampled vectors is
            # mathematically identical while avoiding full-volume activations.
            projected["raw_gen"].append(g_feat)
            projected["raw_ct"].append(raw_ct)
            projected["raw_mri"].append(y_teacher_feat.detach())
            projected["raw_mri_teacher"].append(y_teacher_feat.detach())
            projected["gen"].append(F.normalize(raw_gen, dim=1, eps=1e-6))
            projected["ct"].append(F.normalize(raw_ct, dim=1, eps=1e-6))
            projected["mri"].append(F.normalize(raw_mri, dim=1, eps=1e-6))
            projected["mri_teacher"].append(
                F.normalize(raw_mri_teacher, dim=1, eps=1e-6)
            )
        return projected

    def project_medrecl_target_features(
        self,
        target_feats: List[torch.Tensor],
        use_teacher: bool = False,
    ) -> List[torch.Tensor]:
        """Project MRI features into the normalized structure space."""
        heads = self.medrecl_proj_y_ema if use_teacher else self.medrecl_proj_y
        outputs: List[torch.Tensor] = []
        for feature, head in zip(target_feats, heads):
            if use_teacher:
                with torch.no_grad():
                    projected = head(feature.float())
            else:
                projected = head(feature)
            outputs.append(F.normalize(projected, dim=1, eps=1e-6))
        return outputs

    def project_medrecl_appearance_vectors(
        self,
        vectors: torch.Tensor,
        level: int,
        use_teacher: bool = False,
    ) -> torch.Tensor:
        """Project sampled appearance vectors without materializing a 3D map."""
        heads = self.medrecl_app_proj_y_ema if use_teacher else self.medrecl_app_proj_g
        if use_teacher:
            with torch.no_grad():
                return heads[level].forward_vectors(vectors.float())
        return heads[level].forward_vectors(vectors.float())

    @torch.no_grad()                                     # 它是装饰器（修饰函数）。写在函数定义上方，作用是：进入这个函数后，关闭梯度计算、关闭计算图记录。
    def forward_mu_var_cov(
        self, source: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单次矩传播推理：本文主方法。

        输入一张 CT，只跑一次解析矩传播分支，同时得到：
        1. pred：重建 MRI 均值图；
        2. uncertainty：Moment propagation variance map。

        这个函数不更新参数：
        - 不调用 loss.backward()
        - 不调用 optimizer.step()
        - 只是把训练好的卷积权重、归一化参数、注意力权重代入矩传播公式

        传播规则：
        - 初始均值 μ_0 = source，表示输入 CT 本身作为确定输入；
        - 初始方差 var_0 = 0，表示这里暂不建模输入 CT 自身噪声，只估计模型/dropout引入的认知不确定性；
        - 编码器/解码器主要传播均值 μ 和对角方差 σ²；
        - bottleneck 注意力聚合处会临时引入通道协方差，用完即释放，兼顾估计精度和显存占用。

        Args:
            source: [B, 1, D, H, W]  输入原始3D CT体数据
        Returns:
            pred: [B, 1, D, H, W]  合成 MRI 图像
            uncertainty: [B, 1, D, H, W]  认知不确定性方差图 variance map
        """
        # 初始化矩：
        # source 直接作为输入均值；方差从 0 开始，后续由 dropout/非线性/注意力等层逐步传播得到。
        mu = source
        var = torch.zeros_like(source)

        # ---- 编码器阶段：基础卷积块矩传播，只传递均值+对角方差 ----
        mu_s1, var_s1 = self.enc1.forward_mu_var(mu, var)
        mu_s2, var_s2 = self.enc2.forward_mu_var(mu_s1, var_s1)
        mu_s3, var_s3 = self.enc3.forward_mu_var(mu_s2, var_s2)
        mu, var = self.enc4.forward_mu_var(mu_s3, var_s3)

        # ---- 瓶颈层阶段：带协方差计算的注意力残差块矩传播 ----
        history_mu: List[torch.Tensor] = []  # 保存每层瓶颈输出特征均值
        history_var: List[torch.Tensor] = [] # 保存每层瓶颈输出特征方差
        for blk in self.bottleneck:
            # 传入当前均值、方差与全部历史特征，块内部计算注意力协方差
            mu, var = blk.forward_mu_var_cov(mu, var, history_mu, history_var)
            # 本层输出存入历史列表，供给下一层注意力聚合使用
            history_mu.append(mu)
            history_var.append(var)

        # ---- 解码器阶段：上采样跳跃块矩传播，输入对应层跳跃特征的均值、方差 ----
        mu, var = self.dec3.forward_mu_var(mu, var, mu_s3, var_s3)
        mu, var = self.dec2.forward_mu_var(mu, var, mu_s2, var_s2)
        mu, var = self.dec1.forward_mu_var(mu, var, mu_s1, var_s1)

        # ---- 输出头 1×1卷积，分别对均值、方差做线性矩传播 ----
        # 均值支路：正常卷积得到未激活 MRI 预测均值。
        pred_mu = self.head(mu)
        # 方差支路：线性变换方差公式 Var(Wx) = W² * Var(x)，用权重平方卷积实现。
        pred_var = F.conv3d(var, self.head.weight ** 2, bias=None,
                            stride=self.head.stride, padding=self.head.padding)

        # 输出层如果使用 sigmoid，则均值先过 sigmoid，方差用一阶泰勒近似传播。
        if self.final_activation == "sigmoid":
            pred = torch.sigmoid(pred_mu)
            # Sigmoid导数 s'(z) = s(z)*(1-s(z))
            s = pred
            dsigmoid = s * (1.0 - s)
            # 非线性方差近似：Var[f(z)] ≈ [f'(μ)]² * Var(z)
            uncertainty = (dsigmoid ** 2) * pred_var
        else:
            # 无输出激活时，1x1卷积后的方差直接作为 variance map。
            pred = pred_mu
            uncertainty = pred_var

        # 数值误差可能让极小方差出现负数；方差物理意义必须非负，所以在出口统一截断。
        uncertainty = uncertainty.clamp_min(0.0)
        return pred, uncertainty

    @torch.no_grad()
    def mc_dropout_inference(
        self, source: torch.Tensor, num_passes: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout 推理（参考对照，不是本文默认主推理方法）。

        目的：
            使用同一个已经训练好的模型权重，对同一张 CT 重复前向 num_passes 次。
            每次只让 dropout mask 不同，得到一组不同预测，再对这些预测求方差。
            该方差图作为 MC variance map，用来和 forward_mu_var_cov() 的
            Moment variance map 做参考对比。

        重要实现细节：
            本函数不会把整个模型切成 train()。
            它先 self.eval()，让卷积、归一化、注意力等层保持稳定推理状态；
            然后只把 MomentDrop3d 层临时设为 train()，让 dropout 继续随机采样。
            这样得到的 MC 对照更干净，不会让其他层的训练态行为混入结果。

        这里同样不更新参数：
            不 backward，不 optimizer.step，只做重复推理和统计。

        Args:
            source: [B, 1, D, H, W]  输入3D CT
            num_passes: MC Dropout 推理次数，默认 200
        Returns:
            pred_mean: [B, 1, D, H, W]  多次推理平均后的重建 MRI
            pred_var: [B, 1, D, H, W]  多次预测的方差图，即 MC variance map
        """
        # 保存每个模块原始训练/评估状态，推理结束后逐一恢复，避免影响外部训练/验证流程。
        module_states = [(module, module.training) for module in self.modules()]
        self.eval()
        for module in self.modules():
            if isinstance(module, MomentDrop3d):
                module.train()

        try:
            pred_mean = None
            pred_m2 = None
            # Welford在线统计不保存全部200个体数据；模型前向仍可使用AMP，
            # 但均值和方差累积固定为FP32，降低显存并避免小方差下溢。
            for pass_index in range(num_passes):
                pred = self.forward(source).float()
                if pred_mean is None:
                    pred_mean = pred.clone()
                    pred_m2 = torch.zeros_like(pred_mean)
                    continue
                count = float(pass_index + 1)
                delta = pred - pred_mean
                pred_mean = pred_mean + delta / count
                pred_m2 = pred_m2 + delta * (pred - pred_mean)
        finally:
            # 无论中途是否报错，都恢复每个模块原始 training 状态。
            for module, training in module_states:
                module.training = training

        if pred_mean is None or pred_m2 is None:
            raise ValueError("num_passes must be at least 1")
        # unbiased=False：除以N；与原先 torch.var(..., unbiased=False)一致。
        pred_var = (pred_m2 / float(num_passes)).clamp_min(0.0)

        return pred_mean, pred_var


# ============================================================
# Training / Validation
# 训练与验证相关函数
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    amp,
    medrecl_criterion: Optional[MedReCLLoss] = None,
    start_step: int = 0,
    total_steps: int = 1,
    max_grad_norm: float = 5.0,
):
    """
    训练 1 个 epoch。

    本函数是真正更新模型参数的阶段：
    1. model.train()：打开训练态，MomentDrop3d 会随机失活中间特征；
    2. 前向得到预测 MRI；
    3. 计算重建损失，如果启用 Med-ReCL，则额外计算表征对比损失；
    4. loss.backward() 反向传播；
    5. optimizer.step() 更新卷积、归一化、注意力、Med-ReCL 投影头等可训练参数。

    注意：
        训练阶段不调用 forward_mu_var_cov()。
        矩传播不确定性是在训练完成后，加载 best.pth 做测试推理时才计算。

    参数：
        model:      神经网络模型
        loader:     训练数据加载器 DataLoader
        optimizer:  优化器
        criterion:  损失函数
        device:     设备（cpu或cuda）
        scaler:     混合精度训练用的梯度缩放器GradScaler
        amp:        是否开启自动混合精度 autocast

    返回：
        一个字典，包含该 epoch 的平均 loss / rec_loss / medrecl_loss / mae / psnr / ssim
    """
    model.train()
    running_loss = 0.0
    running_rec_loss = 0.0
    running_medrecl_loss = 0.0
    running_medrecl_weighted_loss = 0.0
    running_medrecl_weight = 0.0
    running_medrecl_structure_weight = 0.0
    running_medrecl_appearance_weight = 0.0
    running_medrecl_structure_scale = 0.0
    running_medrecl_appearance_scale = 0.0
    running_medrecl_structure_grad_ratio = 0.0
    running_medrecl_appearance_grad_ratio = 0.0
    running_medrecl_appearance_amplitude_loss = 0.0
    running_medrecl_appearance_contextual_loss = 0.0
    running_medrecl_appearance_stat_loss = 0.0
    running_medrecl_soft_positive_similarity = 0.0
    running_medrecl_false_negative_ratio = 0.0
    running_medrecl_invariance_loss = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_grad_norm = 0.0
    grad_norm_batches = 0
    skipped_optimizer_steps = 0
    consecutive_skipped_steps = 0
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
    optimizer_steps = 0
    ssim_fn = SSIM3D(channels=1).to(device)
    num_batches = 0

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp):
            if medrecl_criterion is not None:
                pred, feature_dict = model.forward_with_features(source)
                rec_loss = criterion(pred, target)
                current_step = start_step + num_batches + 1
                structure_weight, appearance_weight = medrecl_criterion.progressive_weights(
                    current_step=current_step,
                    total_steps=total_steps,
                )
                if structure_weight <= 0.0 and appearance_weight <= 0.0:
                    medrecl_loss = rec_loss.new_zeros(())
                    weighted_medrecl_loss = rec_loss.new_zeros(())
                    medrecl_weight = 0.0
                    medrecl_metrics = {
                        "structure_weight": 0.0,
                        "appearance_weight": 0.0,
                        "structure_gradient_scale": 1.0,
                        "appearance_gradient_scale": 1.0,
                        "structure_gradient_ratio": 0.0,
                        "appearance_gradient_ratio": 0.0,
                        "appearance_amplitude_loss": 0.0,
                        "appearance_contextual_loss": 0.0,
                        "appearance_stat_loss": 0.0,
                        "soft_positive_similarity": 0.0,
                        "false_negative_ratio": 0.0,
                        "invariance_loss": 0.0,
                    }
                else:
                    weighted_medrecl_loss, medrecl_metrics, medrecl_loss = medrecl_criterion.weighted_loss(
                        model=model,
                        source=source,
                        target=target,
                        pred=pred,
                        feature_dict=feature_dict,
                        rec_loss=rec_loss,
                        current_step=current_step,
                        total_steps=total_steps,
                    )
                    medrecl_weight = medrecl_metrics["effective_weight"]
                loss = rec_loss + weighted_medrecl_loss
            else:
                pred = model(source)
                rec_loss = criterion(pred, target)
                medrecl_loss = rec_loss.new_zeros(())
                weighted_medrecl_loss = rec_loss.new_zeros(())
                medrecl_weight = 0.0
                medrecl_metrics = {
                    "structure_weight": 0.0,
                    "appearance_weight": 0.0,
                    "structure_gradient_scale": 1.0,
                    "appearance_gradient_scale": 1.0,
                    "structure_gradient_ratio": 0.0,
                    "appearance_gradient_ratio": 0.0,
                    "appearance_amplitude_loss": 0.0,
                    "appearance_contextual_loss": 0.0,
                    "appearance_stat_loss": 0.0,
                    "soft_positive_similarity": 0.0,
                    "false_negative_ratio": 0.0,
                    "invariance_loss": 0.0,
                }
                loss = rec_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss before backward: "
                f"total={loss.detach().item()}, rec={rec_loss.detach().item()}, "
                f"medrecl={weighted_medrecl_loss.detach().item()}"
            )

        optimizer_step_succeeded = False
        grad_norm_value = float("nan")
        if scaler is not None and amp:
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=False,
            )
            grad_norm_value = float(grad_norm.detach().float().item())
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale >= old_scale and math.isfinite(grad_norm_value):
                optimizer_steps += 1
                optimizer_step_succeeded = True
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=False,
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
                raise FloatingPointError(
                    "AMP skipped 12 consecutive optimizer steps. "
                    "The run was stopped before producing a corrupted checkpoint."
                )
        if math.isfinite(grad_norm_value):
            running_grad_norm += grad_norm_value
            grad_norm_batches += 1

        if (
            optimizer_step_succeeded
            and medrecl_criterion is not None
            and hasattr(model, "update_medrecl_teacher")
        ):
            model.update_medrecl_teacher(
                momentum=medrecl_criterion.config.teacher_momentum,
            )

        with torch.no_grad():
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)
            mae = compute_mae(pred, target)
            psnr = compute_psnr(pred, target)
            ssim = ssim_fn(pred, target)

        running_loss += loss.item()
        running_rec_loss += rec_loss.item()
        running_medrecl_loss += medrecl_loss.item() if medrecl_criterion is not None else 0.0
        running_medrecl_weighted_loss += weighted_medrecl_loss.item() if medrecl_criterion is not None else 0.0
        running_medrecl_weight += medrecl_weight
        running_medrecl_structure_weight += float(medrecl_metrics["structure_weight"])
        running_medrecl_appearance_weight += float(medrecl_metrics["appearance_weight"])
        running_medrecl_structure_scale += float(medrecl_metrics["structure_gradient_scale"])
        running_medrecl_appearance_scale += float(medrecl_metrics["appearance_gradient_scale"])
        running_medrecl_structure_grad_ratio += float(medrecl_metrics["structure_gradient_ratio"])
        running_medrecl_appearance_grad_ratio += float(medrecl_metrics["appearance_gradient_ratio"])
        running_medrecl_appearance_amplitude_loss += float(
            medrecl_metrics["appearance_amplitude_loss"]
        )
        running_medrecl_appearance_contextual_loss += float(
            medrecl_metrics["appearance_contextual_loss"]
        )
        running_medrecl_appearance_stat_loss += float(
            medrecl_metrics["appearance_stat_loss"]
        )
        running_medrecl_soft_positive_similarity += float(
            medrecl_metrics["soft_positive_similarity"]
        )
        running_medrecl_false_negative_ratio += float(
            medrecl_metrics["false_negative_ratio"]
        )
        running_medrecl_invariance_loss += float(
            medrecl_metrics["invariance_loss"]
        )
        running_mae += mae.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()
        batch_rec_components = getattr(criterion, "last_components", {})
        for name in rec_component_names:
            running_rec_components[name] += float(batch_rec_components.get(name, 0.0))
        num_batches += 1

    results = {
        "loss": running_loss / max(1, num_batches),
        "rec_loss": running_rec_loss / max(1, num_batches),
        "medrecl_loss": running_medrecl_loss / max(1, num_batches),
        "medrecl_weighted_loss": running_medrecl_weighted_loss / max(1, num_batches),
        "medrecl_weight": running_medrecl_weight / max(1, num_batches),
        "medrecl_structure_weight": running_medrecl_structure_weight / max(1, num_batches),
        "medrecl_appearance_weight": running_medrecl_appearance_weight / max(1, num_batches),
        "medrecl_structure_gradient_scale": running_medrecl_structure_scale / max(1, num_batches),
        "medrecl_appearance_gradient_scale": running_medrecl_appearance_scale / max(1, num_batches),
        "medrecl_structure_gradient_ratio": running_medrecl_structure_grad_ratio / max(1, num_batches),
        "medrecl_appearance_gradient_ratio": running_medrecl_appearance_grad_ratio / max(1, num_batches),
        "medrecl_appearance_amplitude_loss": running_medrecl_appearance_amplitude_loss / max(1, num_batches),
        "medrecl_appearance_contextual_loss": running_medrecl_appearance_contextual_loss / max(1, num_batches),
        "medrecl_appearance_stat_loss": running_medrecl_appearance_stat_loss / max(1, num_batches),
        "medrecl_soft_positive_similarity": running_medrecl_soft_positive_similarity / max(1, num_batches),
        "medrecl_false_negative_ratio": running_medrecl_false_negative_ratio / max(1, num_batches),
        "medrecl_invariance_loss": running_medrecl_invariance_loss / max(1, num_batches),
        "mae": running_mae / max(1, num_batches),
        "psnr": running_psnr / max(1, num_batches),
        "ssim": running_ssim / max(1, num_batches),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "grad_norm": running_grad_norm / max(1, grad_norm_batches),
        "amp_scale": float(scaler.get_scale()) if scaler is not None else 1.0,
        "next_step": start_step + num_batches,
    }
    for name in rec_component_names:
        results[f"rec_{name}"] = running_rec_components[name] / max(1, num_batches)
    return results


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    amp,
    divisor: int = 8,
    eval_crop_size: Optional[Tuple[int, int, int]] = None,
):
    """
    验证或普通测试 1 个 epoch。

    本函数只评估重建质量，不更新权重：
    1. model.eval()：关闭 dropout 随机失活；
    2. torch.no_grad()：不记录计算图，不做 backward；
    3. 使用普通 forward() 输出预测 MRI；
    4. 计算 loss / MAE / PSNR / SSIM。

    它不输出 uncertainty map。
    默认测试图里的 Moment variance 是由 train.py 中 save_test_comparison_figures()
    额外调用 forward_mu_var_cov() 得到的。

    参数：
        model:      神经网络模型
        loader:     验证数据加载器 DataLoader
        criterion:  损失函数
        device:     设备（cpu 或 cuda）
        amp:        是否开启自动混合精度
        divisor:    输入尺寸需要被多少整除（例如 8）

    返回：
        一个字典，包含该 epoch 的平均 loss / mae / psnr / ssim
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_foreground_mae = 0.0
    running_foreground_psnr = 0.0
    running_foreground_ssim = 0.0
    running_gradient_mae = 0.0
    running_hfen = 0.0
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
    num_batches = 0

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source, target = crop_pair_around_foreground(source, target, eval_crop_size)

        source, pad = pad_tensor_to_divisible(source, divisor=divisor)

        with autocast(device_type=device.type, enabled=amp):
            pred = model(source)
            pred = unpad_tensor(pred, pad)
            loss = criterion(pred, target)

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        mae = compute_mae(pred, target)
        psnr = compute_psnr(pred, target)
        ssim = ssim_fn(pred, target)
        foreground_mask = target > 0.01
        foreground_mae = compute_masked_mae(pred, target, foreground_mask)
        foreground_psnr = compute_masked_psnr(pred, target, foreground_mask)
        foreground_ssim = ssim_fn(pred, target, mask=foreground_mask)
        gradient_mae = compute_gradient_mae(pred, target, mask=foreground_mask)
        hfen = compute_hfen(pred, target, mask=foreground_mask)

        running_loss += loss.item()
        running_mae += mae.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()
        running_foreground_mae += foreground_mae.item()
        running_foreground_psnr += foreground_psnr.item()
        running_foreground_ssim += foreground_ssim.item()
        running_gradient_mae += gradient_mae.item()
        running_hfen += hfen.item()
        batch_rec_components = getattr(criterion, "last_components", {})
        for name in rec_component_names:
            running_rec_components[name] += float(batch_rec_components.get(name, 0.0))
        num_batches += 1

    results = {
        "loss": running_loss / max(1, num_batches),
        "mae": running_mae / max(1, num_batches),
        "psnr": running_psnr / max(1, num_batches),
        "ssim": running_ssim / max(1, num_batches),
        "foreground_mae": running_foreground_mae / max(1, num_batches),
        "foreground_psnr": running_foreground_psnr / max(1, num_batches),
        "foreground_ssim": running_foreground_ssim / max(1, num_batches),
        "gradient_mae": running_gradient_mae / max(1, num_batches),
        "hfen": running_hfen / max(1, num_batches),
    }
    for name in rec_component_names:
        results[f"rec_{name}"] = running_rec_components[name] / max(1, num_batches)
    return results


@torch.no_grad()
def validate_with_uncertainty(
    model: AttnResCTtoMRI,
    loader,
    device,
    amp: bool = True,
    divisor: int = 8,
    mc_passes: int = 200,
    save_uncertainty_dir: Optional[str] = None,
    csv_log_path: Optional[str] = None,
    eval_crop_size: Optional[Tuple[int, int, int]] = None,
    case_callback=None,
    moment_calibration_scale: float = 1.0,
) -> Dict:
    """
    测试阶段的不确定性对比函数。

    这个函数只在需要 MC Dropout 参考对照时调用。
    它不是训练函数，不更新权重，也不改变 best.pth。

    两条分支：
    1. Moment propagation（本文主方法）
       调用 forward_mu_var_cov()，单次前向得到：
           pred_moment
           uncertainty_moment_var

    2. MC Dropout（参考对照）
       调用 mc_dropout_inference()，重复 num_passes 次普通 forward，
       得到：
           pred_mc
           uncertainty_mc_var

    对比指标：
    - unc_rmsd_vs_mc：Moment variance 与 MC variance 在脑区 mask 内的 RMSD；
    - unc_mae_brain_vs_mc：两张方差图在脑区 mask 内的平均绝对差。
    - Pearson/Spearman：Moment 与 MC，以及 uncertainty 与重建误差的相关性；
    - top10 overlap：最高 10% 不确定区域覆盖最高 10% 重建误差区域的比例；
    - 推理时间、峰值显存和前向次数。

    输出单位说明：
    - uncertainty_moment_var 和 uncertainty_mc_var 都是 variance map；
    - 这里不把 variance 开平方成 std，所以不要把它写成 HU/std 单位。

    参数：
        model:           AttnResCTtoMRI模型
        loader:          数据加载器
        device:          设备
        amp:             是否使用AMP
        divisor:         尺寸整除要求
        mc_passes:       MC Dropout推理次数，默认200
        save_uncertainty_dir: 可选；不为空时保存 Moment/MC 方差图和 pred_moment 的 nii.gz
        csv_log_path:    可选，CSV日志路径

    返回：
        包含重建指标和不确定性方差对比指标的字典。
    """
    model.eval()

    ssim_fn = SSIM3D(channels=1).to(device)
    num_batches = 0

    # 累积普通重建指标和方差图对比指标。
    running_loss = 0.0
    running_mae = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_mc_mae = 0.0
    running_mc_psnr = 0.0
    running_mc_ssim = 0.0
    running_unc_rmsd = 0.0        # 不确定性方差图RMSD (单次矩传播 vs MC Dropout)
    running_unc_mae_brain = 0.0   # 全脑平均不确定性方差差异
    metric_names = (
        "unc_pearson_moment_vs_mc",
        "unc_spearman_moment_vs_mc",
        "moment_error_pearson",
        "moment_error_spearman",
        "mc_error_pearson",
        "mc_error_spearman",
        "moment_top10_error_overlap",
        "mc_top10_error_overlap",
        "moment_raw_variance_error_mae",
        "moment_calibrated_variance_error_mae",
        "moment_inference_seconds",
        "mc_inference_seconds",
        "moment_peak_memory_mb",
        "mc_peak_memory_mb",
    )
    running_metrics = {name: 0.0 for name in metric_names}
    case_rows = []

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source, target = crop_pair_around_foreground(source, target, eval_crop_size)
        case_id = batch.get("case_id", [f"case_{num_batches}"])[0] if isinstance(
            batch.get("case_id", ["unknown"]), (list, tuple)
        ) else batch.get("case_id", f"case_{num_batches}")

        source_padded, pad = pad_tensor_to_divisible(source, divisor=divisor)

        # ---- 分支 1: 单次矩传播推理（本文方法） ----
        # 只跑一次解析矩传播，直接得到 Moment variance map。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            moment_memory_start = torch.cuda.memory_allocated(device)
        else:
            moment_memory_start = 0
        moment_start = time.perf_counter()
        with autocast(device_type=device.type, enabled=amp):
            pred_moment, uncertainty_moment_var = model.forward_mu_var_cov(source_padded)
            pred_moment = unpad_tensor(pred_moment, pad)
            uncertainty_moment_var = unpad_tensor(uncertainty_moment_var, pad)
            uncertainty_moment_var = uncertainty_moment_var.float().clamp_min(0.0)
            uncertainty_moment_calibrated_var = (
                uncertainty_moment_var * max(0.0, float(moment_calibration_scale))
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            moment_peak_memory = max(
                0,
                torch.cuda.max_memory_allocated(device) - moment_memory_start,
            ) / (1024.0 ** 2)
        else:
            moment_peak_memory = 0.0
        moment_seconds = time.perf_counter() - moment_start

        # ---- 分支 2: MC Dropout多次推理（参考对照） ----
        # 重复 forward num_passes 次，对预测结果求方差，得到 MC variance map。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            mc_memory_start = torch.cuda.memory_allocated(device)
        else:
            mc_memory_start = 0
        mc_start = time.perf_counter()
        with autocast(device_type=device.type, enabled=amp):
            pred_mc, uncertainty_mc_var = model.mc_dropout_inference(
                source_padded, num_passes=mc_passes
            )
            pred_mc = unpad_tensor(pred_mc, pad)
            uncertainty_mc_var = unpad_tensor(uncertainty_mc_var, pad)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            mc_peak_memory = max(
                0,
                torch.cuda.max_memory_allocated(device) - mc_memory_start,
            ) / (1024.0 ** 2)
        else:
            mc_peak_memory = 0.0
        mc_seconds = time.perf_counter() - mc_start

        # ---- 重建质量评估 ----
        # 重建指标使用本文主方法的 pred_moment 计算。
        pred = torch.clamp(pred_moment, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)

        loss = F.l1_loss(pred, target_c)
        mae = compute_mae(pred, target_c)
        psnr = compute_psnr(pred, target_c)
        ssim = ssim_fn(pred, target_c)
        pred_mc_c = torch.clamp(pred_mc, 0.0, 1.0)
        mc_mae = compute_mae(pred_mc_c, target_c)
        mc_psnr = compute_psnr(pred_mc_c, target_c)
        mc_ssim = ssim_fn(pred_mc_c, target_c)

        # ---- 不确定性方差图对比 ----
        # 使用 target>0.01 的简单 mask 排除大面积背景空气，使方差图差异主要来自有效组织区域。
        brain_mask = target_c > 0.01  # 简单阈值mask（排除背景空气）
        if brain_mask.sum() > 0:
            moment_var_values = uncertainty_moment_var.clamp_min(0.0)[brain_mask]
            moment_calibrated_values = uncertainty_moment_calibrated_var[brain_mask]
            mc_var_values = uncertainty_mc_var.clamp_min(0.0)[brain_mask]
            moment_error_values = torch.abs(pred - target_c)[brain_mask]
            moment_squared_error_values = (pred - target_c).square()[brain_mask]
            mc_error_values = torch.abs(pred_mc_c - target_c)[brain_mask]
            diff = moment_var_values - mc_var_values
            unc_rmsd = torch.sqrt(torch.mean(diff ** 2))
            unc_mae = torch.mean(torch.abs(diff))
            case_metrics = {
                "unc_pearson_moment_vs_mc": compute_pearson_correlation(
                    moment_var_values, mc_var_values
                ).item(),
                "unc_spearman_moment_vs_mc": compute_spearman_correlation(
                    moment_var_values, mc_var_values
                ).item(),
                "moment_error_pearson": compute_pearson_correlation(
                    moment_var_values, moment_error_values
                ).item(),
                "moment_error_spearman": compute_spearman_correlation(
                    moment_var_values, moment_error_values
                ).item(),
                "mc_error_pearson": compute_pearson_correlation(
                    mc_var_values, mc_error_values
                ).item(),
                "mc_error_spearman": compute_spearman_correlation(
                    mc_var_values, mc_error_values
                ).item(),
                "moment_top10_error_overlap": compute_top_quantile_overlap(
                    moment_var_values, moment_error_values
                ).item(),
                "mc_top10_error_overlap": compute_top_quantile_overlap(
                    mc_var_values, mc_error_values
                ).item(),
                "moment_raw_variance_error_mae": torch.mean(
                    torch.abs(moment_var_values - moment_squared_error_values)
                ).item(),
                "moment_calibrated_variance_error_mae": torch.mean(
                    torch.abs(moment_calibrated_values - moment_squared_error_values)
                ).item(),
            }
        else:
            unc_rmsd = torch.tensor(0.0, device=device)
            unc_mae = torch.tensor(0.0, device=device)
            case_metrics = {
                name: 0.0
                for name in metric_names
                if not name.endswith("_seconds") and "memory" not in name
            }
        case_metrics.update({
            "moment_inference_seconds": float(moment_seconds),
            "mc_inference_seconds": float(mc_seconds),
            "moment_peak_memory_mb": float(moment_peak_memory),
            "mc_peak_memory_mb": float(mc_peak_memory),
        })

        running_loss += loss.item()
        running_mae += mae.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()
        running_mc_mae += mc_mae.item()
        running_mc_psnr += mc_psnr.item()
        running_mc_ssim += mc_ssim.item()
        running_unc_rmsd += unc_rmsd.item()
        running_unc_mae_brain += unc_mae.item()
        for name in metric_names:
            running_metrics[name] += float(case_metrics[name])
        num_batches += 1

        case_row = {
            "case_id": str(case_id),
            "moment_forward_count": 1,
            "mc_forward_count": int(mc_passes),
            "moment_calibration_scale": float(moment_calibration_scale),
            "moment_mae": float(mae.item()),
            "moment_psnr": float(psnr.item()),
            "moment_ssim": float(ssim.item()),
            "mc_mean_mae": float(mc_mae.item()),
            "mc_mean_psnr": float(mc_psnr.item()),
            "mc_mean_ssim": float(mc_ssim.item()),
            "unc_variance_rmsd_brain": float(unc_rmsd.item()),
            "unc_variance_mae_brain": float(unc_mae.item()),
            **case_metrics,
        }
        case_rows.append(case_row)

        if case_callback is not None:
            case_callback(
                case_id=str(case_id),
                source=source,
                target=target_c,
                pred_moment=pred,
                uncertainty_moment_var=uncertainty_moment_var.clamp_min(0.0),
                uncertainty_moment_calibrated_var=(
                    uncertainty_moment_calibrated_var.clamp_min(0.0)
                ),
                pred_mc=pred_mc_c,
                uncertainty_mc_var=uncertainty_mc_var.clamp_min(0.0),
                mc_passes=int(mc_passes),
            )

        # ---- 可选: 保存不确定性方差 NIfTI ----
        # 只有传入 save_uncertainty_dir 时才保存体数据；否则只返回指标。
        if save_uncertainty_dir is not None:
            save_dir = Path(save_uncertainty_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            for name, arr in [
                ("uncertainty_moment_variance_raw", uncertainty_moment_var),
                ("uncertainty_moment_variance_calibrated", uncertainty_moment_calibrated_var),
                ("uncertainty_mc_variance", uncertainty_mc_var),
                ("pred_moment", pred_moment),
                ("pred_mc_mean", pred_mc),
            ]:
                nii_data = arr.squeeze().cpu().numpy()
                nii_img = nib.Nifti1Image(nii_data, np.eye(4))
                fname = save_dir / f"{case_id}_{name}.nii.gz"
                nib.save(nii_img, str(fname))

    n = max(1, num_batches)
    results = {
        "loss": running_loss / n,
        "mae": running_mae / n,
        "psnr": running_psnr / n,
        "ssim": running_ssim / n,
        "mc_mean_mae": running_mc_mae / n,
        "mc_mean_psnr": running_mc_psnr / n,
        "mc_mean_ssim": running_mc_ssim / n,
        "unc_rmsd_vs_mc": running_unc_rmsd / n,
        "unc_mae_brain_vs_mc": running_unc_mae_brain / n,
        "moment_forward_count": 1,
        "mc_forward_count": int(mc_passes),
        "moment_calibration_scale": float(moment_calibration_scale),
    }
    for name in metric_names:
        results[name] = running_metrics[name] / n
    results["mc_over_moment_time_ratio"] = (
        results["mc_inference_seconds"]
        / max(results["moment_inference_seconds"], 1e-12)
    )

    # ---- 可选: 写入CSV日志 ----
    if csv_log_path is not None:
        summary_row = {
            "case_id": "MEAN",
            "moment_forward_count": 1,
            "mc_forward_count": int(mc_passes),
            "moment_calibration_scale": float(moment_calibration_scale),
            "moment_mae": results["mae"],
            "moment_psnr": results["psnr"],
            "moment_ssim": results["ssim"],
            "mc_mean_mae": results["mc_mean_mae"],
            "mc_mean_psnr": results["mc_mean_psnr"],
            "mc_mean_ssim": results["mc_mean_ssim"],
            "unc_variance_rmsd_brain": results["unc_rmsd_vs_mc"],
            "unc_variance_mae_brain": results["unc_mae_brain_vs_mc"],
            **{name: results[name] for name in metric_names},
        }
        csv_path = Path(csv_log_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(summary_row.keys())
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in case_rows:
                writer.writerow(row)
            writer.writerow(summary_row)

    return results


@torch.no_grad()
def fit_moment_variance_scale(
    model: AttnResCTtoMRI,
    loader,
    device,
    amp: bool = True,
    divisor: int = 8,
    eval_crop_size: Optional[Tuple[int, int, int]] = None,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> Dict[str, float]:
    """Fit U_cal=sU on validation squared error, never on test labels."""
    model.eval()
    numerator = 0.0
    denominator = 0.0
    raw_error = 0.0
    calibrated_error = 0.0
    voxel_count = 0
    case_count = 0

    cached_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source, target = crop_pair_around_foreground(source, target, eval_crop_size)
        source_padded, pad = pad_tensor_to_divisible(source, divisor=divisor)
        with autocast(device_type=device.type, enabled=amp):
            pred, variance = model.forward_mu_var_cov(source_padded)
            pred = unpad_tensor(pred, pad)
            variance = unpad_tensor(variance, pad)
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        variance = variance.float().clamp_min(0.0)
        squared_error = (pred - target).square()
        mask = target > 0.01
        if not bool(mask.any()):
            continue
        u = variance[mask].detach().cpu()
        e2 = squared_error[mask].detach().cpu()
        numerator += float(torch.sum(u * e2).item())
        denominator += float(torch.sum(u.square()).item())
        raw_error += float(torch.sum(torch.abs(u - e2)).item())
        voxel_count += int(u.numel())
        case_count += 1
        cached_pairs.append((u, e2))

    if denominator <= 1e-20 or case_count == 0:
        scale = 1.0
    else:
        scale = max(min_scale, min(max_scale, numerator / denominator))
    for u, e2 in cached_pairs:
        calibrated_error += float(torch.sum(torch.abs(scale * u - e2)).item())
    count = max(1, voxel_count)
    return {
        "scale": float(scale),
        "validation_cases": int(case_count),
        "validation_voxels": int(voxel_count),
        "raw_variance_to_squared_error_mae": raw_error / count,
        "calibrated_variance_to_squared_error_mae": calibrated_error / count,
    }


# 保存训练检查点。
# best.pth 和 epoch_xxxx.pth 都通过这个函数写出；
# 其中 best.pth 用验证集最优 SSIM 选择，epoch_xxxx.pth 用于中断后续训。
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
    args,
    scheduler=None,
    scaler=None,
    global_step: Optional[int] = None,
    metric_name: str = "ssim",
    best_metrics: Optional[Dict[str, float]] = None,
):
    """
    保存训练状态到 pth 文件。

    保存内容不只有模型权重，还包括：
    - optimizer：保证 resume 后优化器动量/自适应学习率状态不丢；
    - scheduler：保证学习率调度从正确位置继续；
    - scaler：保证 AMP 混合精度训练可以稳定续训；
    - epoch / best_metric / global_step：保证日志和 Med-ReCL warmup 步数连续。

    参数：
        path:         保存路径
        model:        模型
        optimizer:    优化器
        epoch:        当前 epoch
        best_metric:  当前记录到的最佳指标
        args:         训练参数
        scheduler:    可选，学习率调度器
        scaler:       可选，AMP梯度缩放器
        global_step:  可选，全局训练步数
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }
    if best_metrics is not None:
        ckpt["best_metrics"] = {
            name: float(value) for name, value in best_metrics.items()
        }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if global_step is not None:
        ckpt["global_step"] = global_step
    ckpt["rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, str(path))


# 把一行指标写进 CSV 文件。
def write_log_row(csv_path, row):
    """
    将一行实验指标写入 CSV 日志文件。

    如果 CSV 不存在，会先写表头；如果已经存在，则直接追加一行。
    train.py 中的 log.csv、test_results.csv、uncertainty_results.csv 都使用这个函数。

    参数：
        csv_path: CSV文件路径
        row:      字典；key 是列名，value 是该列要写入的数值或字符串
    """
    csv_path = Path(csv_path)
    incoming_fields = list(row.keys())
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=incoming_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        existing_rows = list(reader)

    merged_fields = existing_fields + [
        name for name in incoming_fields if name not in existing_fields
    ]
    if merged_fields != existing_fields:
        # Resume can introduce new metrics into an existing log. Rewrite the
        # same CSV with a union header so historical and new epochs stay aligned.
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writerow(row)
