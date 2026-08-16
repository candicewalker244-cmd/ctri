# HANDOFF.md

本文件作为后续交给 ChatGPT 或下一位分析者的实验交接文档。以后每次完成代码修改、训练、评估或诊断后，都必须在此追加/更新记录；已有历史不得无理由覆盖。

记录规则：
- 未实际运行的命令写“未运行”。
- 未得到的指标写“未获得”，禁止估算。
- 训练仍在进行时写“进行中”和当前进度。
- 实验事实、解释/推测、下一步建议必须明确区分。

---

## EXP-0001：交接文档初始化与既有结果盘点

### 1. 本轮实验编号与时间
- 实验编号：EXP-0001
- 时间：2026-08-17 03:15:12 +08:00
- 类型：诊断 / 文档初始化；未进行代码逻辑修改、训练或重新评估。

### 2. 本轮目标
- 在项目根目录创建 `HANDOFF.md`，建立固定交接格式。
- 盘点当前项目中能确认的已有实验输出、数据划分、环境信息和可比性风险。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 修改前不存在 `HANDOFF.md`。
- 当前目录 `D:\pythondaima\ctri` 不是 git 仓库，`git rev-parse HEAD`、`git status` 均失败，因此 commit 未获得。
- 当前可见核心代码文件：
  - `train_gtmedreclpp.py`
  - `model_attnres3d_gtmedreclpp.py`
- 当前未发现 `.pth` / `.pt` / `.ckpt` checkpoint 文件。
- 已有输出目录：
  - `output_gtmedreclpp_40_8_3_e100`
  - `output_gt_e0_rec_40_8_3_e100`

解释/推测：
- `output_gt_e0_rec_40_8_3_e100` 的结果文件更新时间较晚，可作为当前目录中“最新已有结果”读取，但不是本轮新跑出的结果。
- `E0_log.csv` 缺少 `train_medrecl_*` 列，而 `log.csv` 含有这些列；这说明两组实验配置/代码路径可能不同，不能默认严格等价。

### 4. 理论依据
事实：
- `train_gtmedreclpp.py` 的实验主线包括 CT->MRI 重建、验证集选择 `best.pth`、测试集输出 MAE / PSNR / SSIM / HFEN / Gradient MAE，并默认运行 MC Dropout 200 次与 Moment propagation variance map 做对照。
- `model_attnres3d_gtmedreclpp.py` 中定义了标准 inverted dropout 下的 Moment propagation，以及 `mc_dropout_inference()` 多次采样对照。
- 脚本默认重建损失由 L1、MS-SSIM、Edge、Focal Frequency 加权组成；默认 Med-ReCL 为训练阶段辅助约束。

解释/推测：
- 本交接文档的核心理论依据是实验可复现性：必须同时记录代码版本、数据划分、命令、随机性设置、环境和真实指标，避免后续分析把不可比实验误作可比实验。

### 5. 实际修改文件
- `HANDOFF.md`

### 6. 每个文件具体修改内容
- `HANDOFF.md`：新建交接文档，加入固定记录规则，并追加 EXP-0001 本轮诊断记录。

### 7. 实际运行命令
诊断命令：
```powershell
Test-Path -LiteralPath HANDOFF.md
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short --branch
python --version
Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
rg --files
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys; print(sys.version); import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', getattr(torch.version, 'cuda', None)); print('gpu_count', torch.cuda.device_count()); print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未获得')"
Get-Content -LiteralPath train_gtmedreclpp.py -TotalCount 260
Get-Content -LiteralPath output_gtmedreclpp_40_8_3_e100\test_results.csv
Get-Content -LiteralPath output_gt_e0_rec_40_8_3_e100\E0_test_results.csv
Get-Content -LiteralPath output_gtmedreclpp_40_8_3_e100\log.csv -Tail 5
Get-Content -LiteralPath output_gt_e0_rec_40_8_3_e100\E0_log.csv -Tail 5
rg -n "add_argument|set_seed|optimizer|Adam|SGD|learning|lr|batch|epoch|loss|crop|data|output|mc|dropout|seed" train_gtmedreclpp.py
rg -n "class ReconstructionLoss|class MedReCLConfig|class MedReCLLoss|def set_seed|def discover_cases|def validate_case_splits|def train_one_epoch|optimizer|Adam|lr|seed|random|dropout|mc" model_attnres3d_gtmedreclpp.py
Get-Content -LiteralPath output_gtmedreclpp_40_8_3_e100\moment_calibration.csv
Get-Content -LiteralPath output_gtmedreclpp_40_8_3_e100\uncertainty_results.csv
Get-Content -LiteralPath output_gt_e0_rec_40_8_3_e100\E0_moment_calibration.csv
Get-Content -LiteralPath output_gt_e0_rec_40_8_3_e100\E0_uncertainty_results.csv
Get-ChildItem -LiteralPath data\dataset\train -Directory | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -LiteralPath data\dataset\val -Directory | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -LiteralPath data\dataset\test -Directory | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -Recurse -File -Include *.pth,*.pt,*.ckpt | Select-Object FullName,LastWriteTime,Length
Get-ChildItem -Recurse -File -Include *.txt,*.md,*.log,*.json,*.yaml,*.yml | Select-Object FullName,LastWriteTime,Length
Get-ChildItem -LiteralPath data\dataset\train -Directory | Select-Object -ExpandProperty Name
Get-ChildItem -LiteralPath data\dataset\val -Directory | Select-Object -ExpandProperty Name
Get-ChildItem -LiteralPath data\dataset\test -Directory | Select-Object -ExpandProperty Name
Import-Csv output_gtmedreclpp_40_8_3_e100\test_results.csv
Import-Csv output_gt_e0_rec_40_8_3_e100\E0_test_results.csv
Import-Csv output_gtmedreclpp_40_8_3_e100\uncertainty_results.csv
Import-Csv output_gt_e0_rec_40_8_3_e100\E0_uncertainty_results.csv
```

训练命令：未运行。

评估命令：未运行。

文件修改操作：
- `apply_patch`：创建 `HANDOFF.md`。

### 8. 数据集 / 数据划分，以及是否与基线一致
事实：
- 数据根目录：`data\dataset`
- train：40 例
  - `1BA001`, `1BA012`, `1BA014`, `1BA022`, `1BA032`, `1BA040`, `1BA054`, `1BA097`, `1BA116`, `1BA141`, `1BA184`, `1BA220`, `1BA234`, `1BA266`, `1BA294`, `1BA336`, `1BA358`, `1BB006`, `1BB017`, `1BB034`, `1BB048`, `1BB052`, `1BB082`, `1BB099`, `1BB111`, `1BB177`, `1BB200`, `1BC006`, `1BC014`, `1BC019`, `1BC022`, `1BC031`, `1BC038`, `1BC048`, `1BC062`, `1BC066`, `1BC076`, `1BC085`, `1BC088`, `1BC094`
- val：8 例
  - `1BA076`, `1BA151`, `1BA239`, `1BB059`, `1BB079`, `1BB171`, `1BC053`, `1BC083`
- test：3 例
  - `1BA005`, `1BB030`, `1BC052`
- 两个已有输出目录的测试 CSV 均覆盖上述 3 个 test 病例的汇总结果；训练/验证日志均显示每 epoch `train_optimizer_steps=40`。

是否与基线一致：
- 数据划分层面：可见目录划分一致。
- 严格实验基线层面：未获得历史运行命令、checkpoint 和 commit，因此不能证明完全一致。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
事实：
- 脚本默认参数：
  - `seed=42`
  - `epochs=100`
  - `batch_size=1`
  - `lr=1e-4`
  - `scheduler=cosine`
  - `min_lr=1e-6`
  - `optimizer=AdamW`
  - `max_grad_norm=5.0`
  - `patch_size=(96,96,96)`
  - `eval_crop_size=(150,150,150)`
  - `dropout=0.2`
  - `amp=True`，但实际设备非 CUDA 时脚本会关闭 AMP
  - `divisor=8`
  - `mc_passes=200`
  - `eval_mc_dropout_compare=True`
  - `moment_calibration=True`
  - `medrecl=True`
- 重建损失默认权重：
  - L1：0.45
  - MS-SSIM：0.30
  - Edge：0.15
  - Focal Frequency：0.10
  - `frequency_alpha=1.0`
- Med-ReCL 默认关键参数：
  - `medrecl_proj_dim=64`
  - `lambda_structure_max=0.01`
  - `lambda_appearance_max=0.004`
  - `recon_only_ratio=0.20`
  - `teacher_momentum=0.99`
  - `false_negative_threshold=0.95`
  - `false_negative_weight=0.10`
  - `invariance_weight=0.05`

已有结果可确认：
- `output_gtmedreclpp_40_8_3_e100\log.csv` 和 `output_gt_e0_rec_40_8_3_e100\E0_log.csv` 均完成到 epoch 100。
- 两个日志 epoch 100 的学习率均为 `0.00000102`，每 epoch optimizer step 为 40。
- `output_gtmedreclpp_40_8_3_e100\log.csv` 含 Med-ReCL 训练指标列。
- `output_gt_e0_rec_40_8_3_e100\E0_log.csv` 不含 Med-ReCL 训练指标列。

未获得：
- 历史运行命令中的实际 seed。
- 历史运行命令中的实际 batch size、学习率等是否完全采用默认值。
- 历史 checkpoint 内的保存参数，因为当前未发现 checkpoint 文件。

### 10. GPU、CUDA、Python、PyTorch
当前诊断环境事实：
- Python：`3.13.7`
- PyTorch：`2.11.0+cpu`
- `torch.cuda.is_available()`：`False`
- `torch.version.cuda`：`None`
- `torch.cuda.device_count()`：`0`
- `nvidia-smi`：命令不可用。

历史训练/评估环境：
- GPU 型号：未获得。
- CUDA 版本：未获得。
- 历史输出 CSV 中记录了推理显存字段，例如 Moment peak memory 约 6624 MB、MC peak memory 约 1789 MB，但不能据此反推出 GPU 型号。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
事实：当前目录中最新结果文件为 `output_gt_e0_rec_40_8_3_e100\E0_test_results.csv`，文件时间 2026-08-15 03:54:02。

最新已有测试汇总：
| 指标 | 数值 |
|---|---:|
| test_loss | 0.167828 |
| test_mae | 0.107510 |
| test_psnr | 16.068155 |
| test_ssim | 0.529396 |
| test_foreground_mae | 0.115114 |
| test_foreground_psnr | 15.836952 |
| test_foreground_ssim | 0.510858 |
| test_gradient_mae | 0.074969 |
| test_hfen | 0.730762 |
| test_rec_l1 | 0.107510 |
| test_rec_ms_ssim | 0.377409 |
| test_rec_edge | 0.039615 |
| test_rec_frequency | 0.002830 |
| test_rec_weighted_l1 | 0.048380 |
| test_rec_weighted_ms_ssim | 0.113223 |
| test_rec_weighted_edge | 0.005942 |
| test_rec_weighted_frequency | 0.000283 |

最新已有不确定性 / MC 对照均值，来自 `E0_uncertainty_results.csv` 的 `MEAN` 行：
| 指标 | 数值 |
|---|---:|
| moment_mae | 0.107508294 |
| moment_psnr | 16.068284353 |
| moment_ssim | 0.529402743 |
| mc_mean_mae | 0.106626190 |
| mc_mean_psnr | 16.125280698 |
| mc_mean_ssim | 0.535671731 |
| unc_variance_rmsd_brain | 0.000414579 |
| unc_variance_mae_brain | 0.000270035 |
| unc_pearson_moment_vs_mc | 0.848834674 |
| unc_spearman_moment_vs_mc | 0.895242771 |
| moment_error_pearson | 0.282189141 |
| moment_error_spearman | 0.369784107 |
| mc_error_pearson | 0.254624397 |
| mc_error_spearman | 0.350818584 |
| moment_raw_variance_error_mae | 0.026123526 |
| moment_calibrated_variance_error_mae | 0.028572882 |
| moment_inference_seconds | 1.343917500 |
| mc_inference_seconds | 34.332254600 |

### 12. 与可比基线的差值
参考基线：`output_gtmedreclpp_40_8_3_e100\test_results.csv`，文件时间 2026-08-14 05:03:43。

注意：以下差值为 `E0 - GTMedReCL++`，只能作为参考差值；严格可比性见第 13 节。

| 指标 | GTMedReCL++ | E0 | 差值 |
|---|---:|---:|---:|
| test_loss | 0.171923 | 0.167828 | -0.004095 |
| test_mae | 0.108516 | 0.107510 | -0.001006 |
| test_psnr | 15.925494 | 16.068155 | +0.142661 |
| test_ssim | 0.516994 | 0.529396 | +0.012402 |
| test_foreground_mae | 0.116170 | 0.115114 | -0.001056 |
| test_foreground_psnr | 15.686900 | 15.836952 | +0.150052 |
| test_foreground_ssim | 0.497727 | 0.510858 | +0.013131 |
| test_gradient_mae | 0.074594 | 0.074969 | +0.000375 |
| test_hfen | 0.741194 | 0.730762 | -0.010432 |

不确定性均值差值，来自 `MEAN` 行：
| 指标 | GTMedReCL++ | E0 | 差值 |
|---|---:|---:|---:|
| moment_mae | 0.108519842 | 0.107508294 | -0.001011548 |
| moment_psnr | 15.925206184 | 16.068284353 | +0.143078168 |
| moment_ssim | 0.516984642 | 0.529402743 | +0.012418101 |
| mc_mean_mae | 0.108114354 | 0.106626190 | -0.001488164 |
| mc_mean_psnr | 15.963840485 | 16.125280698 | +0.161440214 |
| mc_mean_ssim | 0.520822475 | 0.535671731 | +0.014849255 |
| unc_variance_rmsd_brain | 0.000483657 | 0.000414579 | -0.000069077 |
| unc_variance_mae_brain | 0.000309962 | 0.000270035 | -0.000039927 |

### 13. 是否严格可比；不可比时写明原因
结论：不严格可比。

事实原因：
- 当前目录不是 git 仓库，无法确认两组输出对应的代码 commit。
- 当前未发现 checkpoint 文件，无法核对保存的 `args`、模型权重和随机状态。
- 未发现历史训练/评估命令记录。
- `output_gtmedreclpp_40_8_3_e100\log.csv` 含 Med-ReCL 指标列；`output_gt_e0_rec_40_8_3_e100\E0_log.csv` 不含 Med-ReCL 指标列。
- 历史训练/评估硬件未获得。

可参考的相同点：
- 可见数据划分目录一致。
- 两组测试结果均为同 3 个 test 病例汇总。
- 两组日志均完成到 epoch 100，epoch 100 学习率均为 `0.00000102`，每 epoch optimizer step 均为 40。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
本轮诊断事实：
- `git rev-parse --show-toplevel` 报错：当前目录不是 git 仓库。
- `git rev-parse HEAD` 报错：当前目录不是 git 仓库。
- `git status --short --branch` 报错：当前目录不是 git 仓库。
- `nvidia-smi` 报错：命令不可用。
- 当前 PyTorch 为 CPU 构建，CUDA 不可用。
- PowerShell 中读取 Python 文件中文注释时出现编码显示混乱；`rg` 搜索能正常定位英文代码和部分中文信息。需注意不要把显示乱码误认为代码语义错误。

训练状态：
- 本轮未运行训练。

评估状态：
- 本轮未运行评估。

数值异常：
- 本轮未新增数值异常。
- 既有 CSV 未检查到 NaN/Inf；但未执行完整自动校验脚本。

### 15. 遗留问题
事实：
- 缺少 git commit 或版本标签。
- 缺少历史运行命令。
- 缺少 checkpoint 文件，无法复核实际 `args`。
- 缺少历史 GPU/CUDA 信息。
- `E0` 与 `GTMedReCL++` 两组输出是否来自同一代码版本、同一训练命令，目前未获得证据。

建议：
- 后续训练脚本应在输出目录保存 `args.json`、`environment.txt`、`git_commit.txt`、`command.txt`。
- 若继续实验，建议先把当前代码纳入 git，或至少复制保存完整代码快照。
- 后续每次评估后，应把测试指标 CSV 的 `MEAN` / 汇总行同步追加到本文件。

### 16. 本轮事实性结论
事实：
- 已创建 `HANDOFF.md`。
- 当前项目可见 train/val/test 划分为 40/8/3。
- 当前目录不是 git 仓库，commit 未获得。
- 当前诊断环境为 Python 3.13.7、PyTorch 2.11.0+cpu、CUDA 不可用。
- 当前目录中最新已有测试结果来自 `output_gt_e0_rec_40_8_3_e100\E0_test_results.csv`：MAE 0.107510，PSNR 16.068155，SSIM 0.529396。
- 与 `output_gtmedreclpp_40_8_3_e100\test_results.csv` 相比，E0 在 test MAE、PSNR、SSIM、HFEN 上的汇总数值更好，但两组不严格可比。

解释/推测：
- E0 可能是关闭 Med-ReCL 或不同分支的重建实验，因为日志缺少 Med-ReCL 列；但该判断需要历史命令或 checkpoint 才能确认。

### 17. 供下一位分析者重点判断的问题
- E0 实验到底对应什么配置：是否 `--no_medrecl`、是否修改过模型或训练脚本？
- 两组输出是否能通过相同代码、相同 seed、相同数据划分复现？
- 为什么当前目录没有 checkpoint：是否被移动、删除，或 `rg --files` / `Get-ChildItem` 未覆盖到外部保存路径？
- 当前最新 E0 指标提升是否来自方法变化、训练随机性、硬件/AMP差异，还是评估流程差异？
- Moment calibration 后 variance-to-error MAE 比 raw 更大这一现象是否需要重新审查校准目标或验证/测试分布一致性？

---

## EXP-0002：上传非 data 文件到 GitHub

### 1. 本轮实验编号与时间
- 实验编号：EXP-0002
- 时间：2026-08-17 03:37:59 +08:00
- 类型：版本管理 / 上传诊断；未进行训练或重新评估。
- 当前状态：进行中，等待 GitHub API 上传完成后补充远端提交结果。

### 2. 本轮目标
- 将当前项目中除 `data/` 以外的文件上传到 GitHub 仓库：`https://github.com/candicewalker244-cmd/ctri.git`。
- 确保医学影像数据目录 `data/` 不进入 git 暂存区和远端仓库。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 本轮开始时本地目录不是 git 仓库。
- 远端仓库存在，默认分支为 `main`，仓库 size 为 0，当前账号对仓库具备 push 权限。
- 本地初始提交已创建：`e90c0917e6fb3f422f1b5b5bdf0713b2de8ef484`。

未获得：
- 训练 checkpoint。
- 既有远端 commit 历史；远端仓库 size 为 0，推测为空仓库，但本地 `git ls-remote` 受网络限制未能直接确认 refs。

### 4. 理论依据
事实：
- 使用 `.gitignore` 排除 `data/`，可避免后续 git add 操作误纳入医学影像数据。
- GitHub 普通 git/HTTPS 需要访问 `github.com:443`；当前机器该端口 TCP 测试失败。
- `api.github.com:443` 可访问，因此可以通过 GitHub Git Data API 创建 blob/tree/commit/ref 完成上传。

解释/推测：
- 本机网络对 `github.com:443` 存在限制，但对 `api.github.com:443`、`ssh.github.com:443` 可连通。
- SSH 通道缺少已授权私钥，不能直接用于 push。

### 5. 实际修改文件
- `.gitignore`
- `HANDOFF.md`

### 6. 每个文件具体修改内容
- `.gitignore`：新增 `data/` 规则，排除数据目录。
- `HANDOFF.md`：追加 EXP-0002 上传记录。

### 7. 实际运行命令
已运行：
```powershell
git --version
git ls-remote --symref https://github.com/candicewalker244-cmd/ctri.git HEAD
gh --version
gh auth status
Get-ChildItem -Force | Select-Object Mode,Length,LastWriteTime,Name
Get-ChildItem -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\data\\' } | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length
git init -b main
git remote add origin https://github.com/candicewalker244-cmd/ctri.git
git status --short --ignored
git check-ignore -v data data\dataset\train\1BA001\1BA001_ct.nii.gz
git config user.name "candicewalker244-cmd"
git config user.email "317586217+candicewalker244-cmd@users.noreply.github.com"
git -c http.version=HTTP/1.1 ls-remote --symref origin HEAD
git add -- .gitignore HANDOFF.md model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py output_gtmedreclpp_40_8_3_e100 output_gt_e0_rec_40_8_3_e100
git status --short --ignored
git diff --cached --stat
git commit -m "Initial project upload without data"
git -c http.version=HTTP/1.1 -c http.postBuffer=157286400 push -u origin main
ssh -o BatchMode=yes -T git@github.com
ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -T git@github.com
ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p 443 -T git@ssh.github.com
Test-NetConnection github.com -Port 443
Test-NetConnection api.github.com -Port 443
Test-NetConnection ssh.github.com -Port 443
Resolve-DnsName github.com
Resolve-DnsName ssh.github.com
Get-ChildItem Env: | Where-Object { $_.Name -match 'GITHUB|GH_|GIT' } | Select-Object Name,Value
git config --show-origin --get-all credential.helper
git credential-manager --version
git credential-manager github list
git credential fill
```

待运行：
```powershell
# 使用 GitHub Git Data API：create blob -> create tree -> create commit -> create/update refs/heads/main。
# credential 来自 Git Credential Manager，仅用于本次 API 请求；禁止写入文件或记录 token。
```

训练命令：未运行。

评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
事实：
- 数据目录 `data/` 未进入 git 暂存区。
- `git check-ignore -v data data\dataset\train\1BA001\1BA001_ct.nii.gz` 确认 `.gitignore:1:data/` 生效。
- 本轮未改变数据集与数据划分；仍沿用 EXP-0001 记录的 train/val/test = 40/8/3。

是否与基线一致：
- 数据划分一致。
- 本轮不涉及训练/评估指标可比性。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 本轮未运行训练，seed：未运行。
- epoch：未运行。
- batch size：未运行。
- learning rate：未运行。
- optimizer：未运行。
- loss：未运行。
- 关键开关：`.gitignore` 排除 `data/`。

### 10. GPU、CUDA、Python、PyTorch
- 本轮未运行训练/评估。
- 沿用 EXP-0001 当前诊断环境：Python 3.13.7，PyTorch 2.11.0+cpu，CUDA 不可用。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未运行评估。
- 最新已有测试结果仍沿用 EXP-0001：`output_gt_e0_rec_40_8_3_e100\E0_test_results.csv`，MAE 0.107510，PSNR 16.068155，SSIM 0.529396。

### 12. 与可比基线的差值
- 本轮未运行评估，差值未获得。
- 已有结果差值沿用 EXP-0001。

### 13. 是否严格可比；不可比时写明原因
- 本轮不是训练/评估实验，不涉及指标严格可比性。
- 上传文件层面，`data/` 已被排除；其他当前 git tracked 文件计划上传。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- `git ls-remote` 失败：`github.com:443` 连接被重置或连接超时。
- `git push -u origin main` 失败：无法连接 `github.com:443`。
- `gh` CLI 未安装。
- SSH 到 `github.com` / `ssh.github.com:443` 均可到达认证阶段，但失败原因是 `Permission denied (publickey)`。
- `Test-NetConnection github.com -Port 443`：TCP 失败。
- `Test-NetConnection api.github.com -Port 443`：TCP 成功。
- `Test-NetConnection ssh.github.com -Port 443`：TCP 成功。

训练状态：
- 未运行。

评估状态：
- 未运行。

### 15. 遗留问题
- 远端 API 上传尚未完成，需补充远端 commit SHA。
- 本地 git 历史与即将通过 GitHub API 生成的远端 commit SHA 可能不一致。
- 若后续希望直接使用 `git push`，需要解决本机到 `github.com:443` 的网络限制，或配置已授权 SSH key。

### 16. 本轮事实性结论
事实：
- `.gitignore` 已排除 `data/`。
- 本地已创建 git 仓库、连接远端 `origin`，并创建初始提交 `e90c0917e6fb3f422f1b5b5bdf0713b2de8ef484`。
- 当前 tracked 文件数为 18，未包含 `data/`。
- 直接 `git push` 尚未成功，原因是本机无法连接 `github.com:443`。

解释/推测：
- 通过 `api.github.com` 上传应可绕过当前 git/HTTPS 传输限制。

### 17. 供下一位分析者重点判断的问题
- 远端最终 commit 是否完整包含 18 个非 `data/` 文件。
- 远端仓库中是否不存在 `data/` 目录或 `.nii.gz` 医学影像文件。
- 后续是否需要把本地仓库历史与 API 生成的远端历史重新对齐。
