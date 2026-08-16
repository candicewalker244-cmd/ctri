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
- 当前状态：已完成。

### 2. 本轮目标
- 将当前项目中除 `data/` 以外的文件上传到 GitHub 仓库：`https://github.com/candicewalker244-cmd/ctri.git`。
- 确保医学影像数据目录 `data/` 不进入 git 暂存区和远端仓库。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 本轮开始时本地目录不是 git 仓库。
- 远端仓库存在，默认分支为 `main`，仓库 size 为 0，当前账号对仓库具备 push 权限。
- 本地初始提交已创建：`e90c0917e6fb3f422f1b5b5bdf0713b2de8ef484`。
- 本地包含 EXP-0002 记录的提交：`0558851398547ffee7262ee452286e6c42113ee4`。
- 远端初始化提交：`f7ac7cade683ba349c494caeb509adf2016ffd8e`。
- 远端完整上传提交：`8b23967dd83434bea4448d22e13b3680b248b57d`。
- 远端 HANDOFF 首次结果同步提交：`347b737fa8bad0f801258c2e57a97bb956a1b20e`。

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
Invoke-RestMethod -Method Put -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/contents/.gitignore"
Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/git/blobs"
Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/git/trees"
Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/git/commits"
Invoke-RestMethod -Method Patch -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/git/refs/heads/main"
Invoke-RestMethod -Method Put -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/contents/HANDOFF.md"
Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/candicewalker244-cmd/ctri/git/trees/<tree_sha>?recursive=1"
```

已运行：
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
- 上传文件层面，`data/` 已被排除；其他当前 git tracked 文件已上传。

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
- 远端完整上传已完成：`8b23967dd83434bea4448d22e13b3680b248b57d`。
- 本地 git 历史与通过 GitHub API 生成的远端 commit SHA 不一致。
- 若后续希望直接使用 `git push`，需要解决本机到 `github.com:443` 的网络限制，或配置已授权 SSH key。

### 16. 本轮事实性结论
事实：
- `.gitignore` 已排除 `data/`。
- 本地已创建 git 仓库、连接远端 `origin`，并创建初始提交 `e90c0917e6fb3f422f1b5b5bdf0713b2de8ef484`。
- 当前 tracked 文件数为 18，未包含 `data/`。
- 直接 `git push` 尚未成功，原因是本机无法连接 `github.com:443`。
- 已通过 GitHub Git Data API 将 18 个 tracked 文件上传到远端 `main`。
- 远端完整上传 commit：`8b23967dd83434bea4448d22e13b3680b248b57d`。
- 远端校验时 HEAD：`347b737fa8bad0f801258c2e57a97bb956a1b20e`。
- 远端校验 blob 数：18。
- 远端校验 `data/` 或 `.nii.gz` 文件数：0。

解释/推测：
- 通过 `api.github.com` 上传已绕过当前 git/HTTPS 传输限制。
- 本地 git 历史与远端 API 历史不同；后续若要继续从本机常规 push，建议先解决网络/SSH 后重新 clone 或对齐历史。

### 17. 供下一位分析者重点判断的问题
- 远端最终 commit 是否完整包含 18 个非 `data/` 文件。
- 远端仓库中是否不存在 `data/` 目录或 `.nii.gz` 医学影像文件。
- 后续是否需要把本地仓库历史与 API 生成的远端历史重新对齐。

---

## EXP-0003：新文件实现 RA-MedReCL++（不改旧训练/模型代码）

### 1. 本轮实验编号与时间
- 实验编号：EXP-0003
- 时间：2026-08-17 05:05:48 +08:00
- 类型：代码实现 / 轻量诊断；未运行正式训练或正式评估。

### 2. 本轮目标
- 按用户要求，不在旧代码文件上直接修改，而是基于旧代码复制出新版本文件。
- 在新版本中一次性接入对话方案中的 RA/RG-MedReCL++：重建导向 latent 对齐、多尺度解剖/图像一致性、error/hard-region mining、频域一致性、梯度一致性、自适应 hard-region 权重和多尺度医学一致性。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 修改前本地 HEAD：`ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 旧文件 `model_attnres3d_gtmedreclpp.py` 和 `train_gtmedreclpp.py` 在本轮结束时仍无 git diff。
- 本轮基于旧文件复制出新文件后，只在新文件上实现新方法。

### 4. 理论依据
事实：
- 用户要求 MedReCL 必须改善最终 CT→MRI 重建任务，而不是只完成独立 feature contrast。
- 新实现把辅助目标从“CT/MRI 跨模态表征对齐”收敛为“生成 MRI / 预测 MRI 与真实 MRI 的重建导向医学表征一致性”。

解释/推测：
- 预测 MRI latent 对齐真实 MRI latent，比 CT feature 直接对齐 MRI feature 更贴合 paired voxel-wise reconstruction。
- error 与解剖梯度共同构造 hard-region 权重，可让辅助约束集中在边界、小结构和当前重建误差大的区域。
- 频域与梯度一致性直接对应 HFEN、Gradient MAE 和视觉边缘质量，理论上比单纯 feature similarity 更可能服务定量/定性重建。

### 5. 实际修改文件
- 新增：`model_attnres3d_ramedreclpp.py`
- 新增：`train_ramedreclpp.py`
- 修改：`.gitignore`
- 修改：`HANDOFF.md`

### 6. 每个文件具体修改内容
- `model_attnres3d_ramedreclpp.py`
  - 从旧 `model_attnres3d_gtmedreclpp.py` 复制得到，不改旧文件。
  - 将 `MedReCLConfig` 文档和参数升级为 RA-MedReCL++。
  - 新增结构项权重：`structure_contrast_weight`。
  - 新增重建导向项：`latent_alignment_weight`、`image_multiscale_weight`、`gradient_consistency_weight`、`frequency_consistency_weight`。
  - 新增 hard-region 自适应权重参数：`uncertainty_blend`、`uncertainty_boost`、`adaptive_weight_max`、`hard_region_quantile`。
  - 新增 hard-region 对 InfoNCE 温度、anchor 权重、negative alpha 的调节参数：`kappa_u`、`beta_u`、`rho_u`。
  - 将 `lambda_R` 默认从 0.0 改为 0.5，使 error-guided hard negative 默认参与。
  - 在 `forward_components()` 中新增 `pred_mri_feats = model.extract_target_features(pred)`，并通过 `project_medrecl_target_features()` 形成 prediction MRI latent。
  - 新增 `_adaptive_reconstruction_weight()`，用当前 error map 与 target gradient map 构造 stop-gradient hard-region 权重。
  - 新增 `_prediction_latent_alignment_loss()`，对齐 prediction MRI latent 与 EMA true MRI latent。
  - 新增 `_multiscale_image_consistency_loss()`，进行 hard-region 加权多尺度强度和局部统计一致性约束。
  - 新增 `_gradient_consistency_loss()`，进行 hard-region 加权 3D 梯度幅值一致性约束。
  - 新增 `_frequency_consistency_loss()`，进行 focal-style 3D 频域一致性约束。
  - 将 hard-region 权重接入 `_build_level_context()` 与 `_single_case_level_loss()`，影响 anchor 采样、温度、omega、hard negative 和 alpha。
  - 新增训练返回指标：`medrecl_latent_alignment_loss`、`medrecl_image_multiscale_loss`、`medrecl_gradient_consistency_loss`、`medrecl_frequency_consistency_loss`、`medrecl_adaptive_weight_mean`、`medrecl_hard_region_fraction`。
- `train_ramedreclpp.py`
  - 从旧 `train_gtmedreclpp.py` 复制得到，不改旧文件。
  - import 改为 `import model_attnres3d_ramedreclpp as net`。
  - 方法名打印改为 RA-MedReCL++。
  - 新增 CLI 参数：`--medrecl_structure_contrast_weight`、`--medrecl_latent_alignment_weight`、`--medrecl_image_multiscale_weight`、`--medrecl_gradient_consistency_weight`、`--medrecl_frequency_consistency_weight`、`--medrecl_uncertainty_blend`、`--medrecl_uncertainty_boost`、`--medrecl_adaptive_weight_max`、`--medrecl_hard_region_quantile`、`--medrecl_error_negative_weight`、`--medrecl_beta_u`、`--medrecl_rho_u`、`--medrecl_kappa_u`。
  - 将上述参数传入 `MedReCLConfig`。
  - 新增 CSV 训练日志列和控制台输出：Lat / MSImg / GradC / FreqC / Hard 等。
- `.gitignore`
  - 新增 `__pycache__/`，避免 py_compile 生成物进入版本控制。
- `HANDOFF.md`
  - 追加本轮 EXP-0003 记录。

### 7. 实际运行命令
已运行：
```powershell
git status --short --ignored
git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py
Copy-Item -LiteralPath model_attnres3d_gtmedreclpp.py -Destination model_attnres3d_ramedreclpp.py
Copy-Item -LiteralPath train_gtmedreclpp.py -Destination train_ramedreclpp.py
python -m py_compile model_attnres3d_ramedreclpp.py train_ramedreclpp.py
python train_ramedreclpp.py --help
```

轻量 smoke 命令：
```powershell
@'
import torch
import model_attnres3d_ramedreclpp as net

torch.manual_seed(7)
model = net.AttnResCTtoMRI(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    bottleneck_blocks=1,
    dropout=0.1,
    use_medrecl=True,
    medrecl_proj_dim=8,
)
criterion = net.ReconstructionLoss(frequency_weight=0.02)
med = net.MedReCLLoss(net.MedReCLConfig(
    proj_dim=8,
    lambda_structure_max=0.01,
    lambda_appearance_max=0.004,
    anchor_samples=16,
    normal_negative_samples=8,
    hard_negative_samples=4,
    easy_background_samples=4,
    candidate_pool_size=64,
    appearance_context_samples=8,
    invariance_samples=8,
    gradient_balance_interval=999,
))
source = torch.rand(1, 1, 16, 16, 16)
target = torch.rand(1, 1, 16, 16, 16)
model.train()
pred, features = model.forward_with_features(source)
rec_loss = criterion(pred, target)
weighted, metrics, raw = med.weighted_loss(
    model=model,
    source=source,
    target=target,
    pred=pred,
    feature_dict=features,
    rec_loss=rec_loss,
    current_step=90,
    total_steps=100,
)
loss = rec_loss + weighted
loss.backward()
print('pred_shape', tuple(pred.shape))
print('rec_loss', float(rec_loss.detach()))
print('raw_medrecl', float(raw.detach()))
print('weighted_medrecl', float(weighted.detach()))
for key in ['latent_alignment_loss','image_multiscale_loss','gradient_consistency_loss','frequency_consistency_loss','adaptive_weight_mean','hard_region_fraction']:
    print(key, metrics[key])
'@ | python -
```

训练命令：未运行。

评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 本轮未改动 `data/`。
- 本轮未运行训练/评估，因此未实际使用完整数据集。
- 数据划分仍沿用 EXP-0001：train/val/test = 40/8/3。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
正式训练：
- seed：未运行。
- epoch：未运行。
- batch size：未运行。
- learning rate：未运行。
- optimizer：未运行。

轻量 smoke：
- seed：`torch.manual_seed(7)`。
- batch size：1。
- 输入尺寸：`16x16x16` 随机张量。
- 模型：`base_channels=4`、`bottleneck_blocks=1`、`dropout=0.1`、`medrecl_proj_dim=8`。
- ReconstructionLoss：`frequency_weight=0.02`。
- MedReCLConfig smoke：`anchor_samples=16`、`candidate_pool_size=64`、`appearance_context_samples=8`、`invariance_samples=8`、`gradient_balance_interval=999`。
- optimizer：未使用，仅验证 `loss.backward()`。
- 关键开关：新入口默认启用 RA-MedReCL++，旧入口不受影响。

### 10. GPU、CUDA、Python、PyTorch
当前诊断环境：
- Python：3.13.7
- PyTorch：2.11.0+cpu
- CUDA available：False
- torch CUDA version：None

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未运行正式训练或评估。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN / Gradient MAE：未获得。
- 最新真实测试结果仍沿用 EXP-0001 中已有 E0 结果：MAE 0.107510，PSNR 16.068155，SSIM 0.529396。

轻量 smoke 输出事实：
- `pred_shape (1, 1, 16, 16, 16)`
- `rec_loss 0.27169516682624817`
- `raw_medrecl 11.600701332092285`
- `weighted_medrecl 0.000997519469819963`
- `latent_alignment_loss 0.32534298300743103`
- `image_multiscale_loss 0.23910541832447052`
- `gradient_consistency_loss 0.47299808263778687`
- `frequency_consistency_loss 0.008381301537156105`
- `adaptive_weight_mean 1.0`
- `hard_region_fraction 0.2001953125`

### 12. 与可比基线的差值
- 本轮未运行正式评估，差值未获得。
- 不能用 smoke 随机张量结果与 E0/GTMedReCL++ 做任何性能比较。

### 13. 是否严格可比；不可比时写明原因
- 本轮代码 smoke 不可与历史实验严格可比。
- 原因：未使用真实数据、未训练、未评估 test split、未加载历史 checkpoint。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- 第一次 smoke 命令使用 Bash 风格 `python - <<'PY'`，在 PowerShell 中失败；随后改用 PowerShell here-string 成功。
- `python -m py_compile model_attnres3d_ramedreclpp.py train_ramedreclpp.py` 成功。
- `python train_ramedreclpp.py --help` 成功，退出码 0。
- 轻量 smoke 前向、loss 计算、`loss.backward()` 成功。
- 删除 `__pycache__/` 的 PowerShell 删除命令被环境策略拦截；已改为在 `.gitignore` 中忽略 `__pycache__/`。

数值异常：
- smoke 未发现 NaN/Inf。
- 未运行正式训练，无法判断长程训练稳定性。

### 15. 遗留问题
- 需要在真实数据上先跑 1 epoch 或少量 batch，观察显存、速度、loss 数值量级和是否出现梯度异常。
- 需要与 E0 reconstruction-only 基线严格同划分、同 seed、同 epoch、同评估脚本对比。
- 由于 RA-MedReCL++ 新增 prediction encoder pass 和频域/梯度项，训练显存与耗时会增加，需实测。
- 本轮未上传 GitHub；若需要同步远端，应注意当前本机直接 `git push` 仍可能受 `github.com:443` 网络问题影响。

### 16. 本轮事实性结论
事实：
- 已按用户要求在新文件上实现，没有改旧 `model_attnres3d_gtmedreclpp.py` 与 `train_gtmedreclpp.py`。
- 新增 `model_attnres3d_ramedreclpp.py` 和 `train_ramedreclpp.py`。
- 新训练入口能解析 RA-MedReCL++ 参数。
- 新模型文件通过 py_compile 和随机张量 backward smoke。

解释/推测：
- 该实现更贴合“MedReCL 服务 reconstruction”的目标，因为新增项直接约束 `pred MRI` 与 `GT MRI` 的 latent、图像、多尺度局部统计、梯度和频域一致性。
- 是否提升 MAE/PSNR/SSIM/HFEN 必须等待真实训练与测试，当前禁止估算。

### 17. 供下一位分析者重点判断的问题
- RA-MedReCL++ 新默认权重是否过强，是否需要先小权重 warmup 或提高 `recon_only_ratio`？
- `lambda_R=0.5` 的 error-guided hard negative 是否稳定，是否需要与 `hard_region_quantile` 联合消融？
- prediction MRI latent 经过 target encoder 是否会带来额外显存瓶颈？
- 频域一致性和原 ReconstructionLoss 的 frequency 项是否存在重复，需要真实训练后观察梯度和指标。
- 首个真实实验建议命令应优先小规模 smoke：`--epochs 1 --save_dir ./output_ra_smoke --max_test_figures 1 --no_eval_mc_dropout_compare`。

---

## EXP-0004：RA-MedReCL++ 二次代码审查与真实数据轻量诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0004
- 时间：2026-08-17 05:14:49 +08:00
- 类型：代码审查 / 诊断；未修改模型与训练逻辑，未运行正式训练或正式评估。

### 2. 本轮目标
- 复查 EXP-0003 新增 RA-MedReCL++ 是否存在代码错误、无效梯度、尺寸问题、数值异常或与方案不一致之处。
- 严格保留旧版 `model_attnres3d_gtmedreclpp.py` 与 `train_gtmedreclpp.py`，不在旧版上修改。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 本地 HEAD 仍为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 审查对象为未提交的新文件 `model_attnres3d_ramedreclpp.py` 与 `train_ramedreclpp.py`。
- `git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py` 无输出，旧版文件仍未改动。

### 4. 理论依据
事实：
- 对话方案要求 reconstruction-guided latent alignment、multi-scale consistency、error-guided hard mining、frequency/gradient consistency 以及 uncertainty-guided adaptive weighting。
- 代码审查必须区分“能运行”与“严格实现方案并能产生独立训练信号”。

解释/推测：
- 如果所谓 uncertainty map 只由重建误差和 GT 梯度构成，则它是 hard-region score，不是 Moment predictive variance。
- 如果辅助频域项与主重建 FFL 完全相同，则它只是在另一调度权重下重复同一个目标，没有引入新的频域信息。

### 5. 实际修改文件
- 仅修改：`HANDOFF.md`（追加本轮诊断记录）。
- 模型代码：未修改。
- 训练代码：未修改。

### 6. 每个文件具体修改内容
- `HANDOFF.md`
  - 追加 EXP-0004 的审查命令、真实 smoke 事实、发现的问题、风险和后续判断项。

### 7. 实际运行命令
已运行：
```powershell
git status --short
git diff --stat
git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py
git diff --no-index -- model_attnres3d_gtmedreclpp.py model_attnres3d_ramedreclpp.py
git diff --no-index -- train_gtmedreclpp.py train_ramedreclpp.py
rg -n "...关键类、函数、配置与日志字段..." model_attnres3d_ramedreclpp.py train_ramedreclpp.py
ruff check model_attnres3d_ramedreclpp.py --output-format concise
ruff check train_ramedreclpp.py --output-format concise
ruff check model_attnres3d_gtmedreclpp.py --output-format concise
```

多尺寸、多 batch 和单项梯度 smoke：
```powershell
@'
# seed=23；测试 batch=2/shape=16、batch=1/shape=17、调度边界、
# multiscale/gradient/frequency 单项梯度以及常量体数据。
'@ | python -
```

真实 NIfTI 小块 smoke：
```powershell
@'
# seed=31；从 data/dataset/train 读取真实病例，随机裁剪 16x16x16，
# 使用 base_channels=4、bottleneck_blocks=1 完成 forward/loss/backward。
'@ | python -
```

默认 `96x96x96` 裁剪前景比例诊断：
```powershell
@'
# seed=42；遍历 40 个训练病例各一个随机 96^3 patch，统计 target > 0.02 比例。
'@ | python -
```

频域损失等价性检查：
```powershell
@'
# seed=5；比较 ReconstructionLoss._focal_frequency_loss 与
# MedReCLLoss._frequency_consistency_loss。
'@ | python -
```

训练命令：未运行。

评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 真实轻量诊断读取 `data/dataset/train`，发现 40 个训练病例。
- 默认 `96^3` 裁剪前景比例诊断使用全部 40 个训练病例，每例抽取一个 patch。
- 未使用 val/test 计算模型指标。
- 数据划分未改变，仍为 train/val/test = 40/8/3，与既有基线划分一致；但本轮不是正式可比实验。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
正式训练：
- seed：未运行。
- epoch：未运行。
- batch size：未运行。
- learning rate：未运行。
- optimizer：未运行。

诊断 smoke：
- 随机张量 seed：23；batch size 2 和 1；尺寸 `16^3`、`17^3`。
- 真实 NIfTI smoke seed：31；batch size 1；尺寸 `16^3`。
- 默认 patch 前景统计 seed：42；patch size `96^3`。
- 频域等价性 seed：5。
- loss：`ReconstructionLoss` + `MedReCLLoss`；真实 smoke 在 `current_step=90/100` 激活结构和外观分支。
- optimizer：未使用；执行了 `loss.backward()`。

### 10. GPU、CUDA、Python、PyTorch
- Python：3.13.7
- PyTorch：2.11.0+cpu
- CUDA available：False
- GPU/CUDA 显存与 AMP：未获得，当前环境无法验证。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未运行正式训练或 test 评估。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN / Gradient MAE：未获得。
- 最新真实测试结果仍为 EXP-0001 的 E0：MAE 0.107510，PSNR 16.068155，SSIM 0.529396。

真实训练 patch 轻量 smoke（不是测试指标）：
- `rec_loss = 0.5629761219024658`
- `raw_medrecl = 9.064045906066895`
- `weighted_medrecl = 0.0015969834057614207`
- `structure_gradient_ratio = 6.8334330868927`
- `structure_gradient_scale = 0.011707146171292595`
- 所有已产生参数梯度均为有限值。

默认 `96^3` patch 前景统计：
- 40 个 patch 的 `target > 0.02` 前景比例最小值 0.252527、median 0.697283、mean 0.677276、最大值 0.971408。
- 近空白 patch（前景小于 1% 或 5%）：0/40。

### 12. 与可比基线的差值
- 未运行正式评估，PSNR/SSIM/MAE 差值未获得。
- smoke loss 不可与历史 E0 或 GT-MedReCL++ 指标比较。

### 13. 是否严格可比；不可比时写明原因
- 不严格可比。
- 原因：未训练、未加载同一 checkpoint、未在 test split 上评估；模型 smoke 使用缩小网络和小 patch。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- 首次真实数据 smoke 错误使用 `data/train`，报 `FileNotFoundError`；确认项目实际路径为 `data/dataset/train` 后重跑成功。
- batch=2、`16^3` 与 batch=1、`17^3` 的完整 RA loss/backward 均成功，无 NaN/Inf。
- multiscale、gradient、frequency 三个新增直接项均产生有限且非零的预测梯度。
- 常量体数据的 adaptive map 为有限值且均值 1.0。
- `train_ramedreclpp.py` 通过 Ruff。
- 新旧 model 文件均有相同的 9 个 Ruff 告警（3 个未使用 import、2 个未使用局部变量、4 个歧义变量名），属于继承的旧代码问题，不是本轮 RA 新增逻辑引入。

已确认问题：
1. 严格方案缺口：`_adaptive_reconstruction_weight()` 只接收 `target_aux`、`grad_map`、`error_map`，没有接入 Moment variance 或其他预测不确定性。因此当前是 error/edge-guided，而不是真正 uncertainty-guided。
2. 重复目标：默认 RA `_frequency_consistency_loss()` 与 `ReconstructionLoss._focal_frequency_loss()` 数学实现相同。seed=5 随机张量实测二者均为 `0.07139548659324646`，绝对差 `0.0`。
3. 日志错误：`hard_region_fraction` 使用 `adaptive_map >= quantile`。近空白真实 patch 因大量值并列，日志为 `1.0`，而按训练 hard mask 同样的严格 `>` 规则实际比例为 `0.003173828125`。该问题会误导实验诊断，但训练 hard mask 本身使用 `>`。
4. 有效权重风险：真实小 patch 上结构分支 raw gradient ratio 为 6.833433，梯度平衡将其 scale 压到 0.011707，最终 weighted MedReCL 仅 0.001597，而 rec loss 为 0.562976。该行为符合现有 cap 逻辑，但新增项可能因此贡献过弱；是否影响指标必须训练验证。
5. 资源风险：结构激活后每批至少增加 prediction MRI online encoder、GT MRI online encoder、GT MRI EMA encoder，并在 invariance 开启时再增加增强 MRI online encoder。当前 CPU 小模型通过，但默认 `96^3`、base_channels=32 的 GPU 显存/速度未验证。

### 15. 遗留问题
- 需要决定是否将真实 Moment variance 以 stop-gradient 形式接入 hard-region weighting，或明确将方法改名为 error/edge-guided，避免概念不一致。
- 需要把 RA 频域项改成与基础 FFL 不重复的约束（例如分频带、局部频域或频率-空间耦合），或者移除重复项并只保留主重建 FFL。
- 需要修正 `hard_region_fraction` 的并列分位数统计，使日志反映训练实际 hard mask。
- 需要确认 gradient cap 是否让新模块过弱，建议先记录每个新增项对共享解码器的独立梯度比例。
- 需要在 CUDA 环境执行默认 `96^3` 配置的一批显存/耗时 smoke，再决定是否保留四次 MRI encoder 前向。

### 16. 本轮事实性结论
事实：
- 新代码具备基本可运行性：多 batch、奇数尺寸、常量输入、真实 NIfTI 小块均可完成前向和反向，未发现 NaN/Inf。
- 旧版模型与训练文件没有被修改。
- 当前实现没有真正使用 Moment uncertainty。
- 当前新增频域项与主重建 FFL 完全重复。
- 当前 `hard_region_fraction` 在分位数并列时可能严重虚报。

解释/推测：
- 以上三个设计/日志问题不一定导致训练崩溃，但会削弱“严格按方案实现”的可信度，并使后续实验解释不可靠。
- 在正式长训练前修正这些点，比直接跑 100 epoch 更稳妥。

下一步建议：
- 只在新文件 `model_attnres3d_ramedreclpp.py`、`train_ramedreclpp.py` 基础上继续建立下一版，不回改旧 GT 文件。
- 优先修复真实 uncertainty 接入、非重复频域约束和 hard fraction 日志，再做 1 epoch CUDA smoke。

### 17. 供下一位分析者重点判断的问题
- 训练期 Moment variance 的计算成本是否可接受；若不可接受，应选择低成本 uncertainty proxy 并准确命名。
- 新频域目标应采用哪种非重复形式，才能直接对应 HFEN/Gradient MAE 改善。
- 是否要将 prediction latent alignment 与原结构对比拆开独立做 gradient cap，避免高 raw contrast 梯度把所有新重建项一起压低。
- 默认四次 MRI encoder 前向在目标 GPU 上的峰值显存和单 batch 时间是多少。
