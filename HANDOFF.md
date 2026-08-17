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

---

## EXP-0005：按纠正后方案完善 RA-MedReCL++ 训练实现

### 1. 本轮实验编号与时间
- 实验编号：EXP-0005
- 时间：2026-08-17 05:48:22 +08:00
- 类型：代码修改 / 训练链路诊断；未运行正式训练或正式评估。

### 2. 本轮目标
- 严格按对话中纠正后的方案完善 RA-MedReCL++。
- 明确 Moment propagation 仍只用于训练后不确定性推理/评估，不作为 RA 训练权重。
- 训练期 hard-region 仅使用可监督的 reconstruction error 与 target gradient。
- 修复 EXP-0004 发现的重复频域目标、hard 日志虚报和新重建引导项被旧结构对比统一压缩的问题。
- 所有模型/训练修改只发生在新 RA 文件，不修改旧 GT-MedReCL++ 文件。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 本地 HEAD：`ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 修改基于 EXP-0003/EXP-0004 的 `model_attnres3d_ramedreclpp.py` 与 `train_ramedreclpp.py`。
- 本轮结束时 `git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py` 仍无输出。

### 4. 理论依据
事实：
- Moment 的 `forward_mu_var_cov()` 是训练后推理路径；本轮没有将其接入训练 loss。
- paired CT->MRI 的可监督 hard-region 信号可直接来自当前 prediction/GT error 和 GT anatomy gradient。
- 原 ReconstructionLoss 已含 focal complex-spectrum loss，RA 辅助频域项不应再次复制相同公式。
- 旧结构 contrast 梯度远大于新的 output-guided 梯度时，对二者统一缩放会把真正服务 reconstruction 的新增项一起压弱。

解释/推测：
- 前景内精确 top-k hard mining 比全体素 quantile 更能避免背景并列值造成阈值和日志失真。
- 窗口化、径向多频带等权 log-amplitude 与 target-supported phase consistency 提供了不同于原 focal frequency loss 的频谱监督。
- 分开约束 structure contrast、reconstruction guidance、appearance 三条梯度，可保留安全上限，同时避免旧对比项支配新重建项的缩放。

### 5. 实际修改文件
- 修改：`model_attnres3d_ramedreclpp.py`
- 修改：`train_ramedreclpp.py`
- 修改：`HANDOFF.md`
- 未修改：`model_attnres3d_gtmedreclpp.py`
- 未修改：`train_gtmedreclpp.py`

### 6. 每个文件具体修改内容
- `model_attnres3d_ramedreclpp.py`
  - 将训练 hard score 参数从误导性的 uncertainty 命名改为 `hard_error_blend`、`hard_region_boost`、`kappa_h`、`beta_h`、`rho_h`。
  - `_adaptive_reconstruction_weight()` 现在同时返回连续 adaptive weight 与前景内精确 top-k hard mask。
  - hard mask 按每个样本独立选择；`hard_region_quantile=0.80` 表示选择约 20% 前景体素，不受相同分数并列影响。
  - `_build_level_context()` 使用 nearest interpolation 传递离散 hard mask。
  - contrast hard negative 直接使用该 hard mask，不再从归一化 adaptive map 二次估计 quantile。
  - `hard_region_fraction` 改为真实全体素 hard mask 比例；新增 `hard_region_foreground_fraction` 记录前景内比例。
  - 将重复的 focal complex-spectrum 辅助项替换为窗口化多频带频域一致性：3D Hann window、去均值、低/中/高径向频带等权 log-amplitude loss，加 target-amplitude 支持的 phase loss。
  - 新增 `frequency_phase_weight`。
  - 将 loss 拆分为 `structure_loss`、`guidance_loss`、`appearance_loss`。prediction latent、多尺度图像、gradient、frequency 归入 reconstruction guidance。
  - 新增独立 `guidance_cap_ratio`、`guidance_gradient_scale` 与 `guidance_gradient_ratio`。
  - 总辅助梯度仍受 `contrast_max_ratio` 约束。
  - 删除一次没有被任何 loss 使用的 GT online target encoder/projector 前向；保留 prediction online encoder、GT EMA teacher 和 invariance augmentation 路径。
- `train_ramedreclpp.py`
  - 新增规范参数：`--medrecl_hard_error_blend`、`--medrecl_hard_region_boost`、`--medrecl_beta_h`、`--medrecl_rho_h`、`--medrecl_kappa_h`。
  - 保留旧 `uncertainty_*`、`*_u` 参数名作为命令行兼容别名，但内部不再解释为 Moment uncertainty。
  - 新增 `--medrecl_frequency_phase_weight` 与 `--medrecl_guidance_cap_ratio`。
  - 增加所有新增权重、hardness 系数和 gradient cap 的非负校验。
  - 启动信息明确打印：训练期 hard guidance = supervised error + target gradient；Moment variance 仅训练后使用。
  - CSV 新增 structure/guidance loss、guidance gradient scale/ratio、hard 前景比例。
  - 控制台新增 `Str`、`Guide`、`GuideG`、`HardAll/FG`。
- `HANDOFF.md`
  - 追加 EXP-0005 实现和验证记录。

### 7. 实际运行命令
已运行：
```powershell
python -m py_compile model_attnres3d_ramedreclpp.py train_ramedreclpp.py
python train_ramedreclpp.py --help
ruff check train_ramedreclpp.py --output-format concise
ruff check model_attnres3d_ramedreclpp.py --output-format concise
git diff --check
git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py
```

已运行随机张量完整 RA smoke：
```powershell
@'
# seed=41；batch=2，shape=16^3；完整 forward、三路 loss、梯度平衡和 backward；
# 同时比较原 FFL 与新 RA frequency，并检查 100 前景体素 top-k。
'@ | python -
```

已运行真实 NIfTI smoke：
```powershell
@'
# seed=31；读取 data/dataset/train，真实 16^3 patch；
# base_channels=4、bottleneck_blocks=1，完整 forward/loss/backward。
'@ | python -
```

已运行完整训练循环 smoke：
```powershell
@'
# seed=53；2 个随机 batch，调用 train_one_epoch，执行 2 次 AdamW optimizer.step()，
# 检查全部新增聚合指标。
'@ | python -
```

已运行 recon-only warmup 分支 smoke：
```powershell
@'
# current_step=1/100，调用 train_one_epoch，确认 RA 未激活时新增日志键完整。
'@ | python -
```

已运行默认尺寸频域 backward：
```powershell
@'
# seed=61；单独对 96^3 prediction/target 执行新 frequency loss 与 backward。
'@ | python -
```

正式训练命令：未运行。

正式评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 真实 smoke 使用 `data/dataset/train` 中一个训练病例的随机 `16^3` patch。
- 未改动数据文件与数据划分。
- train/val/test 仍为 40/8/3，与既有基线划分一致。
- 未使用 val/test 产生本轮模型指标。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
正式实验：
- seed：未运行。
- epoch：未运行。
- batch size：未运行。
- learning rate：未运行。
- optimizer：未运行。

诊断：
- 随机完整 RA smoke：seed 41，batch size 2，`16^3`。
- 真实 NIfTI smoke：seed 31，batch size 1，`16^3`。
- 完整训练循环 smoke：seed 53，2 batches，AdamW，learning rate `1e-4`，2 次 optimizer step。
- 默认尺寸频域 smoke：seed 61，batch size 1，`96^3`。
- ReconstructionLoss：默认 `0.45 L1 + 0.30 MS-SSIM + 0.15 Edge + 0.10 FFL`。
- RA 默认 guidance 内部权重：latent 0.35、multiscale image 0.25、gradient 0.20、windowed multiband frequency 0.15。
- 三路 gradient cap：structure 0.08、guidance 0.08、appearance 0.06；总 cap 0.15。
- Moment 训练开关：不存在；Moment 仍只用于训练后推理/分析。

### 10. GPU、CUDA、Python、PyTorch
- Python：3.13.7
- PyTorch：2.11.0+cpu
- CUDA available：False
- CUDA / GPU / AMP：未获得，未验证。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未运行正式训练和测试评估。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN / Gradient MAE：未获得。
- 最新正式真实测试结果仍为 EXP-0001 E0：MAE 0.107510，PSNR 16.068155，SSIM 0.529396。

随机完整 RA smoke 事实：
- raw loss 11.050734；structure 10.638406；guidance 0.304689；appearance 0.107639。
- weighted MedReCL 0.002400。
- structure/guidance/appearance gradient ratio：10.118992 / 0.100155 / 0.105177。
- structure/guidance/appearance scale：0.005390 / 0.544610 / 0.388957。
- hard 全体素比例 0.196533；hard 前景内比例 0.200099。
- 原 FFL 0.006504；新 RA frequency 0.252760；绝对差 0.246256，不再重复。
- 所有已产生梯度有限。

精确 top-k 事实：
- 100 个前景体素、quantile 0.80 时选中 20 个 hard 体素。
- 前景内 hard 比例 0.200000。
- adaptive weight mean 1.000000。

真实 NIfTI `16^3` smoke 事实：
- raw loss 9.086047；structure 8.540684；guidance 0.407364；appearance 0.138000。
- weighted MedReCL 0.003210。
- structure/guidance/appearance scale：0.009094 / 0.492167 / 0.776538。
- hard 全体素比例 0.000732；hard 前景内比例 0.230769（前景仅 13 个体素，ceil 后选 3 个）。
- loss 和所有已产生梯度有限。

完整两 batch 训练循环事实：
- optimizer steps 2；skipped steps 0。
- loss 0.264191；rec loss 0.262055；weighted MedReCL 0.002136。
- 所有聚合 metrics 为有限值。

`96^3` frequency backward 事实：
- loss 0.213406。
- prediction gradient absolute sum 1.083886，梯度有限。
- 当前 CPU 实测该单项耗时约 0.0417 秒；该时间不能外推到完整 GPU 训练。

### 12. 与可比基线的差值
- 未运行正式训练/测试，PSNR、SSIM、MAE 差值未获得。
- smoke loss 不可与 E0 或 GT-MedReCL++ 测试结果比较。

### 13. 是否严格可比；不可比时写明原因
- 不严格可比。
- 原因：仅执行缩小模型/随机张量/单 patch 诊断；没有同 seed、同 epoch、同 checkpoint 在 test split 上评估。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- `py_compile` 成功。
- `train_ramedreclpp.py --help` 退出码 0。
- `train_ramedreclpp.py` Ruff 全部通过。
- RA model Ruff 仍报告与旧 GT model 相同的 9 个继承问题：3 个未使用 import、2 个未使用局部变量、4 个歧义变量名；本轮没有新增 Ruff 问题。
- `git diff --check` 通过。
- 随机、真实 NIfTI、recon-only、两 batch optimizer 和 `96^3` frequency smoke 均未出现 NaN/Inf。
- 未运行 CUDA/AMP，显存和混合精度异常情况未获得。

### 15. 遗留问题
- 需要在目标 CUDA GPU 上用默认 `96^3`、base_channels=32 跑一批，记录峰值显存、耗时和三路 gradient ratio/scale。
- 需要运行至少 1 epoch 真实数据 smoke，检查 CSV 字段、checkpoint 保存和验证链路。
- 需要与 E0 做同 split、seed、epoch、预处理和评估脚本的正式对比。
- 需要通过消融确认 prediction latent、hard multiscale image、gradient、windowed multiband frequency 各自是否提升最终指标。
- EMA teacher 是否在 40 例数据上稳定仍需从长程 latent loss 与验证指标判断。

### 16. 本轮事实性结论
事实：
- Moment 没有被加入训练，仍保持训练后不确定性推理用途。
- 训练 hard-region 现在严格由 error + target gradient 构造，并在前景内精确 top-k。
- Hard 日志已与实际 hard mask 对齐。
- RA frequency 已与主 ReconstructionLoss FFL 形成不同数学目标。
- reconstruction guidance 已从旧 structure contrast 中拆出并独立做 gradient cap。
- 真实 smoke 中 guidance scale 约 0.492，而 structure scale 约 0.009，说明新 guidance 不再被旧 contrast 的大梯度统一压制。
- 未修改旧 GT-MedReCL++ 文件。

解释/推测：
- 本轮消除了已确认的实现冲突，但是否提升 CT->MRI 的 PSNR/SSIM/MAE/HFEN 必须由严格可比训练决定。
- 独立 guidance 梯度通道比 EXP-0003 更符合“MedReCL 服务 reconstruction”的设计目标。

下一步建议：
- 先在 CUDA 上运行 1 epoch 真实 smoke；确认显存、速度、CSV、gradient scale 后再启动 100 epoch 主实验。
- 正式实验应保持 E0 的数据划分、seed、训练轮数、checkpoint 选择和测试脚本不变。

### 17. 供下一位分析者重点判断的问题
- 默认三路 cap 0.08/0.08/0.06 与总 cap 0.15 是否在真实完整训练中稳定。
- guidance gradient ratio 是否长期落在可学习区间，而不是再次降至接近 0。
- 新多频带 frequency 是否改善 HFEN/Gradient MAE，还是与 gradient consistency 冗余。
- prediction online encoder + GT EMA teacher 的 latent loss 是否持续下降且不发生表征坍缩。
- RA 新增收益是否来自真正 reconstruction guidance，而不是 checkpoint 或评估波动。

---

## EXP-0006：新增干净的 E0 + Region-aware PatchNCE（RG-ReCL）实验线

### 1. 本轮实验编号与时间
- 实验编号：EXP-0006
- 时间：2026-08-17 06:24:09 +08:00
- 类型：代码实现 / 训练链路诊断；未运行正式训练或正式评估。

### 2. 本轮目标
- 按最新对话方案新增一个完整、独立的 RG-ReCL 版本。
- 保持 E0 backbone、ReconstructionLoss、数据、checkpoint、验证、测试和不确定性评估链路不变。
- 一次训练中只增加一个辅助方法：Region-aware PatchNCE。
- 固定总损失：`L_total = L_rec + 0.05 * L_region_patchnce`。
- 不在 RG-ReCL 中加入 EMA teacher、旧 Med-ReCL、Moment training weight、额外 image/gradient/frequency consistency。
- 不修改旧 GT 或已存在的 RA 文件。

### 3. 修改前基线版本 / 模型 / commit
事实：
- 本地 HEAD：`ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- E0 backbone 来源为 `model_attnres3d_gtmedreclpp.py` 中 `use_medrecl=False` 的模型路径。
- 本轮结束时 `git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py` 无输出。
- `model_attnres3d_ramedreclpp.py` 与 `train_ramedreclpp.py` 也未在本轮继续修改。

### 4. 理论依据
事实：
- paired CT->MRI 数据允许把 prediction feature 与 GT MRI feature 的同一空间位置定义为正样本。
- 同一病例内其他空间位置作为 negative candidates。
- Region hard mining 不是额外 loss；它根据 `abs(pred - GT)` 的 top 20% 区域提高 PatchNCE anchor 的采样概率和 anchor loss 权重。
- PatchNCE temperature 固定为 0.07，feature dimension 固定为 64，lambda 固定为 0.05。

解释/推测：
- PatchNCE 约束 prediction/GT 的局部特征对应关系，region sampling 减少简单区域主导训练的概率。
- 该方案比 RA-MedReCL++ 更干净，能单独检验 region-aware contrast 是否真正改善 paired CT->MRI reconstruction。

### 5. 实际修改文件
- 新增：`model_attnres3d_rgrecl.py`
- 新增：`train_rgrecl.py`
- 修改：`HANDOFF.md`
- 未修改：`model_attnres3d_gtmedreclpp.py`
- 未修改：`train_gtmedreclpp.py`
- 未修改：`model_attnres3d_ramedreclpp.py`
- 未修改：`train_ramedreclpp.py`

### 6. 每个文件具体修改内容
- `model_attnres3d_rgrecl.py`
  - 复用已验证 E0 的数据、backbone、ReconstructionLoss、评估、checkpoint、Moment 与 MC-Dropout 推理工具。
  - 新 `AttnResCTtoMRI` 强制调用 E0 `use_medrecl=False`，不会实例化旧 target encoder、EMA teacher 或 Med-ReCL projection 参数。
  - 新增三尺度共享 sampled-vector MLP projection heads；默认 feature dimension 64。
  - GT MRI 通过现有 E0 encoder 提取三尺度特征，不新增大 target network。
  - GT encoder 路径使用 deterministic eval + stop-gradient；执行后恢复 encoder 原训练状态。
  - 新增 `RegionPatchNCEConfig`：temperature 0.07、hard ratio 0.20、feature dim 64、anchors 256、negative pool 1024。
  - 新增 `RegionPatchNCELoss`。
  - positive：prediction decoder feature 与 GT encoder feature 的同位置 patch。
  - negative：同一病例 GT feature 的其他位置；若 negative pool 含 anchor 同位置，会从 denominator 中 mask 掉。
  - 每尺度根据下采样后的 `abs(pred-GT)` 精确选择 top 20% hard positions。
  - sampling probability：普通位置为 1，hard 位置为 `1 + error`。
  - sampled hard anchor 的 per-anchor NCE 权重同样为 `1 + error`。
  - 先采 anchor/negative vectors，再投影到 64 维，避免生成完整 `64x96x96x96` 投影激活。
  - 新增训练日志指标：真实 hard 比例、采样 hard 比例、正样本 cosine similarity。
  - 新增 E0 checkpoint 兼容加载：旧 E0 权重严格加载，RG projector 保持新初始化。
  - 新增专用 `train_one_epoch()`；每 batch 只进行一次 `loss.backward()` 和一次 optimizer step。
- `train_rgrecl.py`
  - 基于原训练/验证/测试编排复制为新入口，不改旧入口。
  - import 改为 `model_attnres3d_rgrecl`。
  - 删除全部 Med-ReCL CLI、criterion、日志和控制台字段。
  - 新增参数：`--rgrecl_lambda`、`--rgrecl_temperature`、`--rgrecl_hard_ratio`、`--rgrecl_feature_dim`、`--rgrecl_num_patches`、`--rgrecl_negative_pool`、`--no_rgrecl`。
  - 默认启用 RG-ReCL；`--no_rgrecl` 可运行同入口 E0 对照。
  - 增加 RG 参数范围校验。
  - CSV/控制台记录 raw/weighted RG loss、hard 比例、sampled hard 比例和 positive similarity。
  - validation/test 仍只计算重建指标，不把 PatchNCE 或 Moment 混入 PSNR/SSIM/MAE。
- `HANDOFF.md`
  - 追加 EXP-0006。

### 7. 实际运行命令
已运行：
```powershell
Copy-Item -LiteralPath train_gtmedreclpp.py -Destination train_rgrecl.py
python -m py_compile model_attnres3d_rgrecl.py train_rgrecl.py
python train_rgrecl.py --help
ruff check model_attnres3d_rgrecl.py train_rgrecl.py --output-format concise
git diff --check
git diff -- model_attnres3d_gtmedreclpp.py train_gtmedreclpp.py
```

已运行随机完整 forward/loss/backward smoke：
```powershell
@'
# seed=79；batch=2、16^3、base_channels=4、feature_dim=8；
# 验证 L_total 公式、投影头梯度、hard sampling、无 EMA 参数。
'@ | python -
```

已运行两 batch optimizer smoke、真实 NIfTI smoke 和 Moment 推理回归：
```powershell
@'
# seed=83 + 真实数据 seed=31；执行 train_one_epoch 两个 batch、
# data/dataset/train 真实 16^3 patch backward、forward_mu_var_cov。
'@ | python -
```

已运行 E0 checkpoint/state 兼容性模拟：
```powershell
@'
# 创建同配置 E0(use_medrecl=False) state_dict，strict=True 加载到 RG-ReCL；
# 比较普通 forward 输出。
'@ | python -
```

正式训练命令：未运行。

正式评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 真实 smoke 使用 `data/dataset/train` 中一个训练病例的随机 `16^3` patch。
- 数据文件与数据划分未修改。
- train/val/test 仍为 40/8/3，与 E0 一致。
- 正式训练未运行，因此本轮没有实际遍历完整划分。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
正式实验：
- seed：未运行。
- epoch：未运行。
- batch size：未运行。
- learning rate：未运行。
- optimizer：未运行。

代码默认配置：
- seed 42；epochs 100；batch size 1；learning rate `1e-4`；optimizer AdamW；cosine scheduler。
- ReconstructionLoss 完全沿用 E0：0.45 L1 + 0.30 MS-SSIM + 0.15 Edge + 0.10 Focal Frequency。
- RG lambda 0.05；temperature 0.07；hard ratio 0.20；feature dim 64。
- anchors 256/病例/尺度；negative pool 1024/病例/尺度。
- 默认 patch size `96^3`。
- Moment training 开关：不存在；Moment 仅在训练后推理/评估路径使用。

诊断配置：
- 随机 backward seed 79；batch 2；`16^3`；base_channels 4；feature dim 8；anchors 32；negative pool 64。
- optimizer smoke seed 83；2 batches；AdamW `1e-4`；anchors 16；negative pool 32。
- 真实 NIfTI smoke seed 31；batch 1；`16^3`。

### 10. GPU、CUDA、Python、PyTorch
- Python：3.13.7
- PyTorch：2.11.0+cpu
- CUDA available：False
- GPU/CUDA/AMP：未获得，未验证。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未运行正式训练和 test 评估。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN / Gradient MAE：未获得。
- 最新正式真实测试结果仍为 EXP-0001 E0：MAE 0.107510，PSNR 16.068155，SSIM 0.529396。

随机 backward 事实：
- rec loss 0.255721。
- raw RG loss 10.288367。
- total loss 0.770139。
- `total - rec - 0.05*RG` 的绝对误差为 0。
- hard region fraction 0.201497；sampled hard fraction 0.265625。
- 所有梯度有限；projector gradient 非零。
- 模型中 EMA/旧 Med-ReCL target 参数数量为 0。

两 batch optimizer smoke 事实：
- total loss 0.747160；rec loss 0.309116；raw RG loss 8.760863；weighted RG loss 0.438043。
- hard fraction 0.201497；sampled hard fraction 0.281250。
- optimizer steps 2；skipped steps 0；所有聚合 metrics 有限。

真实 NIfTI smoke 事实：
- rec loss 0.461221；raw RG loss 6.743037；total loss 0.798373。
- hard fraction 0.201497；sampled hard fraction 0.208333；positive similarity 0.464218。
- loss 和梯度有限。

Moment 推理回归事实：
- mean/variance shape 均为 `(1,1,16,16,16)`，数值有限。
- 该结果只证明推理路径可运行，不是训练指标。

E0 兼容性事实：
- E0 state_dict 以 `strict=True` 加载到 RG 模型成功，所有 key matched。
- 加载后 E0 与 RG 普通重建 forward 的最大绝对输出差为 0.0。
- 默认 E0 参数量 30,384,737；RG-ReCL 参数量 30,424,353；新增 39,616（约 0.1304%）。

### 12. 与可比基线的差值
- 未运行正式评估，PSNR/SSIM/MAE/HFEN 差值未获得。
- smoke loss 不可与 E0 测试指标比较。

### 13. 是否严格可比；不可比时写明原因
- 本轮 smoke 不严格可比。
- 原因：未训练 100 epochs，未加载同一初始 checkpoint 完成正式训练，未在 test split 上评估。
- 代码层面已验证：加载相同 E0 state 后，辅助 loss 之外的普通 forward 输出完全一致。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- `py_compile` 成功。
- `train_rgrecl.py --help` 退出码 0，帮助中只有 RG-ReCL 参数，没有 Med-ReCL 参数。
- 两个新文件 Ruff 全部通过。
- `git diff --check` 通过。
- 随机/真实数据 forward、backward、两 batch optimizer 和 Moment inference 均未出现 NaN/Inf。
- 初版曾先对完整特征图投影；诊断发现默认 `96^3`、64 通道会产生过大激活，已在同轮改为先采样 vectors 再投影。

数值风险：
- 随机初始化两 batch smoke 中 weighted RG loss 0.438043，高于 rec loss 0.309116。
- 这是固定 `lambda=0.05` 与标准 InfoNCE 初始量级共同产生的真实结果，不代表代码公式错误，但可能导致早期辅助梯度偏强。
- 未获得 CUDA 默认配置的独立梯度范数比例，不能判断正式训练是否冲突。

### 15. 遗留问题
- 需要在目标 CUDA GPU 上用默认 `96^3` 跑至少一个真实 batch，记录峰值显存、耗时、rec/RG loss 和总梯度范数。
- 需要观察固定 lambda 0.05 是否在前几百步让 reconstruction loss 恶化；当前按用户指定未加 warmup 或 gradient cap。
- 需要运行 1 epoch 真实 smoke，验证 CSV、checkpoint、val 和 test 编排。
- 最终必须与 E0 使用相同 seed、split、epoch、预处理、checkpoint 选择和评估脚本。
- 需要保存 sampled hard fraction；若长期接近原始 0.20，说明 `1+error` 的采样增强太弱。

### 16. 本轮事实性结论
事实：
- 已新增独立 `model_attnres3d_rgrecl.py` 与 `train_rgrecl.py`。
- RG-ReCL 实际训练只有 E0 ReconstructionLoss + Region-aware PatchNCE。
- positive、negative、temperature、hard ratio、hard weight、lambda 和 feature dimension 均已按方案实现。
- hard mining 作用于 PatchNCE sampling/anchor weighting，不是额外 hard loss。
- Moment、EMA、额外 image/gradient/frequency consistency 均未加入 RG training。
- 新方法只增加约 0.13% 参数，且先采样后投影。
- 旧 GT 和 RA 文件未修改。

解释/推测：
- 当前代码足以验证“Region-aware PatchNCE 是否适配 paired CT->MRI E0”这一单一假设。
- 是否提升重建质量尚无证据；固定 lambda 的初始相对量级值得重点监控。

下一步建议：
- 先执行默认模型的一批 CUDA smoke，再执行 1 epoch 真实训练。
- 正式主实验之前不要再叠加 RA、EMA、uncertainty-guided 或其他辅助 loss，以保持因果归因清楚。

### 17. 供下一位分析者重点判断的问题
- 固定 lambda 0.05 下，RG 梯度相对 reconstruction 梯度是多少，是否需要仅作为后续消融讨论而不是直接改公式。
- sampled hard fraction 是否稳定显著高于 0.20。
- prediction decoder feature 与 stop-gradient GT encoder feature 的 positive similarity 是否随训练上升。
- Region-aware PatchNCE 是否改善 PSNR/SSIM/MAE/HFEN/Gradient MAE，而不是只降低 contrastive loss。
- E0 与 RG 的正式实验是否真正共享相同初始化、数据顺序和评估口径。

## EXP-0007：将未运行的 RA-MedReCL++ 原位收敛为精简 RA-ReCL

### 1. 本轮实验编号与时间
- 实验编号：EXP-0007。
- 时间：2026-08-17 06:39:57 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 按最新方案直接修改尚未正式运行的 `model_attnres3d_ramedreclpp.py` 与 `train_ramedreclpp.py`，不再新增另一套 RA 训练入口。
- 将总损失严格收敛为 `L_total = L_rec + 0.05 * L_latent + 0.05 * L_region_patchnce`。
- 保留 prediction/GT latent alignment 与 Region-aware PatchNCE，删除旧 RA 中和 E0 重复或可能冲突的模块。

### 3. 修改前基线版本 / 模型 / commit
- 当前分支：`main`。
- 当前本地 HEAD：`ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 远程：`origin = https://github.com/candicewalker244-cmd/ctri.git`。
- 重建基线：E0 `model_attnres3d_gtmedreclpp.py`，`use_medrecl=False`。
- 修改对象是 EXP-0003/EXP-0005 建立但从未正式训练的 RA-MedReCL++ 文件；EXP-0006 的 RG-ReCL 文件保持不变。

### 4. 理论依据
事实：
- 任务是配准的 paired CT->MRI voxel reconstruction，最终监督目标是预测 MRI 与 GT MRI 的重建质量。
- E0 的 `ReconstructionLoss` 已包含 L1、MS-SSIM、edge 和 frequency 四类约束。
- 精简版只增加两类不重复的表征目标：预测路径 decoder latent 与同位置 GT encoder latent 的余弦对齐，以及依据当前绝对重建误差选择 hard region 的 paired PatchNCE。
- 同位置 prediction/GT feature 是 positive；其他 GT 空间位置是 negatives；temperature 为 0.07；hard region 为误差 top 20%。

解释/推测：
- 删除额外 image/gradient/frequency loss 可降低与 E0 原损失重复优化的风险。
- 删除 EMA teacher、旧 CT/MRI structure/appearance contrast 和训练期 uncertainty 权重可减少辅助目标偏离 reconstruction 的风险。
- 上述机制是否实际提升 PSNR、SSIM、MAE、HFEN 或 Gradient MAE，当前没有正式实验结果支持。

### 5. 实际修改文件
- `model_attnres3d_ramedreclpp.py`。
- `train_ramedreclpp.py`。
- `HANDOFF.md`。
- 临时建立的重复文件 `model_attnres3d_rarecl.py` 已并入并改名为现有 RA 模型文件，最终工作区不存在该重复文件。

### 6. 每个文件具体修改内容
- `model_attnres3d_ramedreclpp.py`：
  - 将旧 RA-MedReCL++ 整体收敛为精简 `RA-ReCL`。
  - 复用 EXP-0006 已验证的 E0/RG 数据、backbone、重建、评估与 Moment 推理 API。
  - 新增 `RAReCLConfig`：`temperature=0.07`、`hard_ratio=0.20`、`feature_dim=64`、每尺度 latent samples 256、region patches 256、negative pool 1024。
  - 三个尺度共享同一组 projection heads，同时供 latent alignment 与 Region-PatchNCE 使用，避免重复参数和重复 GT 特征提取。
  - latent alignment 在每病例每尺度均匀采样同位置 prediction decoder / GT encoder vectors，投影后使用 `mean(1-cosine_similarity)`。
  - Region-PatchNCE 使用 top 20% 绝对重建误差区域；hard 区采样权重及 sampled-hard anchor loss 权重均为 `1 + error`。
  - 先采样 vectors 再投影，避免对 96^3 完整体素图做 projection。
  - E0 state_dict 可 `strict=True` 初始化；新增投影头保留自身初始化。
  - `train_one_epoch` 每 batch 只做一次总 loss backward 和一次 optimizer step，并分别记录两项 raw/weighted loss 与相似度/hard fraction。
  - 未包含旧 structure/appearance contrast、EMA teacher、额外 image/gradient/frequency consistency 或 Moment-based training weight。
- `train_ramedreclpp.py`：
  - 基于最新 `train_rgrecl.py` 的已验证数据、训练、验证、checkpoint 和测试编排重建 RA 入口。
  - 改为导入 `model_attnres3d_ramedreclpp`，构造 `RAReCLConfig` 和 `RAReCLLoss`。
  - 新 CLI：`--rarecl_lambda_latent 0.05`、`--rarecl_lambda_region 0.05`、`--rarecl_temperature 0.07`、`--rarecl_hard_ratio 0.20`、`--rarecl_feature_dim 64`、`--rarecl_latent_samples 256`、`--rarecl_region_patches 256`、`--rarecl_negative_pool 1024`、`--no_rarecl`。
  - 默认输出目录改为独立的 `./output_rarecl`，避免覆盖 E0/RG 的 `./output` checkpoint 与日志。
  - CSV 和控制台分别记录 rec、latent、region、加权贡献、hard fraction、sampled-hard fraction、latent similarity 和 region positive similarity。
  - 保留普通 validation/test、best SSIM/PSNR/HFEN checkpoint、Moment 和 MC Dropout 训练后评估链路。
- `HANDOFF.md`：追加 EXP-0007。

### 7. 实际运行命令
- `python -m py_compile model_attnres3d_ramedreclpp.py train_ramedreclpp.py`。
- `ruff check model_attnres3d_ramedreclpp.py train_ramedreclpp.py`。
- `python train_ramedreclpp.py --help`，并筛选 RA 参数。
- 内联 Python：随机 batch 前向、两项 loss、公式核对、backward、projector gradient、参数残留和 E0 strict load/output 一致性检查。
- 内联 Python：2 batch `train_one_epoch` optimizer smoke 与 `use_rarecl=False` 回退测试。
- 内联 Python：读取真实 NIfTI 病例 `1BA001` 的 16^3 patch，执行 RA forward/backward 与 `forward_mu_var_cov()`。
- 内联 Python：统计 E0 与精简 RA 默认参数量。
- `git diff --check`、`git status --short` 和旧 GT/RG 文件 SHA256 核对。
- 正式训练命令：未运行。
- 正式 validation/test 命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录：`data/dataset`。
- 代码实际发现：train 40、val 8、test 3。
- 真实 smoke 使用 train 病例 `1BA001` 的随机 16x16x16 patch；不是正式评估。
- 正式训练沿用 E0/RG 的 `CTMRIDataset`、`clip01` 归一化、split 发现和交叉 split 校验逻辑，设计上与基线一致。
- 因本轮未正式训练，尚不能声称已完成严格可比实验。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 默认 seed：42。
- 默认 epoch：100；本轮正式训练未运行。
- 默认 batch size：1；两 batch smoke 使用 batch size 1，随机公式检查使用 batch size 2。
- 默认 learning rate：1e-4。
- optimizer：AdamW。
- scheduler：默认 cosine，minimum LR 1e-6。
- E0 reconstruction weights：L1 0.45、MS-SSIM 0.30、edge 0.15、frequency 0.10。
- 精简 RA：latent weight 0.05、Region-PatchNCE weight 0.05、temperature 0.07、hard ratio 0.20、feature dim 64。
- 默认 dropout 0.2、patch size 96x96x96、AMP 在 CUDA 时开启、max grad norm 5.0。
- `--no_rarecl` 可严格退回 E0-only loss；Moment 不参与训练。

### 10. GPU、CUDA、Python、PyTorch
- GPU：未获得（当前执行环境无 CUDA GPU）。
- CUDA available：False。
- CUDA runtime：None。
- Python：3.13.7。
- PyTorch：2.11.0+cpu。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及其他指标
- 本轮正式训练：未运行。
- 本轮正式 test：未运行。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN：未获得。
- Gradient MAE：未获得。
- 最新正式真实测试结果仍为 EXP-0001 的 E0：MAE 0.107510、PSNR 16.068155、SSIM 0.529396。

代码/数值 smoke 事实，不作为效果指标：
- 随机公式检查：rec 0.271279、raw latent 0.571716、weighted latent 0.028586、raw region 7.432250、weighted region 0.371613、total 0.671477。
- `total = rec + 0.05*latent + 0.05*region` 的浮点核对误差为 `2.38e-08`。
- hard fraction 0.201497，sampled-hard fraction 0.229167；投影头梯度非零且全部梯度有限。
- 两 batch optimizer smoke：optimizer steps 2、skipped 0；所有聚合 loss/相似度有限。
- `--no_rarecl` smoke 中 total 与 rec 差值为 0，两项 RA loss 均为 0。
- 真实 NIfTI smoke：rec 0.512464、raw latent 0.535457、raw region 6.953218、total 0.886898；所有 loss、梯度、Moment mean/variance 有限。
- E0 权重 `strict=True` 加载成功；相同权重下 E0 与 RA 普通 forward 最大输出差为 0.0。
- 默认 E0 参数 30,384,737；精简 RA 参数 30,424,353；新增 39,616，约 0.1304%。

### 12. 与可比基线的差值
- PSNR 差值：未获得。
- SSIM 差值：未获得。
- MAE 差值：未获得。
- HFEN / Gradient MAE 差值：未获得。
- 参数量相对 E0 增加 39,616，约 0.1304%；这是结构规模差值，不是质量收益。

### 13. 是否严格可比；不可比时写明原因
- 本轮不严格可比。
- 原因：没有在相同初始化、相同数据顺序、相同 100 epochs 和相同独立 test 上分别完成 E0 与精简 RA 正式训练评估。
- 已验证的代码可比性仅限：E0 权重严格加载后，普通 reconstruction forward 输出完全一致；数据、重建 loss、验证和测试 API 沿用同一路径。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- `py_compile`、Ruff、`--help`、`git diff --check` 均通过。
- 精简 RA 自身 state_dict 在同配置模型间 `strict=True` 回载成功，所有 key matched。
- 随机/真实 forward、backward、两 batch optimizer、E0 回退和 Moment inference 均未出现 NaN/Inf。
- 真实 16^3 smoke 随机 crop 的 target 值域为 0.0 至 0.007820，说明该小 patch 基本位于低强度区域；这不代表完整训练数据分布。
- 随机初始化时 weighted Region-PatchNCE 0.371613 高于 rec 0.271279；两 batch 平均 weighted region 0.394233 高于 rec 0.258005。
- 这是指定固定权重 0.05 与 InfoNCE 初始量级共同产生的真实数值风险，不是公式实现错误。
- CPU 环境无法验证默认 96^3 CUDA 显存、AMP 稳定性或训练速度。
- 初检发现默认 `./output` 会与 E0/RG 共用输出目录；已改为 `./output_rarecl`，当前不再存在默认覆盖风险。

### 15. 遗留问题
- 在目标 CUDA GPU 上用默认 96^3 跑至少一个真实 batch，记录峰值显存、耗时、两项 raw/weighted loss 和梯度范数。
- 正式训练前几百 step 重点观察 weighted region 是否压过 reconstruction，及 rec loss 是否相对 E0 恶化。
- 运行 1 epoch 流程 smoke，验证 CSV、checkpoint、validation 和 test 编排。
- 正式 E0、RG 与精简 RA 必须使用同一初始化 checkpoint、seed、split、epoch、数据顺序、checkpoint 选择与评估 crop。
- 当前 fixed lambda 0.05 按用户方案保留；是否需要 warmup 或更小权重只能由正式梯度/指标证据决定。

### 16. 本轮事实性结论
事实：
- 未运行的 RA-MedReCL++ 文件已原位改成精简 RA-ReCL，没有新增第二套 RA 训练入口。
- 实际训练目标只有 E0 reconstruction、prediction/GT latent alignment 和 Region-aware PatchNCE。
- 两项 RA loss 共用 GT encoder features 和三尺度 projection heads，不含 EMA teacher 或旧 Med-ReCL 参数。
- 额外 image/gradient/frequency consistency、structure/appearance contrast 和 uncertainty training weighting 已删除。
- E0 初始化兼容、总损失公式、梯度、两 batch optimizer、真实 NIfTI 和 Moment 推理均已通过 smoke。

解释/推测：
- 精简版比 EXP-0005 旧 RA 更容易归因，也减少了与 E0 重建损失重复和梯度竞争的来源。
- 当前不能断言精简 RA 会提高 CT->MRI 重建质量；Region-PatchNCE 初始加权量级偏大是最需要监控的风险。

下一步建议：
- 先做默认配置 CUDA 单 batch 与 1 epoch 流程验证，再决定是否开始完整 100 epochs。
- 不在得到精简 RA 独立结果前继续叠加新 loss，以免再次失去因果归因。

### 17. 供下一位分析者重点判断的问题
- 固定 0.05 下 latent 和 Region-PatchNCE 各自相对 reconstruction 的梯度范数与夹角是多少。
- Region-PatchNCE weighted loss 在前几百 step 是否快速下降到不主导 reconstruction 的量级。
- latent similarity 与 region positive similarity 上升时，PSNR/SSIM/MAE/HFEN/Gradient MAE 是否同步改善。
- sampled-hard fraction 是否稳定高于原始 hard ratio 0.20；若接近 0.20，`1+error` 采样增强可能过弱。
- 相同 E0 初始化下，RG-ReCL 与精简 RA-ReCL 的差值是否能归因于新增 latent alignment，而非数据顺序或 checkpoint 选择。

## EXP-0008：新增 RA-ReCL v2 完整实验线

### 1. 本轮实验编号与时间
- 实验编号：EXP-0008。
- 时间：2026-08-17 06:50:15 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 不修改 E0、RG-ReCL 或 EXP-0007 精简 RA-ReCL，另建一条独立新实验线。
- 一次性实现参考对话最终收敛方案：E0 reconstruction、零初始化 residual refinement、curriculum multi-scale Region-aware PatchNCE。
- 保持 Moment 仅用于训练后推理，同时让 residual refiner 进入 Moment 均值/方差路径。

### 3. 修改前基线版本 / 模型 / commit
- 当前分支：`main`。
- 当前本地 HEAD：`ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- E0 基线：`model_attnres3d_gtmedreclpp.py`，`use_medrecl=False`。
- 直接代码底座：EXP-0006 已验证的 E0/RG 公共 API；EXP-0007 文件本轮未修改。

### 4. 理论依据
事实：
- E0 已包含 L1、MS-SSIM、edge 和 frequency 重建约束；本轮不再重复增加这些 loss。
- 参考对话最终方案要求多尺度 Region-aware PatchNCE、残差细化和 hard-region curriculum 一次性运行。
- Multi-scale PatchNCE 使用三层 decoder feature `[d1,d2,d3]`，分别对应浅层、中层和近 bottleneck 深层，权重为 `0.20/0.30/0.50`。
- 同位置 prediction-path decoder / GT encoder feature 是 positive，其他 GT 空间位置是 negatives；temperature 0.07。
- hard score 使用当前 final MRI prediction 与 GT 的绝对误差，不使用 Moment uncertainty。
- residual refiner 直接接收 coarse MRI，预测幅度受限的局部残差，最终输出为 `clamp(coarse + residual, 0, 1)`。

解释/推测：
- 深层权重较高用于强调整体解剖/组织对应，浅层仍保留局部边缘信息。
- 零初始化 residual 输出层可使训练开始时 final MRI 与 E0 coarse MRI 完全一致，避免随机 refiner 破坏基线。
- 前 20% 仅重建可让 backbone/refiner 先形成基本 MRI 输出，再启用高初始量级 InfoNCE；这降低但不能消除辅助梯度过强风险。
- 是否提升正式重建指标当前没有证据。

### 5. 实际修改文件
- 新增 `model_attnres3d_rareclv2.py`。
- 新增 `train_rareclv2.py`。
- 更新 `HANDOFF.md`。
- 本轮未修改现有 E0、GT、RG 或 EXP-0007 精简 RA 文件。

### 6. 每个文件具体修改内容
- `model_attnres3d_rareclv2.py`：
  - 新增 `RAReCLV2Config`，固定多尺度 PatchNCE 与 curriculum 默认参数。
  - 新增 `ResidualRefinementHead3D`：`1->8->1` 两层 3D convolution、GELU、tanh bounded residual、默认 residual scale 0.10；末层 weight/bias 全零初始化。
  - residual refiner 新增一阶对角 Moment propagation；明确忽略 coarse/residual covariance，与项目 attention 外部的对角近似口径一致。
  - 新增 `AttnResCTtoMRI` v2 包装：普通 forward、feature forward、Moment forward 和继承的 MC Dropout forward 都使用 final refined MRI。
  - E0 state_dict 可 `strict=True` 加载；新增 projector/refiner 保留自身初始化。
  - 新增 `MultiScaleRegionPatchNCELoss`：三尺度加权、同位置 positive、其他位置 negative、top-error hard sampling、`1+error` hard sampling/anchor weighting。
  - curriculum：总训练进度 `<0.20` 为 stage 0，仅 reconstruction；`[0.20,0.60)` 为 stage 1，PatchNCE + top 30%；`>=0.60` 为 stage 2，PatchNCE + top 20%。
  - 总损失：stage 0 为 `L_rec`；stage 1/2 为 `L_rec + 0.05 * L_multiscale_region_patchnce`。
  - 每 batch 只执行一次 backward 和 optimizer step。
  - 日志新增 stage、effective hard ratio、effective lambda、hard/sample fraction、positive similarity、`mean/max |final-coarse|`。
- `train_rareclv2.py`：
  - 独立训练入口，默认输出 `./output_rareclv2`，避免覆盖其他实验。
  - 保留 E0/RG 的数据、重建 loss、validation/test、best SSIM/PSNR/HFEN checkpoint、Moment calibration 和 MC Dropout 评估编排。
  - 新 CLI 包括 lambda、temperature、curriculum 边界、middle/final hard ratio、三尺度权重、projection size、patch/negative 数、refiner channels/residual scale。
  - `--no_rareclv2` 可只关闭 PatchNCE；`--no_refiner` 可只关闭 refiner；两者同时关闭时严格退回 E0。
  - CSV 和控制台输出所有新增诊断指标。
- `HANDOFF.md`：追加 EXP-0008。

### 7. 实际运行命令
- `python -m py_compile model_attnres3d_rareclv2.py train_rareclv2.py`。
- `ruff check model_attnres3d_rareclv2.py train_rareclv2.py`。
- `python train_rareclv2.py --help` 并筛选 v2/refiner 参数。
- 内联 Python：E0 strict load、零初始化普通 forward/Moment mean/Moment variance 一致性检查。
- 内联 Python：curriculum 在 0、0.1999、0.20、0.5999、0.60、1.0 的边界检查。
- 内联 Python：随机 batch 三尺度 PatchNCE、refiner/projector gradient 与 NaN/Inf 检查。
- 内联 Python：分别以 start step 0、2、6 和 total steps 10 执行两 batch optimizer smoke，覆盖 stage 0/1/2。
- 内联 Python：同时关闭 v2/refiner 后与 E0 参数量和输出一致性检查。
- 内联 Python：真实 NIfTI `1BA001` 16^3 patch 单步训练、训练后 residual delta 与 Moment 推理检查。
- 内联 Python：默认 E0/v2 参数量与 v2 checkpoint strict roundtrip 检查。
- `rg` 旧 loss/EMA 残留检查、`git diff --check`、`git status --short`。
- 正式训练命令：未运行。
- 正式 validation/test 命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录：`data/dataset`。
- 已发现 split：train 40、val 8、test 3，与 EXP-0007 记录一致。
- 真实 smoke 使用 train 病例 `1BA001` 的随机 16x16x16 patch。
- v2 沿用 E0 的病例发现、paired CT/MRI 检查、`CTMRIDataset`、`clip01` 和 split 去重逻辑。
- 正式训练未运行，因此只能确认代码口径设计一致，不能确认完整实验严格可比。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 默认 seed：42。
- 默认 epoch：100；正式训练未运行。
- 默认 batch size：1。
- 默认 learning rate：1e-4。
- optimizer：AdamW。
- scheduler：cosine，minimum LR 1e-6。
- E0 reconstruction：L1 0.45、MS-SSIM 0.30、edge 0.15、frequency 0.10。
- v2 PatchNCE lambda：激活时 0.05；stage 0 有效值为 0。
- temperature 0.07；feature dim 64；patches 256；negative pool 1024。
- `[d1,d2,d3]` level weights：0.20、0.30、0.50。
- curriculum：0%-20% rec-only；20%-60% top 30%；60%-100% top 20%。
- refiner：hidden channels 8、residual scale 0.10、末层零初始化。
- 默认 dropout 0.2、patch 96^3、max grad norm 5.0、CUDA 时 AMP 开启。
- Moment 不参与训练。

### 10. GPU、CUDA、Python、PyTorch
- GPU：未获得（当前环境无 CUDA GPU）。
- CUDA available：False。
- CUDA runtime：None。
- Python：3.13.7。
- PyTorch：2.11.0+cpu。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及其他指标
- 本轮正式训练：未运行。
- 本轮正式 test：未运行。
- PSNR：未获得。
- SSIM：未获得。
- MAE：未获得。
- HFEN：未获得。
- Gradient MAE：未获得。
- 最新正式真实测试结果仍为 EXP-0001 E0：MAE 0.107510、PSNR 16.068155、SSIM 0.529396。

代码/数值 smoke 事实，不作为效果指标：
- 零初始化并加载相同 E0 state 后，E0 与 v2 普通 forward 最大差值 0.0；Moment mean 最大差值 0.0；Moment variance 最大差值 0.0。
- 随机 stage 1：rec 0.268293、raw PatchNCE 9.460297、weighted PatchNCE 0.473015、total 0.741307；所有梯度有限，projector/refiner 均有非零梯度。
- curriculum 边界输出：0.1999 仍为 stage 0；0.20 进入 stage 1/top 30%；0.60 进入 stage 2/top 20%。
- 两 batch stage 0：total=rec 0.257534、PatchNCE=0、effective lambda=0、optimizer steps 2。
- 两 batch stage 1：hard fraction 0.304443、sampled-hard fraction 0.333333、weighted PatchNCE 0.383106。
- 两 batch stage 2：hard fraction 0.201497、sampled-hard fraction 0.197917、weighted PatchNCE 0.383039。
- 真实 NIfTI 单步：rec 0.509106、raw PatchNCE 3.490124、weighted 0.174506、total 0.683612；训练后 mean absolute residual delta 0.0002045；Moment 数值有限。
- 默认 E0 参数 30,384,737；v2 参数 30,424,794；新增 40,057，约 0.1318%。
- v2 state_dict 同配置 `strict=True` roundtrip 所有 keys matched。

### 12. 与可比基线的差值
- PSNR / SSIM / MAE / HFEN / Gradient MAE 差值：未获得。
- 参数量较 E0 增加 40,057，约 0.1318%。
- 零初始化时输出差值为 0.0；这是初始化兼容性，不是训练后质量差值。

### 13. 是否严格可比；不可比时写明原因
- 本轮不严格可比。
- 原因：仅执行 CPU 小模型/小 patch smoke，没有用相同初始化、相同 100 epochs、相同数据顺序分别训练 E0 与 v2，也没有在独立 test split 正式评估。
- 代码层面已确认完全关闭 v2/refiner 时参数量、state 加载和输出都严格回到 E0。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
事实：
- `py_compile`、Ruff、CLI help、残留检查和 `git diff --check` 全部通过。
- 三个 curriculum stage、真实 NIfTI、普通 forward、backward、optimizer、Moment 和 checkpoint roundtrip 均无 NaN/Inf。
- 随机 stage 1 weighted PatchNCE 0.473015 高于 rec 0.268293；两 batch stage 1/2 weighted PatchNCE 约 0.383，高于 rec 约 0.258。
- 真实单 patch weighted PatchNCE 0.174506 低于 rec 0.509106，但单 patch 不代表完整训练分布。
- stage 0 refiner 首次 forward 的 residual delta 为 0，这是末层零初始化的预期行为；一次 optimizer step 后 delta 非零。
- 当前 CPU 环境不能验证默认 96^3 CUDA 显存、AMP 稳定性或速度。

数值风险：
- curriculum 在 20% 边界将有效 lambda 从 0 直接切到 0.05，PatchNCE 初始量级可能造成 loss/梯度突变。
- 当前严格按参考方案实现固定 0.05，未自行增加 lambda ramp 或 gradient cap。

### 15. 遗留问题
- 在目标 CUDA GPU 上运行默认 96^3 单 batch，记录峰值显存、耗时、raw/weighted PatchNCE、总梯度范数和 refiner delta。
- 运行跨过 20% curriculum 边界的短训练，确认 rec loss、grad norm 与 AMP scale 不发生异常跳变。
- 运行 1 epoch 流程 smoke，验证 CSV、checkpoint、validation、best selection 和 test 编排。
- 正式 E0/v2 必须使用相同初始化 checkpoint、seed、split、数据顺序、epoch、评估 crop 和 checkpoint 选择。
- 若 20% 边界出现明显冲突，只能在取得日志证据后测试 lambda ramp 消融，不能把修改后实验与当前 v2 混为同一版本。

### 16. 本轮事实性结论
事实：
- 已新增独立 `model_attnres3d_rareclv2.py` 与 `train_rareclv2.py`；旧实验文件未修改。
- v2 实际训练只有 E0 reconstruction 与一个 curriculum multi-scale Region-PatchNCE，不含独立 latent loss。
- residual refiner 直接修正 coarse MRI，零初始化确保初始普通/Moment 输出均与 E0 一致。
- curriculum 三阶段、top 30%/20%、多尺度权重、固定 0.05 和日志均按方案实现。
- Moment 仍是训练后推理，但已包含 refiner；MC Dropout 通过继承的 `self.forward()` 同样包含 refiner。
- 新方法仅增加约 0.132% 参数。

解释/推测：
- 该版本能同时验证“局部残差修正”和“课程式多尺度区域对比”组合是否改善 CT->MRI。
- 因两个机制同时加入，若正式结果变化，不能仅凭主实验区分 refiner 与 PatchNCE 的单独贡献，需要后续消融。
- 当前不能断言质量会提升，20% 边界的辅助量级是主要风险。

下一步建议：
- 先执行默认 GPU 单 batch和跨 curriculum 边界短训练，再开始完整实验。
- 正式主实验后至少保留 `E0 + refiner only`、`E0 + curriculum PatchNCE only` 两个消融开关组合。

### 17. 供下一位分析者重点判断的问题
- stage 0 到 stage 1 时 weighted PatchNCE、rec loss、总梯度范数和 AMP scale 是否突变。
- `mean/max |final-coarse|` 是否从 0 稳定增加而不过早达到 residual scale 上限。
- 三尺度 positive similarity 是否同步提高；深层 0.50 权重是否过强。
- sampled-hard fraction 在 stage 1 是否高于 0.30、stage 2 是否高于 0.20；若没有，`1+error` 采样增强可能过弱。
- 最终 PSNR/SSIM/MAE/HFEN/Gradient MAE 是否改善，而不只是 PatchNCE loss 下降。
- refiner-only 与 PatchNCE-only 消融能否说明各自独立贡献及组合是否存在协同。

## EXP-0009：现有 E0 是否需要重跑及新实验可比性诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0009。
- 时间：2026-08-17 06:55:40 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 确认 E0 是否已经完成、是否需要代码修改或重新训练。
- 逐项核对 E0、RG、精简 RA 和 RA-ReCL v2 的数据划分、默认训练参数与评估 API 是否一致。

### 3. 修改前基线版本 / 模型 / commit
- 当前分支 `main`，本地 HEAD `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- E0 已有输出：`output_gt_e0_rec_40_8_3_e100`。
- E0 训练日志实际包含 100 行 epoch，epoch 1 至 100。

### 4. 理论依据
事实：
- 可比实验至少要保持病例 split、预处理、训练预算、重建 loss、checkpoint 选择与 test 评估一致。
- 新模块应只改变声明的机制，不能暗中改变数据和评价口径。

解释/推测：
- 当前代码层面的可比口径高度一致；历史 E0 因缺失原始命令、checkpoint 和 commit，不能追溯证明所有运行时参数严格一致。

### 5. 实际修改文件
- 仅更新 `HANDOFF.md`。
- E0、RG、精简 RA、v2 模型和训练代码均未修改。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 E0 完成状态、split/API/默认参数核对结果和是否需要重跑的结论。

### 7. 实际运行命令
- 读取 EXP-0001 与 E0 结果/日志文件。
- 搜索四个训练入口的 patch、归一化、模型规模、epoch、batch、LR、scheduler、loss weights、seed 和 eval crop 默认值。
- 内联 Python 调用四个模块的 `discover_cases()`，逐 split 比较病例 ID 和顺序。
- 内联 Python 检查新模块与 E0 的 Dataset、ReconstructionLoss、validation、uncertainty API 对象一致性。
- 内联 Python读取 `E0_test_results.csv` 与 `E0_log.csv`。
- 训练：未运行。
- 评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录均为 `data/dataset`。
- train 40：四条代码路径病例 ID 和顺序完全一致。
- val 8：四条代码路径病例 ID 和顺序完全一致。
- test 3：`1BA005`、`1BB030`、`1BC052`，四条路径完全一致。
- `CTMRIDataset` 在 RG、精简 RA、v2 中均与 E0 是同一 Python 类对象。
- 数据划分与预处理代码层面一致。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 四个入口默认均为：seed 42、epoch 100、batch size 1、LR 1e-4、cosine scheduler、min LR 1e-6。
- 均为 patch 96^3、CT/MRI `clip01`、base channels 32、bottleneck blocks 6、dropout 0.2。
- E0 reconstruction weights 均为 L1 0.45、MS-SSIM 0.30、edge 0.15、frequency 0.10。
- eval crop 均为 150^3。
- RG、精简 RA、v2 的 `ReconstructionLoss` 与 `validate_one_epoch` 均直接指向 E0 同一实现。
- 历史 E0 实际命令和实际 seed 未获得；日志可确认 epoch 100、每 epoch 40 optimizer steps，但不能反证所有参数均采用默认值。

### 10. GPU、CUDA、Python、PyTorch
- 本轮未重新查询；沿用 EXP-0008 当前环境：无 CUDA GPU、Python 3.13.7、PyTorch 2.11.0+cpu。
- 历史 E0 训练 GPU/CUDA：未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及其他指标
- E0 已完成，不是未运行状态。
- MAE：0.107510。
- PSNR：16.068155 dB。
- SSIM：0.529396。
- HFEN：0.730762。
- Gradient MAE：0.074969。
- 本轮未产生新模型测试指标。

### 12. 与可比基线的差值
- 新 RG、精简 RA、v2 尚未正式训练，和 E0 的 PSNR/SSIM/MAE/HFEN/Gradient MAE 差值均未获得。

### 13. 是否严格可比；不可比时写明原因
- 数据 split、当前默认超参数、Dataset、ReconstructionLoss 和评估 API：一致。
- 现有历史 E0 与未来新实验：高度可比，但不能证明达到完全严格可比。
- 原因：历史 E0 原始运行命令、checkpoint、commit、实际 seed/GPU 未保留，无法确认运行时是否覆盖过默认参数。
- 现阶段方法验证可以直接使用现有 E0 作为基线；论文级严格对照或多 seed 统计时应把 E0 纳入同一受控批次重跑。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 首次病例比较脚本将 `CasePaths` 错当字典，报 `TypeError: 'CasePaths' object is not subscriptable`；修正为属性访问后成功。
- 未发现 split 不一致、API 分叉或默认参数差异。

### 15. 遗留问题
- 历史 E0 checkpoint 缺失，不能用它作为所有新方法的统一初始化文件。
- 历史 E0 原始命令和硬件环境未获得。
- 新方法正式运行后需要确认输出 CSV 的 test 三病例数和指标字段完整。

### 16. 本轮事实性结论
事实：
- E0 已经完成 100 epochs 和独立 test，不需要为了“补代码”重新运行，也不需要修改 E0。
- 新实验的数据划分、预处理、重建 loss 和评估函数与当前 E0 代码一致。
- 当前只需运行尚未得到正式结果的新方法。

解释/推测：
- 现有 E0 足以作为当前方法筛选基线。
- 如果未来需要发表级严格结论，最好在统一脚本/初始化管理下补跑 E0 与候选最优模型的多 seed 对照。

下一步建议：
- 当前不重跑 E0，先运行 RG、精简 RA 或优先运行最终 v2/消融。
- 不修改数据 split。

### 17. 供下一位分析者重点判断的问题
- 新方法实际命令是否覆盖默认参数。
- 新输出是否确实使用同一 test 三病例和 150^3 eval crop。
- 初筛后是否需要对 E0 与最佳候选执行统一初始化、多 seed 的严格复跑。

## EXP-0010：云算力运行前的三实验端到端预检与子目录兼容修复

### 1. 本轮实验编号与时间
- 实验编号：EXP-0010。
- 时间：2026-08-17 07:39:59 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 在用户租赁云算力前确认 RG-ReCL、精简 RA-ReCL、RA-ReCL v2 是否能完整执行。
- 明确三组实验是独立运行还是 checkpoint 串行训练。
- 使用真实 40/8/3 数据执行缩小配置的完整训练、验证、checkpoint 回载、Moment 校准和测试链路。

### 3. 修改前基线版本 / 模型 / commit
- 当前分支 `main`，本地 HEAD `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 用户在本轮开始前将三组文件移动到 `RG-ReCL/`、`jingjian RA-ReCL/`、`zuixinwanzheng RA-ReCL v2/`。
- E0 仍位于项目根目录。

### 4. 理论依据
事实：
- 编译和单 batch smoke 不能证明完整 checkpoint/test 编排可运行，必须实际走过训练、验证、保存、回载和测试。
- PyTorch 2.6+ 的 `torch.load` 默认 `weights_only=True`，不能直接加载包含 optimizer/RNG/NumPy 状态的完整训练 checkpoint。
- 三条方法各自从头建立模型和 optimizer；代码没有自动加载上一种方法输出的逻辑。

解释/推测：
- 缩小 CPU 端到端通过显著提高了流程可靠性，但不能代替目标 GPU 上默认 96^3、AMP 和 100 epoch 的验证。

### 5. 实际修改文件
- `RG-ReCL/model_attnres3d_rgrecl.py`。
- `RG-ReCL/train_rgrecl.py`。
- `jingjian RA-ReCL/model_attnres3d_ramedreclpp.py`。
- `jingjian RA-ReCL/train_ramedreclpp.py`。
- `zuixinwanzheng RA-ReCL v2/model_attnres3d_rareclv2.py`。
- `zuixinwanzheng RA-ReCL v2/train_rareclv2.py`。
- `.gitignore`。
- `HANDOFF.md`。

### 6. 每个文件具体修改内容
- 三个模型文件：根据当前子目录结构加入基于 `__file__` 的项目根目录/RG 依赖目录解析，解决跨目录 import。
- 三个训练文件：对本项目自己生成、可信的完整 checkpoint 显式使用 `torch.load(..., weights_only=False)`，兼容 PyTorch 2.6+；resume 和最终 best load 均修复。
- `.gitignore`：新增 `_cloud_preflight_*/`，防止预检 checkpoint/CSV 被上传。
- `HANDOFF.md`：追加本轮问题、修复和端到端结果。
- 模型结构、loss、数据、超参数默认值和指标口径均未改变。

### 7. 实际运行命令
- 三条子目录训练入口 `--help` 启动测试。
- 三组模型/训练文件 `py_compile` 与 Ruff。
- RG 缩小端到端：1 epoch、patch/eval crop 16^3、base channels 4、bottleneck 1、feature dim 8、patches 16、negative pool 32、CPU FP32、关闭 MC/图片。
- RG 使用 `latest.pth` resume 启动并回载 `best.pth` 完成 Moment calibration/test。
- 精简 RA 同等缩小配置完整 1 epoch。
- v2 同等缩小配置完整 1 epoch，refiner channels 4。
- 检查每组 checkpoint 文件、`log.csv`、`test_results.csv`、`moment_calibration.csv`。
- 最终 `py_compile`、Ruff、`rg torch.load`、`git diff --check`、`git status --short`。
- 默认 96^3 CUDA 命令：未运行。
- 正式 100 epoch 命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录：`data/dataset`。
- 三次端到端预检均实际发现并校验 train 40、val 8、test 3。
- CT/MRI 病例 ID、shape、affine 和跨 split 去重均通过。
- 与 E0 数据划分一致。
- 预检将 patch/eval crop 临时改为 16^3，仅用于流程测试，因此预检指标不可与 E0 正式结果比较。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 预检 seed 42、epoch 1、batch 1、LR 1e-4、AdamW、CPU FP32。
- 预检 base channels 4、bottleneck 1、patch/eval crop 16^3。
- RG：feature dim 8、patches 16、negative pool 32。
- 精简 RA：feature dim 8、latent samples 16、region patches 16、negative pool 32。
- v2：feature dim 8、patches 16、negative pool 32、refiner channels 4。
- 三组均关闭 MC Dropout 对照和测试图片，但保留普通 validation/test 与 Moment calibration。
- 正式默认仍为 100 epochs、96^3、base channels 32、bottleneck 6、dropout 0.2、AMP、方法默认 feature dim 64/patches 256/negative pool 1024。

### 10. GPU、CUDA、Python、PyTorch
- 预检设备：CPU。
- CUDA/GPU：未使用。
- Python 3.13.7，PyTorch 2.11.0+cpu。
- 目标云 GPU/CUDA/显存：未获得、未验证。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及其他指标
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。
- 三组本轮结果是缩小模型/1 epoch/16^3 流程 smoke，不是正式效果指标。
- RG smoke test：MAE 0.261590、PSNR 9.73、SSIM 0.459254、Gradient MAE 0.100968、HFEN 0.712923。
- 精简 RA smoke test：MAE 0.261557、PSNR 9.76、SSIM 0.460714、Gradient MAE 0.099550、HFEN 0.705709。
- v2 smoke test：MAE 0.255124、PSNR 9.97、SSIM 0.471724、Gradient MAE 0.102069、HFEN 0.696286。
- 以上 smoke 指标禁止用于方法优劣结论。

### 12. 与可比基线的差值
- 正式差值：未获得。
- smoke 配置与 E0 正式配置不同，不计算差值。

### 13. 是否严格可比；不可比时写明原因
- 三组预检彼此使用相同缩小配置和同一数据 split，但只运行 1 epoch，主要目的为流程验证。
- 与 E0 正式 100 epoch/默认模型结果不严格可比。
- 三组正式方法彼此独立运行，不是 RG checkpoint -> 精简 RA -> v2 的串行训练。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
失败 1：
- 用户整理子目录后，三个入口最初均因跨目录 import 失败，出现 `ModuleNotFoundError`。
- 已修复；三条 `--help` 均退出码 0。

失败 2：
- RG 首次完整预检完成训练/验证/保存后，在 test 前加载 `best.pth` 时出现 PyTorch `Weights only load failed`。
- 根因是 PyTorch 2.6+ 默认 `weights_only=True`，checkpoint 含 NumPy RNG 等完整状态。
- 三个训练入口的 resume/best load 均显式改为 `weights_only=False`；仅应加载本项目自己生成且可信的 checkpoint。
- 修复后 RG resume/best/test、精简 RA 完整链路、v2 完整链路均成功。

其他事实：
- 三组均完成 40 optimizer steps，skipped steps 0。
- 每组均生成 `best.pth`、`best_ssim.pth`、`best_psnr.pth`、`best_hfen.pth`、`latest.pth`、`epoch_0001.pth`。
- 每组均生成 log、test results 和 Moment calibration CSV。
- RG 预检因修复后在同目录重复执行 test，`test_results.csv`/calibration CSV 有两行；正式新目录单次运行不会出现该预检重复。
- 未出现 NaN/Inf。

### 15. 遗留问题
- 目标云 GPU 上默认 96^3 的峰值显存、CUDA AMP、速度仍未验证。
- 100 epoch 长期稳定性、20% curriculum 边界在正式 v2 中的梯度变化仍未验证。
- 租赁 GPU 后应先做默认尺寸短预检，确认不 OOM，再开始长训练。
- 云端 PyTorch/CUDA 版本和 GPU 型号需写入下一轮 HANDOFF。

### 16. 本轮事实性结论
事实：
- 修复后，RG、精简 RA、v2 三条真实数据缩小版端到端链路全部跑通。
- 当前能确认代码流程完整，但不能保证任意云 GPU 上默认 96^3、100 epoch 百分百成功。
- 三组是独立实验；运行先后不影响权重，RG 不会自动传给精简 RA，精简 RA 也不会自动传给 v2。

解释/推测：
- 按 RG -> 精简 RA -> v2 的顺序便于逐步观察方法增量，但只是实验组织顺序，不是训练继承顺序。
- 若云算力预算有限，可优先做每组默认尺寸短预检，再决定是否完整跑三组。

下一步建议：
- 云端先记录 GPU/CUDA/PyTorch，再对每组运行默认 96^3 的短流程。
- 主实验可按 RG、精简 RA、v2 顺序独立运行；v2 完成后主三组结束。
- 若需要解释 v2 模块贡献，再运行 refiner-only 与 PatchNCE-only 消融。

### 17. 供下一位分析者重点判断的问题
- 云端默认 batch 1、96^3 是否 OOM，峰值显存是多少。
- AMP 下 loss/gradient 是否有限，是否发生 skipped optimizer step。
- 正式 v2 在 20%/60% curriculum 边界是否稳定。
- 三组正式输出是否各自使用独立 save_dir，避免覆盖。
- 是否需要在主三组后补 v2 两个消融实验。

## EXP-0011：云算力基础镜像选择诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0011。
- 时间：2026-08-17 07:44:56 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 从云平台截图中的基础镜像选择适合三组 3D CT->MRI 长训练的稳定环境。

### 3. 修改前基线版本 / 模型 / commit
- 代码基线不变：EXP-0010，HEAD `ef0141b731675b477fee2c50f4a7ea61d1823b03`。

### 4. 理论依据
- PyTorch 官方提供 PyTorch 2.5.1 与 CUDA 12.4 的正式安装组合。
- 相比截图中的 Python 3.13/CUDA 12.8 新组合，Python 3.12/Ubuntu 22.04 对本项目常用科学计算依赖更保守。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- 追加云端镜像建议；未修改代码。

### 7. 实际运行命令
- 联网查询 PyTorch 官方 previous versions 页面。
- 云端命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；仍为 train/val/test = 40/8/3。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch 1、LR 1e-4、AdamW。

### 10. GPU、CUDA、Python、PyTorch
- 推荐基础镜像：PyTorch 2.5.1、CUDA 12.4、Ubuntu 22.04、Python 3.12.7。
- 推荐选择专业版用于长期训练。
- GPU 型号/显存尚未选择、未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及其他指标
- 本轮未训练/评估，新指标未获得；正式 E0 指标沿用 EXP-0009。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮仅环境选择，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 云端未启动，无运行异常可记录。

### 15. 遗留问题
- 需要下一步 GPU 列表/价格截图才能确定具体卡型。
- 需要云端实测默认 96^3 峰值显存与 AMP。

### 16. 本轮事实性结论
事实：
- 截图中建议选 `2.5.1-cuda12.4-ubuntu22.04-Python3.12.7`。

解释/推测：
- 该组合预计比最新 Python 3.13/CUDA 12.8 镜像具有更低的第三方依赖兼容风险。

下一步建议：
- 选择专业版和上述镜像，再根据 GPU 列表优先考虑 40/48 GB 显存卡。

### 17. 供下一位分析者重点判断的问题
- 最终 GPU 型号、显存、价格和驱动版本。
- 24 GB 卡是否能在 AMP、batch 1、96^3 下完成默认单 batch。

## EXP-0012：云端 GPU 型号选择诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0012。
- 时间：2026-08-17 07:47:07 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 在云平台提供的 RTX 4090 24GB、RTX 4090D 24GB、RTX 3090 24GB、RTX 3060 12GB、A100 SXM4 80GB、RTX 5090 32GB、RTX 4090 48GB 中，为三组 3D CT->MRI 实验选择 GPU。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0011；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 三组待运行模型及 E0 代码均未修改。

### 4. 理论依据
事实：
- 三组正式配置均使用 96x96x96 体数据、batch size 1、AMP，并包含 3D backbone 与额外对比分支；目标 GPU 的默认配置峰值显存尚未实测。
- NVIDIA Blackwell 架构需要 CUDA 12.8 起始支持；PyTorch 2.7 官方发布说明加入 Blackwell 与 CUDA 12.8 预编译包支持。

解释/推测：
- 48GB 显存比 24GB 对未实测的 3D 默认配置有更大的显存余量，可降低首次正式运行 OOM 风险。
- A100 80GB 的显存余量最大，但若价格明显高于 RTX 4090 48GB，对本轮单卡验证的性价比可能较低。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 GPU 兼容性、显存风险和选型结论；未修改训练或模型代码。

### 7. 实际运行命令
- 联网查阅 PyTorch 2.7 官方发布说明与 NVIDIA CUDA Toolkit/架构支持矩阵。
- 云端训练、评估及显存测试命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；仍为 train/val/test = 40/8/3，与当前 E0 代码划分一致。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、CUDA AMP、96^3 patch。
- 各方法自己的 loss 与开关未改变。

### 10. GPU、CUDA、Python、PyTorch
- 首选：RTX 4090 48GB，配合 EXP-0011 选择的 PyTorch 2.5.1、CUDA 12.4、Ubuntu 22.04、Python 3.12.7 镜像。
- 显存最稳妥选项：A100 SXM4 80GB；是否选择取决于价格。
- 经济备选：RTX 4090/4090D 24GB，但必须先实测默认 96^3 单 batch 峰值显存。
- 不建议：RTX 3060 12GB。
- RTX 5090 32GB 不应与当前 PyTorch 2.5.1/CUDA 12.4 镜像组合；若使用 5090，应改用明确支持 Blackwell 的 PyTorch 2.7+ / CUDA 12.8+ 环境并重新验证依赖。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练或评估，新 PSNR、SSIM、MAE、HFEN、Gradient MAE 均未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得；本轮仅完成硬件选择诊断。

### 13. 是否严格可比；不可比时写明原因
- 不涉及模型结果比较，因此无严格可比性结论。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 云端尚未创建实例，训练/评估未运行，无新报错或数值异常。
- 默认 96^3 在 24GB、48GB 或 80GB 上的实际峰值显存均未获得。

### 15. 遗留问题
- 云端启动后记录准确 GPU 型号、驱动、CUDA、Python、PyTorch 与可用显存。
- 正式 100 epoch 前，先在默认 96^3、batch 1、AMP 下执行短预检并记录 `nvidia-smi` 峰值显存。
- 尚未获得各 GPU 的实际租赁价格，无法计算精确成本/性能比。

### 16. 本轮事实性结论
事实：
- 当前选择的 PyTorch 2.5.1/CUDA 12.4 镜像不适合作为 RTX 5090 Blackwell 的目标环境。
- 48GB 和 80GB 选项提供的显存高于 24GB；默认 96^3 的真实显存占用仍未获得。

解释/推测：
- RTX 4090 48GB 是当前列表中兼顾兼容性、显存余量和预计成本的首选。
- 若 A100 80GB 与 RTX 4090 48GB 价差很小，A100 80GB 更稳妥；若价差明显，RTX 4090 48GB 更合理。

下一步建议：
- 选择专业版、PyTorch 2.5.1/CUDA 12.4/Python 3.12.7 镜像和 RTX 4090 48GB。
- 实例启动后先做环境检查和默认配置短预检，再开始三组独立的 100 epoch 正式训练。

### 17. 供下一位分析者重点判断的问题
- RTX 4090 48GB 的实际租赁单价是否接近 A100 SXM4 80GB。
- 默认 96^3 在 RTX 4090 48GB 上的峰值显存、单 step 耗时和 AMP 稳定性。
- 若 48GB 卡仍 OOM，具体峰值发生在 reconstruction、PatchNCE、refiner 还是测试阶段。

## EXP-0013：RTX 4090 48GB 主机与存储配置确认

### 1. 本轮实验编号与时间
- 实验编号：EXP-0013。
- 时间：2026-08-17 07:50:05 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 根据云平台最终配置页确认 GPU 数量、主机可用状态、系统内存、数据盘和计费方式是否适合三组正式实验。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0012；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据均未修改。

### 4. 理论依据
事实：
- 页面选择为 RTX 4090 48GB，GPU 数量为 1 张，主机提供 15 核 CPU、60GB 系统内存、50GB 免费数据盘，价格为 2.28 元/小时。
- 截图账户余额为 27.56 元；按 2.28 元/小时静态计算约可运行 12.1 小时，不含其他可能费用。
- 当前本地项目文件总量约 1.018GB，其中 `data/` 约 0.922GB，共 308 个文件。
- 主机列表中 `GPU20001` 显示剩余 GPU 数量 `1/8`；其余截图内主机显示 `0/8`。

解释/推测：
- 50GB 数据盘相对当前约 1GB 项目有充足余量，预计可容纳三组训练 checkpoint、CSV 和测试输出；正式运行后仍需监控输出目录增长。
- 60GB 系统内存对当前 batch size 1、40/8/3 数据链路预计足够，但真实峰值仍需云端测量。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加最终云主机规格、可用主机和本地项目/数据容量核对结果；未修改代码。

### 7. 实际运行命令
- PowerShell 递归统计项目、`data/` 文件大小与文件数量。
- 云端训练、评估和环境检查命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据未改变；train/val/test 仍为 40/8/3，与 E0 一致。
- 本轮仅统计数据文件体积，没有读取或改写数据内容。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 页面最终选择：单张 RTX 4090 48GB。
- 主机：西南E区 `GPU20001`，15 核 CPU、60GB RAM。
- 镜像沿用 EXP-0011：PyTorch 2.5.1、CUDA 12.4、Ubuntu 22.04、Python 3.12.7。
- 实际驱动、`torch.cuda`、显存识别值尚未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练或评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮仅确认硬件和存储配置，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 云端实例尚未运行，无新训练/评估报错。
- 默认 96^3 的 GPU 显存和系统内存峰值未获得。

### 15. 遗留问题
- 创建实例后必须用 `nvidia-smi` 和 PyTorch 实测确认系统识别为单 GPU、约 48GB 可用显存。
- 必须先进行默认 96^3 短预检，再开始 100 epoch。
- 三组正式输出的实际磁盘占用尚未获得，需要训练期间监控。
- 三组各 100 epoch 的总耗时尚未获得，当前余额是否足够完成全部实验不能确定。

### 16. 本轮事实性结论
事实：
- 页面选择明确为 1 张 RTX 4090 48GB，不是将 GPU 数量设置为 2 张。
- 当前唯一显示有剩余 GPU 的截图内主机是 `GPU20001`。
- 当前项目与数据合计约 1.018GB，低于免费 50GB 数据盘容量。
- 当前余额按截图单价约对应 12.1 个运行小时。

解释/推测：
- 当前页面配置适合进入实例创建阶段；50GB 暂无必要付费扩容。

下一步建议：
- 保持 GPU 数量 1、按量计费、扩容 0GB，选择 `GPU20001` 并创建实例。
- 创建后先执行环境和显存检查，再上传代码与数据并进行默认配置短预检。
- 根据短预检测得的每 epoch 时间估算三组总成本，再决定充值或改用包天；停止使用时及时关机以停止计费。

### 17. 供下一位分析者重点判断的问题
- `nvidia-smi` 是否报告单卡约 48GB 显存，而非其他虚拟或聚合形式。
- PyTorch 是否正确识别 CUDA、GPU 名称和显存。
- 三组默认 96^3 单 batch 的峰值显存、CPU RAM、单 step 时间和 checkpoint 总占用。

## EXP-0014：AutoDL 数据盘上传路径诊断

> 更正：本轮将平台误识别为 AutoDL，因此 `/root/autodl-tmp` 结论不适用于当前实例。该判断已由 EXP-0015 根据实际目录截图与算家云资料推翻；当前有效数据盘路径为 `/root/sj-tmp`。

### 1. 本轮实验编号与时间
- 实验编号：EXP-0014。
- 时间：2026-08-17 07:55:35 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 明确如何确保项目代码、`data/` 和训练输出上传/写入 AutoDL 的 50GB 数据盘，而不是 30GB 系统盘。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0013；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- AutoDL 官方文档定义数据盘挂载点为 `/root/autodl-tmp`，系统盘为根目录 `/` 及其下除特殊挂载目录之外的路径。
- 写入 `/root/autodl-tmp` 的内容写入数据盘；JupyterLab 默认工作目录 `/root` 本身属于系统盘。
- 当前训练脚本的默认输出目录为相对路径，因此输出实际落盘位置取决于启动命令时的当前工作目录。

解释/推测：
- 将仓库固定放在 `/root/autodl-tmp/ctri`，并始终从该目录启动训练，可使相对输出目录也保留在数据盘。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 AutoDL 数据盘挂载路径、上传目录和验证方法；未修改代码。

### 7. 实际运行命令
- 联网查询 AutoDL 官方环境与目录文档。
- 云端命令：未运行；待实例 WebShell 中执行 `df -h`、`findmnt`、`pwd` 和目录创建命令。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；仍为 train/val/test = 40/8/3。
- 计划云端路径为 `/root/autodl-tmp/ctri/data/dataset`，保持代码所需目录结构。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 实例页面显示 RTX 4090 单卡、系统盘 30GB、数据盘 50GB、内存 60GB。
- 镜像显示 PyTorch 2.5.1、CUDA 12.4、Ubuntu 22.04、Python 3.12.7。
- WebShell 实际环境检查尚未运行，GPU 显存实际识别值尚未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为存储路径诊断，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 云端训练和评估未运行。
- 尚未验证 SFTP/文件管理器上传后的实际挂载设备。

### 15. 遗留问题
- 在 WebShell 执行 `df -h / /root/autodl-tmp` 与 `findmnt -T /root/autodl-tmp`，确认挂载和剩余容量。
- 上传后执行 `realpath /root/autodl-tmp/ctri`、`df -h /root/autodl-tmp/ctri` 与 `du -sh /root/autodl-tmp/ctri`。
- 训练开始后确认全部 `output_*` 目录位于 `/root/autodl-tmp/ctri` 下。

### 16. 本轮事实性结论
事实：
- AutoDL 数据盘路径是 `/root/autodl-tmp`。
- 将 SFTP/文件管理器远程目标设为 `/root/autodl-tmp/ctri` 可确保项目写入数据盘。
- `/root/ctri` 默认属于系统盘，不应作为本项目上传位置。

解释/推测：
- 当前约 1.018GB 项目上传到数据盘后，50GB 容量足以开始三组实验。

下一步建议：
- 先在 WebShell 建立 `/root/autodl-tmp/ctri`，再通过 SFTP 或文件管理器上传。
- 代码可直接在该目录 `git clone`，本地 `data/` 单独上传并保持 `data/dataset` 层级。

### 17. 供下一位分析者重点判断的问题
- 云端 `df/findmnt` 是否确认 `/root/autodl-tmp` 是独立 50GB 挂载。
- 云端项目是否完整位于 `/root/autodl-tmp/ctri`，特别是 `data/dataset` 与各输出目录。
- 训练是否始终从数据盘项目根目录启动，避免相对输出写入系统盘。

## EXP-0015：更正云平台识别并确定算家云数据盘路径

### 1. 本轮实验编号与时间
- 实验编号：EXP-0015。
- 时间：2026-08-17 07:59:39 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 更正 EXP-0014 的平台识别错误，根据实际文件管理器目录确定当前算家云实例的 50GB 数据盘路径及正确上传方式。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0014；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- 用户实际文件管理器 `/root/` 下显示 `sj-data`、`sj-fs`、`sj-tmp`，不存在 `autodl-tmp`。
- 算家云公开部署资料明确说明数据盘为 `sj-tmp`，完整路径为 `/root/sj-tmp`，免费容量 50GB。
- 算家云资料使用 `/root/sj-data` 访问平台提供的模型/数据社区内容，并将需写入的数据复制到 `/root/sj-tmp`。

解释/推测：
- EXP-0014 因界面相似而误判为 AutoDL；该路径结论不能用于当前实例。
- 当前项目应放在 `/root/sj-tmp/ctri`，从该目录启动训练以保证相对输出也写入数据盘。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：在 EXP-0014 添加失效警告，并追加算家云路径更正、上传步骤和待验证项。

### 7. 实际运行命令
- 联网检索算家云公开教程中 `sj-tmp`、`sj-data` 的用途。
- 云端命令：未运行；待在 WebShell 执行挂载验证。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；仍为 train/val/test = 40/8/3。
- 计划数据路径更正为 `/root/sj-tmp/ctri/data/dataset`。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 实例页面：RTX 4090 单卡、60GB RAM、30GB 系统盘、50GB 数据盘。
- 镜像：PyTorch 2.5.1、CUDA 12.4、Ubuntu 22.04、Python 3.12.7。
- WebShell 实际检测未运行，GPU 显存与挂载设备信息未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮仅更正存储路径，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 诊断错误：EXP-0014 错将当前平台识别为 AutoDL，并给出不存在的 `/root/autodl-tmp` 路径。
- 更正：当前平台为算家云，数据盘路径为 `/root/sj-tmp`。
- 云端训练/评估未运行，无数值异常。

### 15. 遗留问题
- 在 WebShell 执行 `df -hT / /root/sj-tmp` 与 `findmnt -T /root/sj-tmp`，用当前实例实测确认挂载设备和剩余容量。
- 上传后确认 `/root/sj-tmp/ctri/data/dataset` 存在且病例数量正确。
- 正式训练前确认所有输出目录均位于 `/root/sj-tmp/ctri`。

### 16. 本轮事实性结论
事实：
- 当前实例不存在 `/root/autodl-tmp`。
- 当前算家云实例的 50GB 数据盘路径是 `/root/sj-tmp`。
- 在文件管理器停留于 `/root/` 时直接上传会写入系统盘；必须先进入 `sj-tmp`。

解释/推测：
- 将项目放入 `/root/sj-tmp/ctri` 并在该目录运行，能避免代码、数据、相对输出占用系统盘。

下一步建议：
- 文件管理器双击 `sj-tmp`，顶部路径确认显示 `/root/sj-tmp/`，再创建 `ctri` 或在 WebShell 中直接克隆仓库。
- 将本地 `data/` 单独上传到 `/root/sj-tmp/ctri/data`，保持 `data/dataset` 结构。

### 17. 供下一位分析者重点判断的问题
- `df/findmnt` 是否实测确认 `/root/sj-tmp` 对应 50GB 数据盘。
- GitHub clone 与 SFTP 上传后，项目根目录和数据目录层级是否正确。
- 三个训练入口是否始终从 `/root/sj-tmp/ctri` 启动。

## EXP-0016：算家云 SFTP 批量上传失败诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0016。
- 时间：2026-08-17 08:05:27 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 诊断 Windows SFTP 客户端批量上传当前整个工作目录时部分文件失败的原因，并确定更可靠的云端传输流程。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0015；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- 截图显示 SFTP 目标为 `xn-e.suanjiayun.com:2020`，传输队列中 95 个文件成功、37 个失败、185 个仍在队列。
- 明确失败原因包含“密码错误”，其他失败项原因以“已从服务器...”开头，符合会话断开后的连锁失败表现。
- 上传集合包含 `.git/`、`.ruff_cache/`、多个 `__pycache__/`、`_cloud_preflight_*/checkpoints/` 和历史输出文件。

解释/推测：
- 因已有 95 个文件成功，主机、端口和目标目录至少在传输开始时可用；首要故障更可能是 SFTP 客户端重连时使用了错误/过期密码，或多并发连接触发重新认证失败，而不是特定文件格式损坏。
- 大量无关的小文件和 checkpoint 增加重连、队列和部分失败概率，但不是“密码错误”的直接根因。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：记录 SFTP 失败证据、最可能根因、无需上传的目录和推荐重传方案；未修改代码。

### 7. 实际运行命令
- 联网检索 SFTP/SSH 认证与文件传输说明。
- 云端命令：未运行。
- 本地重新上传命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；仍为 train/val/test = 40/8/3。
- 数据尚未确认完整上传至 `/root/sj-tmp/ctri/data/dataset`。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 云端实例规格/镜像沿用 EXP-0015。
- 训练环境命令尚未运行，实际 GPU 显存、CUDA 和 PyTorch 检测结果未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为传输故障诊断，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- SFTP 批量上传部分失败：37 个失败，原因包括“密码错误”和服务器连接中断；185 个文件仍排队。
- 尚未进入训练或评估，因此无训练数值异常。

### 15. 遗留问题
- 需要从当前实例“SSH/SFTP”页面重新复制实时主机、端口、用户名和实例/SFTP密码，清除客户端旧密码后重新连接。
- 需要把并发传输数降到 1 至 2，并先用单个小文件验证写入 `/root/sj-tmp`。
- 需要确认远端部分上传目录是否保留了不完整文件；正式运行前应使用 GitHub clone 的干净代码树并单独上传数据。

### 16. 本轮事实性结论
事实：
- 本次失败发生在 SFTP 认证/连接层，截图未显示 Python 代码解析或训练错误。
- 整个本地工作区包含大量不应上传的缓存、Git 内部文件、预检 checkpoint 和历史输出。

解释/推测：
- 最可靠方案是云端在 `/root/sj-tmp` 通过 GitHub 克隆代码，仅使用 SFTP 上传 `data/`，避免继续同步整个工作目录。

下一步建议：
- 停止当前队列，用平台当前“SSH/SFTP”凭据建立新会话并测试单文件。
- 在 WebShell 执行 `git clone` 获取代码；将 `data/` 打包为单个归档上传，再在数据盘解压。

### 17. 供下一位分析者重点判断的问题
- 新凭据是否能稳定完成单文件上传和一次断线重连。
- 客户端是否保存了旧密码或启用了过高并发连接数。
- 云端干净仓库与本地 HEAD、`HANDOFF.md` 是否一致，数据病例数是否为 40/8/3。

## EXP-0017：算家云单归档上传与解压前检查

### 1. 本轮实验编号与时间
- 实验编号：EXP-0017。
- 时间：2026-08-17 08:12:41 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 在用户改为上传单个 `ctri.tar` 后，明确安全解压、目录层级校验和归档清理步骤。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0016；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- 截图显示 `ctri.tar` 正在通过网页上传，当前进度约 77%，尚未完成。
- tar 归档可能包含顶层 `ctri/` 目录，也可能直接包含项目文件；两种结构需要不同解压目标。

解释/推测：
- 单归档上传可显著减少大量小文件逐个认证/重试造成的失败，但仍需等待上传 100% 并检查归档列表后再解压。

### 5. 实际修改文件
- 仅 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加归档上传状态、解压前检查、两种目录结构的处理和验证要求；未修改代码。

### 7. 实际运行命令
- 云端上传：进行中，截图约 77%。
- 云端解压、校验、训练、评估命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；预期仍为 train/val/test = 40/8/3。
- 归档尚未解压，云端数据目录和病例数未验证。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 沿用当前算家云实例和镜像选择；实际 WebShell 环境检查尚未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为文件传输/解压准备，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- `ctri.tar` 尚在上传，未发生可确认的新错误。
- 不能在上传未完成时解压，否则会得到 truncated archive/Unexpected EOF 一类错误。

### 15. 遗留问题
- 等待上传达到 100%，确认文件大小稳定。
- 使用 `tar -tf` 检查归档顶层结构，再选择解压目录。
- 解压后验证 `HANDOFF.md`、三组训练入口与 `data/dataset`，并核验空间占用。

### 16. 本轮事实性结论
事实：
- 当前归档上传进行中，尚不能解压或删除本地源文件。
- 解压前必须判断归档是否自带 `ctri/` 顶层目录。

解释/推测：
- 若归档完整，单文件上传方式比逐文件上传更可靠。

下一步建议：
- 上传完成后在 WebShell 执行 `ls -lh` 和 `tar -tf ... | head`。
- 解压和项目结构验证全部通过后，才删除云端 `ctri.tar` 释放数据盘空间。

### 17. 供下一位分析者重点判断的问题
- `ctri.tar` 实际所在路径和完整文件大小。
- 归档首层是 `ctri/` 还是直接项目文件。
- 解压后项目是否位于 `/root/sj-tmp/ctri` 且数据 split 完整。

## EXP-0018：算家云项目归档解压完成

### 1. 本轮实验编号与时间
- 实验编号：EXP-0018。
- 时间：2026-08-17 08:15:17 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 确认 `ctri.tar` 的归档层级并将完整项目解压到算家云数据盘。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0017；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 本地模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- `tar -tf /root/sj-tmp/ctri.tar | head -20` 显示归档顶层为 `ctri/`。
- 因归档自带 `ctri/`，使用 `tar -xf /root/sj-tmp/ctri.tar -C /root/sj-tmp` 会得到 `/root/sj-tmp/ctri`，不会多套一层目录。

解释/推测：
- 解压命令无报错并返回 shell 提示符，说明 tar 解包过程完成；文件完整性、数据 split 和挂载位置仍需命令验证。

### 5. 实际修改文件
- 云端：由归档解压生成 `/root/sj-tmp/ctri/` 内容。
- 本地：仅更新 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- 云端项目文件未改写内容，仅从 `ctri.tar` 解包。
- `HANDOFF.md`：记录归档首层、实际解压命令和当前完成状态。

### 7. 实际运行命令
- 云端已运行：`tar -tf /root/sj-tmp/ctri.tar | head -20`。
- 云端已运行：`tar -xf /root/sj-tmp/ctri.tar -C /root/sj-tmp`。
- 云端训练/评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 本地预期 split 目录为 `data/dataset/train`、`val`、`test`，数量应为 40/8/3。
- 云端目录和病例数量尚未实际验证，因此是否与基线一致尚不能确认。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 沿用当前算家云实例/镜像；实际环境检查尚未运行。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮仅完成传输/解压，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- `tar -tf` 与 `tar -xf` 截图未显示错误。
- 归档包含 `.git/` 等完整本地工作区内容，增加磁盘占用但不妨碍解压。
- 训练尚未运行，无数值异常。

### 15. 遗留问题
- 验证 `pwd`、`df -h .`、`du -sh .`、关键文件和 `data/dataset` 三个 split。
- 验证云端代码 HEAD/HANDOFF 是否与本地一致。
- 验证完成后删除云端 `ctri.tar` 释放重复占用；删除前不得操作。

### 16. 本轮事实性结论
事实：
- 归档自带 `ctri/` 顶层目录。
- 已执行正确的解压命令，目标项目路径应为 `/root/sj-tmp/ctri`。
- 尚未运行训练或评估。

解释/推测：
- 若后续目录、split 和磁盘检查通过，即可进入云端环境预检阶段。

下一步建议：
- 先验证项目、数据和数据盘挂载，再删除归档。
- 随后检查 GPU/CUDA/PyTorch，而不是直接开始 100 epoch。

### 17. 供下一位分析者重点判断的问题
- 云端 `/root/sj-tmp/ctri` 是否包含三个新实验入口和最新 `HANDOFF.md`。
- train/val/test 是否分别为 40/8/3，NIfTI 文件是否完整。
- `/root/sj-tmp` 是否确实对应 50GB 数据盘，解压后剩余空间是多少。

## EXP-0019：算家云数据盘、项目和 split 完整性验证

### 1. 本轮实验编号与时间
- 实验编号：EXP-0019。
- 时间：2026-08-17 08:16:40 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 验证解压后的项目位置、独立数据盘挂载、可用空间、三个实验目录和数据 split 数量。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0018；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据内容未修改。

### 4. 理论依据
事实：
- `df -hT .` 可确认当前项目所在目录对应的实际挂载设备、文件系统、容量和剩余空间。
- split 一级病例目录数量应与基线 40/8/3 一致。

解释/推测：
- 路径、挂载和 split 检查全部通过后，可以排除项目误放系统盘及病例目录缺失这两类问题；仍不能替代逐文件校验和训练环境检查。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`。
- 云端本轮仅执行读取/统计命令，未修改项目文件。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加云端实际挂载、空间、项目目录和 split 验证结果。

### 7. 实际运行命令
- 云端已运行：`cd /root/sj-tmp/ctri`、`pwd`、`df -hT .`、`du -sh .`。
- 云端已运行：`ls -l HANDOFF.md`、三个实验目录 `ls -d`。
- 云端已运行：按 split 统计一级病例目录数量。
- 云端训练/评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 云端实际结果：train 40、val 8、test 3。
- 病例目录数量与 E0 基线一致。
- 本轮尚未逐个核对每个病例的 CT/MRI NIfTI 文件对和文件哈希。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- 尚未运行 `nvidia-smi` 或 Python 环境检测；实际值未获得。
- 实例/镜像页面选择沿用前轮记录。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为文件/挂载检查，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 路径、挂载、空间、实验目录和 split 统计均无报错。
- 尚未开始训练，无数值异常。

### 15. 遗留问题
- 删除已完成且验证后的 `/root/sj-tmp/ctri.tar`，释放重复占用空间。
- 检查三个训练入口文件、GPU/CUDA/PyTorch 和 `numpy`/`nibabel`/`matplotlib` 导入。
- 运行默认 96^3 CUDA 短预检并记录峰值显存。

### 16. 本轮事实性结论
事实：
- 云端项目路径为 `/root/sj-tmp/ctri`。
- 该路径位于 `/dev/nvme0n1` 的 XFS 文件系统，容量 50GB、已用约 2.5GB、可用约 48GB，挂载点 `/root/sj-tmp`。
- 解压后项目约 1.1GB。
- `HANDOFF.md` 与三个实验目录存在。
- train/val/test 病例目录数为 40/8/3，与基线一致。

解释/推测：
- 项目与数据已正确放入数据盘，可以进入环境检测阶段。

下一步建议：
- 删除 tar 归档后立即检查 GPU、CUDA、Python、PyTorch 和依赖版本。
- 环境检查通过后运行默认配置短预检，不直接启动 100 epoch。

### 17. 供下一位分析者重点判断的问题
- 云端实际 GPU 名称和显存是否与购买配置一致。
- PyTorch 是否为 CUDA 构建且 `torch.cuda.is_available()` 为 True。
- `nibabel`、`numpy`、`matplotlib` 是否可导入，三个训练入口是否能执行 `--help`。

## EXP-0020：算家云 GPU 识别成功与缺失 NiBabel 诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0020。
- 时间：2026-08-17 08:18:07 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 检查云端实际 GPU/驱动/显存，并验证 Python、PyTorch、NumPy、NiBabel、Matplotlib 运行依赖。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0019；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- `nvidia-smi` 的 CUDA Version 表示驱动可支持的最高 CUDA 版本，不等同于当前 PyTorch wheel 的编译 CUDA 版本；应以 `torch.version.cuda` 为准。
- 项目数据加载直接导入 `nibabel`，缺失时三个训练入口都会在导入基线模块阶段失败。

解释/推测：
- 当前 NVIDIA 驱动版本高于 CUDA 12.4 运行要求，预计可向后兼容镜像中的 PyTorch CUDA 12.4 构建；仍须用 `torch.cuda.is_available()` 和实际张量运算确认。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`。
- 云端尚未安装依赖或修改代码。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 GPU/驱动/显存实测值和 `nibabel` 缺失报错。

### 7. 实际运行命令
- 云端已运行：`nvidia-smi`。
- 云端已运行：Python 多依赖导入与版本打印脚本。
- 依赖安装、训练、评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；云端已确认 train/val/test = 40/8/3。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- GPU：`NVIDIA GeForce RTX 4090`，单卡。
- 总显存：49140 MiB，约 48GB；空闲时使用 1 MiB。
- NVIDIA driver：595.71.05。
- `nvidia-smi` 显示 driver-supported CUDA Version 13.2。
- PyTorch 版本、`torch.version.cuda`、`torch.cuda.is_available()`：因后续导入 `nibabel` 报错导致脚本提前终止，未获得。
- Python 实际版本：未从本次截图获得完整版本字符串。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为运行环境检查，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- Python 环境检查报错：`ModuleNotFoundError: No module named 'nibabel'`。
- 报错发生在依赖导入阶段，尚未进入模型构建、训练或评估。
- GPU 本身识别正常，无运行进程。

### 15. 遗留问题
- 安装与本地已验证环境一致的 `nibabel==5.4.2`，再运行完整依赖检查。
- 获取 Python、PyTorch、PyTorch CUDA runtime、CUDA availability、NumPy 和 Matplotlib 版本。
- 运行一个 CUDA 张量运算和三个训练入口 `--help`。

### 16. 本轮事实性结论
事实：
- 云平台确实提供一张约 48GB 的 RTX 4090，硬件选择与页面一致。
- 当前阻塞条件是缺失 Python 包 `nibabel`，不是 CUDA OOM 或模型错误。
- 尚不能宣称 PyTorch CUDA 路径可用，因为对应打印在异常前未执行。

解释/推测：
- 安装单个 NiBabel 依赖后，环境大概率能够继续进入代码入口检查。

下一步建议：
- 使用当前 Python 执行 `python -m pip install --no-cache-dir nibabel==5.4.2`。
- 重新运行依赖和 CUDA 检查；通过后再做默认 96^3 短预检。

### 17. 供下一位分析者重点判断的问题
- 安装 NiBabel 后是否仍有其他缺失依赖。
- `torch.__version__` 与 `torch.version.cuda` 是否符合所选镜像。
- 实际 CUDA 张量运算、三个入口导入和 `--help` 是否成功。

## EXP-0021：NiBabel 安装成功与缺失 Matplotlib 诊断

### 1. 本轮实验编号与时间
- 实验编号：EXP-0021。
- 时间：2026-08-17 08:21:28 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 补齐 NiBabel 后重新执行依赖/CUDA检查和三个训练入口启动检查，识别剩余阻塞。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0020；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- 三个训练入口都在文件顶部导入 `matplotlib`，因此该依赖缺失会在参数解析和模型运行前统一失败。
- 单一共享环境中安装一次 Matplotlib 即可供三个入口使用。

解释/推测：
- 当前三个入口的相同失败由共享环境缺包导致，不是三个实现分别出现代码错误。

### 5. 实际修改文件
- 云端 Python 环境：安装 `nibabel==5.4.2`。
- 本地：仅更新 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- 未修改项目代码文件。
- `HANDOFF.md`：追加 NiBabel 安装结果与 Matplotlib 缺失报错。

### 7. 实际运行命令
- 云端已运行：`python -m pip install --no-cache-dir nibabel==5.4.2`。
- 云端已运行：依赖/CUDA Python 检查脚本。
- 云端已运行：三个训练入口的 `--help` 检查。
- 训练/评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；云端已确认 train/val/test = 40/8/3。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- GPU/驱动沿用 EXP-0020：RTX 4090 49140 MiB、driver 595.71.05。
- Python site-packages 路径显示 Python 3.12。
- 已有 NumPy 2.0.1、packaging 24.1、typing_extensions 4.11.0。
- NiBabel 5.4.2 安装成功。
- PyTorch 版本、编译 CUDA 和 CUDA availability 再次因 `matplotlib` 导入异常发生在打印前而未获得。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为依赖安装/入口诊断，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- NiBabel 安装成功；root pip 警告为通用环境管理警告，不是安装失败。
- 完整检查报错：`ModuleNotFoundError: No module named 'matplotlib'`。
- RG、精简 RA、RA v2 三个 `--help` 均因同一 `import matplotlib` 缺包而失败。
- 尚未进入训练，无数值异常。

### 15. 遗留问题
- 安装与本地验证环境一致的 `matplotlib==3.10.8`。
- 运行 `pip check`，再执行版本/CUDA张量和三个入口检查。
- 检查通过后执行默认 96^3 CUDA 短预检。

### 16. 本轮事实性结论
事实：
- NiBabel 依赖已补齐。
- 当前唯一已观测到的下一层阻塞是缺失 Matplotlib。
- 三个入口失败原因完全相同，尚无证据表明是模型代码问题。

解释/推测：
- 安装 Matplotlib 后，三个入口预计可继续启动；是否还有更深层依赖问题仍需实际复查。

下一步建议：
- 执行 `python -m pip install --no-cache-dir matplotlib==3.10.8`，然后运行 `python -m pip check`。
- 重新执行 CUDA 张量和三个 `--help` 检查。

### 17. 供下一位分析者重点判断的问题
- Matplotlib 安装后 `pip check` 是否无冲突。
- PyTorch/编译 CUDA/availability 的真实值。
- 三个入口是否终于能够完成参数解析并正常退出。

## EXP-0022：算家云 CUDA 环境与三个训练入口全部通过

### 1. 本轮实验编号与时间
- 实验编号：EXP-0022。
- 时间：2026-08-17 08:24:33 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 安装 Matplotlib 后复查完整 Python/CUDA 环境、实际 CUDA 张量计算和三个训练入口。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0021；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 模型、训练脚本和数据未修改。

### 4. 理论依据
事实：
- `torch.cuda.is_available()` 与实际在 `device='cuda'` 上创建/归约张量共同通过，才能确认 PyTorch CUDA 运行路径可用。
- 三个入口 `--help` 成功说明顶层导入、跨子目录导入和 argparse 解析在云端可运行，但不等于默认模型单 batch 已通过。

解释/推测：
- 当前环境已满足进入默认 96^3 单 batch GPU 预检的前置条件。

### 5. 实际修改文件
- 云端 Python 环境：安装 `matplotlib==3.10.8`。
- 本地：仅更新 `HANDOFF.md`。

### 6. 每个文件具体修改内容
- 未修改项目代码文件。
- `HANDOFF.md`：追加云端环境实测版本、CUDA计算和三个入口结果。

### 7. 实际运行命令
- 云端已运行：安装 Matplotlib。
- 云端已运行：Python/PyTorch/CUDA/NumPy/NiBabel/Matplotlib 版本与 CUDA 张量检查。
- 云端已运行：RG、精简 RA、RA v2 三个训练入口 `--help`。
- `pip check` 结果：截图未显示，未获得。
- 训练/评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 未改变；云端已确认 train/val/test = 40/8/3。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未改变；正式默认仍为 seed 42、100 epochs、batch size 1、learning rate 1e-4、AdamW、AMP、96^3 patch。

### 10. GPU、CUDA、Python、PyTorch
- Python：3.12.7（Anaconda）。
- PyTorch：2.5.1。
- PyTorch CUDA runtime：12.4。
- `torch.cuda.is_available()`：True。
- GPU：NVIDIA GeForce RTX 4090。
- PyTorch 识别总显存：47.37GB。
- NumPy：2.0.1。
- NiBabel：5.4.2。
- Matplotlib：3.10.8。
- CUDA 1024x1024 随机张量归约成功，结果为有限数值 1675.8480224609375。

### 11. 最新真实测试结果：PSNR、SSIM、MAE及项目实际使用的其他指标
- 本轮未训练/评估，新指标未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。

### 13. 是否严格可比；不可比时写明原因
- 本轮为环境与入口检查，不涉及模型结果比较。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 本轮完整环境检查无报错。
- RG 输出 `RG OK`，精简 RA 输出 `Slim RA OK`，RA v2 输出 `RA v2 OK`。
- `pip check` 是否通过未从截图确认。
- 尚未进行默认 96^3 模型 forward/backward，无训练数值结论。

### 15. 遗留问题
- 执行 RG 默认 96^3、batch 1、AMP 的单个真实 batch optimizer step，并记录峰值 allocated/reserved 显存、耗时、loss、梯度和 skipped step。
- RG 通过后分别执行精简 RA 与 RA v2 同规格单 batch 预检。
- 云端 `HANDOFF.md` 来自较早归档，后续需要同步本地最新记录。

### 16. 本轮事实性结论
事实：
- 所选镜像、驱动和 RTX 4090 48GB 能完成 PyTorch CUDA 张量运算。
- 三个训练入口的导入和参数解析全部成功。
- 当前没有依赖层阻塞。

解释/推测：
- 环境已具备执行三组默认配置 GPU 预检的条件，但仍不能保证 96^3 显存充足或 loss/gradient 有限。

下一步建议：
- 先运行 RG 单真实 batch 默认配置预检，不保存 checkpoint、不进入验证/测试。
- 根据 RG 实测结果再继续精简 RA 和 v2，避免同时启动长任务。

### 17. 供下一位分析者重点判断的问题
- RG 默认单 batch 的峰值显存是否显著低于 47.37GB。
- AMP 是否发生 skipped optimizer step，loss 和 grad norm 是否有限。
- 精简 RA 和 v2 的额外 latent/refiner 路径是否增加显存或引入数值问题。

## EXP-0023：算家云 RG-ReCL 默认 96^3 单真实 batch 训练预检

### 1. 本轮实验编号与时间
- 实验编号：EXP-0023。
- 时间：2026-08-17 08:30:17 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 在云端 RTX 4090 48GB 上，以 RG-ReCL 默认核心配置完成一个真实 NIfTI 训练样本的 forward、backward 和 optimizer step，并检查显存、耗时、loss、梯度、AMP 与依赖状态。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0022；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 预检模型为 `RG-ReCL/model_attnres3d_rgrecl.py` 中的 RG-ReCL；本轮未修改模型或训练代码。

### 4. 理论依据
事实：
- 单真实 batch 同时覆盖 NIfTI 数据发现与加载、96^3 patch、CUDA forward、重建损失、Region PatchNCE、AMP backward、梯度裁剪和 AdamW 参数更新。
- `optimizer_steps=1`、`skipped_optimizer_steps=0` 且 loss/grad norm 有限，是正式长训练前必要的数值与执行链路检查。

解释/推测：
- 单 batch 通过能证明当前配置可执行，但不能证明 100 epoch 稳定、模型有效或最终指标优于 E0。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`。
- 云端项目代码与数据未修改。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 RG-ReCL 云端单真实 batch 的命令配置、数值结果、显存、依赖告警和后续判断问题。

### 7. 实际运行命令
- 云端已运行：`cd /root/sj-tmp/ctri`。
- 云端已运行：`python -m pip check`。
- 云端已运行：用户提供的 Python here-document，导入 `RG-ReCL/model_attnres3d_rgrecl.py`，使用一个真实训练病例执行一次 `train_one_epoch()`。
- 正式 100 epoch 训练：未运行。
- 验证与测试：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录：`/root/sj-tmp/ctri/data/dataset`。
- 本轮实际读取：`data/dataset/train` 中 `discover_cases()` 返回数据集的第 1 个病例，`Subset(dataset, [0])`。
- 云端此前已确认 train/val/test 病例目录数为 40/8/3，与 E0 基线一致。
- 本轮只用一个训练病例进行执行链路预检，不构成可与 E0 比较的训练或评估。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- seed：42。
- epoch：仅调用一次 `train_one_epoch()`，DataLoader 只有 1 个 batch；不等同于正式 1 epoch。
- batch size：1。
- patch size：96 x 96 x 96。
- learning rate：1e-4。
- optimizer：AdamW，其他未显式参数沿用 PyTorch 默认值。
- AMP：开启；初始/结果 `amp_scale=65536.0`。
- 梯度裁剪：`max_grad_norm=5.0`。
- 重建损失：L1 0.45、SSIM 0.30、Edge 0.15、Frequency 0.10，frequency alpha 1.0。
- RG-ReCL：开启；lambda 0.05、temperature 0.07、hard ratio 0.20、feature dim 64、num patches 256、negative pool 1024。
- 模型：base channels 32、bottleneck blocks 6、dropout 0.2、sigmoid 输出。

### 10. GPU、CUDA、Python、PyTorch
- GPU：NVIDIA GeForce RTX 4090，PyTorch 可见显存约 47.37GB；沿用 EXP-0022。
- Python：3.12.7；PyTorch：2.5.1；PyTorch CUDA runtime：12.4；`torch.cuda.is_available()` 为 True；沿用 EXP-0022。
- 本轮峰值 allocated 显存：1.75GB。
- 本轮峰值 reserved 显存：2.09GB。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮为单训练 batch 预检，PSNR：未获得；SSIM：未获得；MAE：未获得；HFEN：未获得；Gradient MAE：未获得。
- 单 batch 总 loss：0.9106237888336182。
- reconstruction loss：0.45923343300819397。
- 未加权 RG-ReCL loss：9.027806282043457。
- 加权 RG-ReCL loss：0.4513903260231018。
- grad norm：2.439018487930298。
- optimizer steps：1；skipped optimizer steps：0。
- 耗时：1.39 秒。
- 终止标记：`RG_DEFAULT_96_SINGLE_BATCH_OK`。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。本轮没有验证或测试指标，禁止与 E0 计算效果差值。

### 13. 是否严格可比；不可比时写明原因
- 不可比。本轮仅使用一个训练病例进行一次参数更新，没有完成相同 epoch、checkpoint 选择、验证和测试流程。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 模型链路无报错；所有记录的 loss 与 grad norm 为有限值，无 NaN/Inf。
- AMP 未跳过 optimizer step。
- `pip check` 报告两项环境不一致：PyTorch 2.5.1 需要 `fsspec` 但当前未安装；PyTorch 2.5.1 要求 `sympy==1.13.1`，当前为 1.13.2。
- 加权 RG-ReCL loss 0.451390 与 reconstruction loss 0.459233 量级几乎相同；这不是数值错误，但说明固定 lambda 0.05 在随机初始化首 batch 中并非弱辅助项。

### 15. 遗留问题
- 在正式训练前修复 `fsspec` 缺失和 SymPy 版本不匹配，并重新执行 `python -m pip check`。
- 按同规格分别完成精简 RA-ReCL 与 RA-ReCL v2 单真实 batch 预检。
- 正式 RG 训练前决定是否严格保持指定 lambda 0.05；若保持，必须记录前若干 epoch 的 `rec_loss` 与 `rgrecl_weighted_loss` 比例及 PSNR/SSIM 走势。

### 16. 本轮事实性结论
事实：
- RG-ReCL 默认 96^3、batch 1、AMP 配置在当前 RTX 4090 48GB 上完成了一个真实训练 batch 和一次有效 optimizer step。
- 峰值 reserved 显存仅 2.09GB，当前配置不存在单 batch OOM 问题。
- 当前没有 NaN、Inf、AMP skipped step 或梯度爆炸证据。
- 当前 Python 环境的 `pip check` 未通过，存在 `fsspec` 缺失与 SymPy 版本不匹配。

解释/推测：
- 显存余量足够支持当前 RG 单卡 batch 1 正式训练；但长期稳定性和最终重建收益仍需正式训练与统一测试确认。
- 首 batch 中辅助项约占总 loss 的 49.6%，可能显著影响早期优化方向；是否有害目前没有指标证据。

下一步建议：
- 先修复两项依赖并确认 `pip check` 无冲突。
- 随后运行精简 RA-ReCL 和 RA-ReCL v2 的同规格单真实 batch 预检，再启动顺序正式训练。

### 17. 供下一位分析者重点判断的问题
- 修复依赖后 `pip check` 是否完全通过。
- 精简 RA-ReCL 与 RA-ReCL v2 是否也能完成 optimizer step，显存与 loss 比例分别是多少。
- RG-ReCL 正式训练前几轮中，加权辅助损失是否持续接近或高于 reconstruction loss，以及这种比例是否导致验证 PSNR/SSIM 恶化。

## EXP-0024：算家云 PyTorch 依赖一致性修复

### 1. 本轮实验编号与时间
- 实验编号：EXP-0024。
- 时间：2026-08-17 08:33:15 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 修复 EXP-0023 中 `pip check` 报告的 `fsspec` 缺失与 SymPy 版本不匹配，并重新验证 Python 环境依赖一致性。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0023；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 项目模型和代码未修改。

### 4. 理论依据
事实：
- PyTorch 2.5.1 的当前安装元数据要求存在 `fsspec`，并在 Python >= 3.9 时要求 `sympy==1.13.1`。
- `python -m pip check` 可检查已安装包声明的依赖是否缺失或版本冲突。

解释/推测：
- 补齐声明依赖可降低长训练过程中延迟触发模块导入或符号运算兼容问题的风险。

### 5. 实际修改文件
- 云端 Python 环境：安装 `fsspec==2026.7.0`，将 `sympy` 从 1.13.2 降为 1.13.1。
- 本地 `HANDOFF.md`。
- 项目代码与数据文件未修改。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加依赖安装版本、pip 检查结果和后续预检要求。
- 云端 site-packages：新增 fsspec 2026.7.0；替换 SymPy 1.13.2 为 1.13.1。

### 7. 实际运行命令
- 云端已运行：`cd /root/sj-tmp/ctri`。
- 云端已运行：`python -m pip install --no-cache-dir fsspec "sympy==1.13.1"`。
- 云端已运行：`python -m pip check`。
- 训练与评估：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 本轮未读取或修改数据。
- 沿用已确认的 train/val/test = 40/8/3，与 E0 基线一致。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 本轮未运行训练，均不适用；正式默认配置未改变。

### 10. GPU、CUDA、Python、PyTorch
- 沿用 EXP-0023：RTX 4090 48GB、Python 3.12.7、PyTorch 2.5.1、PyTorch CUDA 12.4。
- 本轮新增确认：fsspec 2026.7.0、SymPy 1.13.1。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未训练或评估；PSNR、SSIM、MAE、HFEN、Gradient MAE 均未获得。
- `pip check` 真实结果：`No broken requirements found.`。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。本轮为环境修复，不涉及模型结果比较。

### 13. 是否严格可比；不可比时写明原因
- 不可比。本轮未执行训练、验证或测试。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 安装成功，无依赖冲突。
- pip 输出 root 用户安装警告；这是环境管理警告，不是安装失败或项目运行报错。
- 未运行训练，因此没有新的训练数值异常结论。

### 15. 遗留问题
- 执行精简 RA-ReCL 默认 96^3、batch 1、AMP 的单真实 batch 预检。
- 随后执行 RA-ReCL v2 同规格预检。
- 两者均通过后再按 RG、精简 RA、RA v2 的顺序启动正式训练。

### 16. 本轮事实性结论
事实：
- fsspec 2026.7.0 与 SymPy 1.13.1 安装成功。
- 当前 `python -m pip check` 已完全通过。
- 本轮没有修改模型代码、数据划分或实验超参数。

解释/推测：
- 当前没有已知 Python 包声明层面的依赖阻塞，可以继续模型单 batch 预检。

下一步建议：
- 运行精简 RA-ReCL 单真实 batch，并记录总 loss、重建 loss、latent/region 辅助项、梯度、AMP step、耗时和峰值显存。

### 17. 供下一位分析者重点判断的问题
- 精简 RA-ReCL 的两个加权辅助项相对 reconstruction loss 的比例是否过高。
- 精简 RA-ReCL 是否无 NaN/Inf、无 AMP skipped step，并保持足够显存余量。
- fsspec 2026.7.0 虽满足 PyTorch 声明，后续训练是否出现任何实际兼容异常。

## EXP-0025：三条训练线 checkpoint 与输出目录保存策略审计

### 1. 本轮实验编号与时间
- 实验编号：EXP-0025。
- 时间：2026-08-17 08:36:15 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 确认单 batch 预检是否保存文件，以及 RG-ReCL、精简 RA-ReCL、RA-ReCL v2 正式训练的 checkpoint 周期、最佳模型、日志、测试结果和图像目录。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0024；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 本轮为只读代码审计，模型与训练代码未修改。

### 4. 理论依据
事实：
- 三个训练入口都在每个 epoch 结束后覆盖保存 `latest.pth`，并按默认 `save_every=5` 保存 `epoch_XXXX.pth`。
- 验证指标刷新时分别保存 SSIM、PSNR、HFEN 最优模型；测试默认加载 `best.pth`。
- inline 单 batch 预检只调用模型、loss、optimizer 和 `train_one_epoch()`，没有调用 `save_checkpoint()` 或写日志函数。

解释/推测：
- 每轮 `latest.pth` 降低意外中断损失；每 5 轮不可覆盖断点用于回退与复现实验阶段。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`。
- 模型、训练脚本、云端环境和数据均未修改。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：记录三条训练线的真实 checkpoint 和输出目录策略。

### 7. 实际运行命令
- 本地已运行 `rg` 与只读 `Get-Content`，检查三个训练入口和 `save_checkpoint()` 实现。
- 训练、评估、云端命令：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 本轮未读取或修改数据；沿用 train/val/test = 40/8/3，与 E0 一致。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- 未运行训练。
- 保存策略默认值：`save_every=5`；每轮保存/覆盖 `latest.pth`。

### 10. GPU、CUDA、Python、PyTorch
- 本轮未使用 GPU；云端环境沿用 EXP-0024。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮未训练或评估；新指标均未获得。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。本轮不涉及结果比较。

### 13. 是否严格可比；不可比时写明原因
- 不可比。本轮为保存逻辑审计。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 无运行异常。
- 发现一个需要通过启动命令规避的路径语义：三个默认 `save_dir` 均相对于启动时当前工作目录，而非脚本所在目录。

### 15. 遗留问题
- 正式训练命令必须显式使用三个互不重叠的绝对 `--save_dir`，确保结果处于数据盘并按实验隔离。
- 完成精简 RA 与 RA v2 单 batch 预检。

### 16. 本轮事实性结论
事实：
- 单 batch inline 预检不保存 checkpoint、日志、测试结果或图像。
- 正式训练每轮覆盖 `checkpoints/latest.pth`，默认每 5 轮另存 `checkpoints/epoch_XXXX.pth`。
- 正式训练还会保存 `best.pth`、`best_ssim.pth`、`best_psnr.pth`、`best_hfen.pth`；checkpoint 包含模型、optimizer、epoch、参数、最佳指标，并按可用情况包含 scheduler、AMP scaler、global step 与随机状态。
- `log.csv`、`test_results.csv`、不确定性 CSV/NIfTI 和测试图均写在所选 `save_dir` 下。
- 默认目录分别为启动目录下 `output`、`output_rarecl`、`output_rareclv2`。

解释/推测：
- 为满足每个实验结果放在自身目录且防止从错误工作目录启动，正式命令应使用源码目录内的绝对输出路径。

下一步建议：
- 先运行精简 RA 单 batch 预检；通过后测试 RA v2。
- 正式训练时分别指定 `/root/sj-tmp/ctri/RG-ReCL/output_rgrecl`、`/root/sj-tmp/ctri/jingjian RA-ReCL/output_rarecl`、`/root/sj-tmp/ctri/zuixinwanzheng RA-ReCL v2/output_rareclv2`。

### 17. 供下一位分析者重点判断的问题
- 精简 RA 与 RA v2 单 batch 是否通过且不会产生意外文件。
- 正式训练启动后 `latest.pth`、编号断点、最佳模型和 CSV 是否均出现在显式绝对 `save_dir` 下。
- 云实例中断前是否需要把输出目录增量同步到外部持久存储。

## EXP-0026：算家云精简 RA-ReCL 默认 96^3 单真实 batch 训练预检

### 1. 本轮实验编号与时间
- 实验编号：EXP-0026。
- 时间：2026-08-17 08:40:04 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 在云端 RTX 4090 48GB 上，以精简 RA-ReCL 默认核心配置完成一个真实 NIfTI 训练样本的 forward、backward 和 optimizer step，并检查两个辅助项、显存、耗时、梯度与 AMP。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0025；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 预检模型为 `jingjian RA-ReCL/model_attnres3d_ramedreclpp.py`；本轮未修改模型或训练代码。

### 4. 理论依据
事实：
- 该预检同时覆盖预测/目标 latent alignment、Region PatchNCE、重建损失、AMP backward、梯度裁剪和 AdamW 参数更新。
- 有限 loss/grad norm、一次成功 optimizer step 和零 skipped step 是正式长训练前必要检查。

解释/推测：
- 单 batch 通过不能证明长期稳定或最终指标提升，但可以排除当前配置的直接执行、OOM 和首步非有限数值问题。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`。
- 云端项目代码、数据与模型文件未修改；inline 预检不保存 checkpoint。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加精简 RA 云端单真实 batch 的配置、数值、显存和辅助损失比例。

### 7. 实际运行命令
- 云端已运行 Python here-document：导入 `jingjian RA-ReCL/model_attnres3d_ramedreclpp.py`，使用一个真实训练病例执行一次 `train_one_epoch()`。
- 正式训练、验证与测试：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录：`/root/sj-tmp/ctri/data/dataset`。
- 实际读取 train split 中 `Subset(dataset, [0])` 的一个病例。
- 已确认完整 train/val/test = 40/8/3，与 E0 基线一致；本轮单病例预检不可与 E0 指标比较。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- seed 42；仅 1 个 batch；batch size 1；patch 96 x 96 x 96；learning rate 1e-4；AdamW。
- AMP 开启；amp scale 65536；梯度裁剪 5.0。
- 重建损失：L1 0.45、SSIM 0.30、Edge 0.15、Frequency 0.10，frequency alpha 1.0。
- 精简 RA：lambda latent 0.05、lambda region 0.05、temperature 0.07、hard ratio 0.20、feature dim 64、latent samples 256、region patches 256、negative pool 1024。
- 模型：base channels 32、bottleneck blocks 6、dropout 0.2、sigmoid 输出。

### 10. GPU、CUDA、Python、PyTorch
- 沿用 EXP-0024：RTX 4090 48GB、Python 3.12.7、PyTorch 2.5.1、PyTorch CUDA 12.4。
- 峰值 allocated 显存 1.75GB；峰值 reserved 显存 2.09GB。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮为单训练 batch 预检；PSNR、SSIM、MAE、HFEN、Gradient MAE 均未获得。
- 总 loss：0.9499029517173767；reconstruction loss：0.45923343300819397。
- latent loss：0.7400932312011719；加权 latent loss：0.037004660815000534。
- region loss：9.073297500610352；加权 region loss：0.4536648690700531。
- hard region fraction：0.20000693698724112；sampled hard fraction：0.2578125。
- latent similarity：0.259906808535258；region positive similarity：0.25018618007500965。
- grad norm：2.512227773666382；optimizer steps 1；skipped steps 0；amp scale 65536.0。
- 耗时 1.24 秒；终止标记：`SLIM_RA_DEFAULT_96_SINGLE_BATCH_OK`。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。本轮没有正式验证或测试指标。

### 13. 是否严格可比；不可比时写明原因
- 不可比。本轮只使用一个训练病例进行一次参数更新，没有完成统一训练、checkpoint 选择与测试。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 无模型报错；所有记录 loss 和 grad norm 有限，无 NaN/Inf；AMP 未跳过 optimizer step。
- 终端回显中出现 `PYint(...)` 一类粘贴显示残片，但 here-document 实际完成且断言通过，不构成 Python 执行失败。
- 两个加权辅助项之和为 0.49066952988505363，约为 reconstruction loss 的 106.84%，约占总 loss 的 51.65%。
- 其中加权 region loss 0.453665 接近全部 reconstruction loss；latent 加权项仅 0.037005。

### 15. 遗留问题
- 执行 RA-ReCL v2 默认 96^3、batch 1、AMP 的同规格单真实 batch 预检。
- 三条线全部通过后再启动正式 RG 训练。
- 正式精简 RA 训练时重点观察 region 加权项与 reconstruction loss 比例及验证 PSNR/SSIM；目前不根据单 batch 擅自改权重。

### 16. 本轮事实性结论
事实：
- 精简 RA-ReCL 在当前 RTX 4090 上完成一个真实 96^3 batch 和一次有效 optimizer step。
- 峰值 reserved 显存 2.09GB，无单 batch OOM、NaN/Inf、AMP skipped step 或梯度爆炸证据。
- 首 batch 中辅助项总和略高于重建损失，主要由 Region PatchNCE 贡献。

解释/推测：
- 当前显存余量足够；辅助损失量级可能显著影响早期训练方向，但是否损害或改善重建尚无验证指标证据。

下一步建议：
- 运行 RA-ReCL v2 单真实 batch 预检，记录 curriculum 阶段、有效 hard ratio、refiner delta、loss、梯度和显存。

### 17. 供下一位分析者重点判断的问题
- RA-ReCL v2 的 warmup 阶段是否按设计使 PatchNCE 加权项为零或足够弱。
- residual refiner 首步输出修正量是否有限且梯度稳定。
- 精简 RA 正式训练时 Region PatchNCE 是否长期压过重建目标并导致验证指标下降。

## EXP-0027：算家云 RA-ReCL v2 warmup 与中期阶段真实 batch 预检

### 1. 本轮实验编号与时间
- 实验编号：EXP-0027。
- 时间：2026-08-17 08:42:35 +08:00（Asia/Shanghai）。

### 2. 本轮目标
- 分别模拟 RA-ReCL v2 curriculum 的 warmup 与 middle stage，用真实 96^3 NIfTI batch 检查 residual refiner、多尺度 Region PatchNCE、AMP backward、optimizer step、显存和数值稳定性。

### 3. 修改前基线版本 / 模型 / commit
- 沿用 EXP-0026；本地 HEAD 为 `ef0141b731675b477fee2c50f4a7ea61d1823b03`。
- 预检模型为 `zuixinwanzheng RA-ReCL v2/model_attnres3d_rareclv2.py`；未修改代码。

### 4. 理论依据
事实：
- v2 在 progress < 0.20 时关闭 PatchNCE，先训练重建与 residual refiner；0.20 <= progress < 0.60 时启用 hard ratio 0.30 的多尺度 Region PatchNCE。
- 分别模拟两个阶段才能覆盖 v2 的 warmup 分支和完整辅助损失分支。

解释/推测：
- warmup 可避免随机初始化 PatchNCE 在训练最初阶段立即主导优化，但最终是否改善重建需正式验证指标确认。

### 5. 实际修改文件
- 仅本地 `HANDOFF.md`；云端代码、数据与模型文件未修改。
- inline 预检未保存 checkpoint、CSV 或图像。

### 6. 每个文件具体修改内容
- `HANDOFF.md`：追加 v2 两阶段预检配置、loss、curriculum、refiner delta、梯度、耗时和显存。

### 7. 实际运行命令
- 云端已运行 Python here-document，使用同一真实训练病例依次调用两次 `train_one_epoch()`：warmup progress 0.0；middle progress 0.5。
- 正式训练、验证与测试：未运行。

### 8. 数据集 / 数据划分，以及是否与基线一致
- 数据根目录 `/root/sj-tmp/ctri/data/dataset`；实际使用 train split 的 `Subset(dataset, [0])`。
- 完整 split 仍为 40/8/3，与 E0 基线一致；本轮不可用于效果比较。

### 9. seed、epoch、batch size、learning rate、optimizer、loss 和关键开关
- seed 42；两个单 batch optimizer step；batch size 1；patch 96 x 96 x 96；lr 1e-4；AdamW；AMP；梯度裁剪 5.0。
- 重建损失：L1 0.45、SSIM 0.30、Edge 0.15、Frequency 0.10，frequency alpha 1.0。
- v2：lambda 0.05、temperature 0.07、feature dim 64、patches 256、negative pool 1024、level weights 0.20/0.30/0.50。
- curriculum：warmup end 0.20、middle end 0.60、middle hard ratio 0.30、final hard ratio 0.20。
- refiner：开启，channels 8，residual scale 0.10。
- 模型：base channels 32、bottleneck blocks 6、dropout 0.2、sigmoid 输出。

### 10. GPU、CUDA、Python、PyTorch
- 沿用 EXP-0024：RTX 4090 48GB、Python 3.12.7、PyTorch 2.5.1、PyTorch CUDA 12.4。
- warmup 峰值 allocated/reserved：1.58GB/1.93GB。
- middle 峰值 allocated/reserved：2.01GB/2.55GB。

### 11. 最新真实测试结果：PSNR、SSIM、MAE 及项目实际使用的其他指标
- 本轮为训练预检，正式 PSNR、SSIM、MAE、HFEN、Gradient MAE 均未获得。
- warmup：total/rec loss 0.45923343300819397/0.45923343300819397；PatchNCE 0；effective lambda 0；stage 0；hard ratio 0；active 0；refinement mean/max delta 0/0；grad norm 0.6392918825149536；step 1、skipped 0；耗时 1.15 秒。
- middle：total/rec loss 0.8488994836807251/0.41074374318122864；PatchNCE 8.763114929199219；weighted PatchNCE 0.43815574049949646；effective lambda 0.05；stage 1；hard ratio 0.3；active 1。
- middle hard region fraction 0.300020565589269；sampled hard fraction 0.4075520833333333；positive similarity 0.2807150185108185。
- middle refinement mean/max delta 0.00024437904357910156/0.00048828125；grad norm 1.8847787380218506；step 1、skipped 0；耗时 0.83 秒。
- 两阶段 amp scale 均为 65536.0；终止标记 `RARECLV2_WARMUP_AND_MIDDLE_96_OK`。
- 最新正式可比结果仍为 E0：MAE 0.107510、PSNR 16.068155 dB、SSIM 0.529396、HFEN 0.730762、Gradient MAE 0.074969。

### 12. 与可比基线的差值
- 未获得。本轮没有正式测试结果。

### 13. 是否严格可比；不可比时写明原因
- 不可比。本轮只模拟两个单 batch 训练阶段，没有完整训练、验证选模和统一测试。

### 14. 训练 / 评估异常、失败实验、报错或数值异常
- 两阶段均完成有效 optimizer step，零 AMP skipped step，loss 与梯度有限，无 OOM、NaN/Inf 或梯度爆炸证据。
- warmup 首次 forward 的 refinement delta 为 0；refiner 输出层为零初始化，第一步更新后 middle forward 的 delta 变为有限非零值，因此该现象符合当前初始化与执行顺序。
- middle weighted PatchNCE 约为 reconstruction loss 的 106.67%，约占 total loss 的 51.61%；量级偏高但尚无验证指标证明有害。
- 终端 `PYint(...)` 为粘贴回显残片，断言与终止标记均通过，不是执行失败。

### 15. 遗留问题
- 三条实验线预检均已完成；下一步正式运行 RG-ReCL。
- 正式训练使用显式绝对 `save_dir`，每轮检查 `latest.pth` 与 `log.csv`。
- RG 无 warmup 且首 batch 加权辅助项接近重建损失，建议至少观察前 3 至 5 个 epoch 的验证指标再决定是否继续完整 100 epoch。

### 16. 本轮事实性结论
事实：
- RA-ReCL v2 warmup、residual refiner 和 middle-stage 多尺度 Region PatchNCE 均在真实 96^3 batch 上完成训练更新。
- curriculum stage、effective lambda 和 hard ratio 与设计一致。
- 当前 GPU 显存余量充分，未观察到数值或执行故障。
- 三条新实验线的云端核心训练路径现均已通过预检。

解释/推测：
- v2 warmup 能隔离最初阶段的高量级 PatchNCE；是否因此优于 RG/精简 RA 必须由严格可比的验证与测试结果决定。

下一步建议：
- 启动 RG-ReCL 正式训练，输出到 `/root/sj-tmp/ctri/RG-ReCL/output_rgrecl`，先观察前 3 至 5 epoch 的 loss 比例和验证 PSNR/SSIM/HFEN。

### 17. 供下一位分析者重点判断的问题
- RG 前 3 至 5 epoch 的加权辅助项是否下降到明显低于 reconstruction loss。
- RG 验证 PSNR/SSIM 是否至少不低于 E0 同 epoch/同口径表现。
- `latest.pth`、周期断点、最佳模型和日志是否全部写入显式数据盘目录，断点续训是否可用。

## EXP-0028 - 三线完整100 epoch正式对比（RG-ReCL -> 精简 RA-ReCL -> RA-ReCL v2）
### 1. 本轮实验编号与时间
- 编号：EXP-0028
- 时间：2026-08-17（基于云端 `root/sj-tmp/ctri` 跑 100 epochs 结果统计）

### 2. 本轮目标
- 目标：对 RG-ReCL、精简 RA-ReCL、RA-ReCL v2 三条新方法在统一分割/统一口径下做正式对比，判断是否相对 E0 有可复现提升。
- 目标指标：PSNR / SSIM / MAE / HFEN / Gradient MAE（同时查看 val 与 test）
- 主结论目标：确认是否“严格可比且多指标同时不退化”。

### 3. 修改前基线版本/模型/commit（能确定则记录）
- 基线：`output_gt_e0_rec_40_8_3_e100`
- 基线结果文件：`output_gt_e0_rec_40_8_3_e100/E0_test_results.csv`
- Git：当前仓库 `HEAD=ef0141b`（未在本轮提交新代码）
- 训练代码基线：E0（`train_gtmedreclpp.py` / `model_attnres3d_gtmedreclpp.py`）

### 4. 理论依据
- RG-ReCL：对齐 prediction 与 GT 表征以补强结构一致性。
- 精简 RA-ReCL：在 RG 的基础上，显式加 prediction/GT latent 对齐 + region patch hard mining，降低随机负样本噪声。
- RA-ReCL v2：在前两者上增加 curriculum 与残差细化器，期望先学稳基础重建，再重点纠正困难区域的多尺度结构。

### 5. 实际修改文件
- 本轮只新增 `HANDOFF.md` 条目（实验结果归档），未改动三条训练代码文件。
- 结果输入文件（已读取）：  
  - `RG-ReCL/output_rgrecl/test_results.csv`  
  - `jingjian RA-ReCL/output_rarecl/test_results.csv`  
  - `zuixinwanzheng RA-ReCL v2/output_rareclv2/test_results.csv`

### 6. 每个文件具体修改内容
- `RG-ReCL/output_rgrecl/test_results.csv`：记录 test MAE/PSNR/SSIM/HFEN/Gradient MAE（RG 结果）
- `jingjian RA-ReCL/output_rarecl/test_results.csv`：记录 test MAE/PSNR/SSIM/HFEN/Gradient MAE（精简 RA 结果）
- `zuixinwanzheng RA-ReCL v2/output_rareclv2/test_results.csv`：记录 test MAE/PSNR/SSIM/HFEN/Gradient MAE（RA v2 结果）
- 训练日志：三条 `log.csv`、`train_console.log`、`train.pid`、`uncertainty_results.csv` 均为本轮依据输入

### 7. 实际运行命令
- RG-ReCL（100 epoch）：
```bash
nohup python -u "RG-ReCL/train_rgrecl.py" --data_dir "/root/sj-tmp/ctri/data/dataset" --save_dir "/root/sj-tmp/ctri/RG-ReCL/output_rgrecl" --epochs 100 --batch_size 1 --patch_size 96 96 96 --ct_norm clip01 --mri_norm clip01 --base_channels 32 --bottleneck_blocks 6 --dropout 0.2 --final_activation sigmoid --lr 1e-4 --scheduler cosine --min_lr 1e-6 --max_grad_norm 5.0 --l1_weight 0.45 --ssim_weight 0.30 --edge_weight 0.15 --frequency_weight 0.10 --frequency_alpha 1.0 --rgrecl_lambda 0.05 --rgrecl_temperature 0.07 --rgrecl_hard_ratio 0.20 --rgrecl_feature_dim 64 --rgrecl_num_patches 256 --rgrecl_negative_pool 1024 --seed 42 --device cuda --num_workers 0 --save_every 5 > "/root/sj-tmp/ctri/RG-ReCL/output_rgrecl/train_console.log" 2>&1 &
```
- 精简 RA-ReCL（100 epoch）：按同数据路径与统一训练参数，启动 `jingjian RA-ReCL/train_ramedreclpp.py`，输出目录 `/root/sj-tmp/ctri/jingjian RA-ReCL/output_rarecl`
- RA-ReCL v2（100 epoch）：按同数据路径与统一训练参数，启动 `zuixinwanzheng RA-ReCL v2/train_rareclv2.py`，输出目录 `/root/sj-tmp/ctri/zuixinwanzheng RA-ReCL v2/output_rareclv2`

### 8. 数据集/数据划分，以及是否与基线一致
- train = 40、val = 8、test = 3（三者一致）
- 与 E0 基线一致：是
- 运行数据目录：`/root/sj-tmp/ctri/data/dataset`
- 推理裁剪：`(150,150,150)`

### 9. seed、epoch、batch size、learning rate、optimizer、loss和关键开关
- `seed=42` / `epochs=100` / `batch_size=1` / `patch_size=96x96x96`  
- `base_channels=32` / `bottleneck_blocks=6` / `dropout=0.2` / `final_activation=sigmoid`  
- 优化器：`AdamW`，`lr=1e-4`，`scheduler=cosine`，`min_lr=1e-6`，`max_grad_norm=5.0`  
- 重建损失权重：`L1 0.45, MS-SSIM 0.30, Edge 0.15, Freq 0.10, freq_alpha 1.0`  
- RG：`lambda=0.05, temperature=0.07, hard_ratio=0.20, feature_dim=64, num_patches=256, negative_pool=1024`  
- 精简 RA-ReCL：`lambda_latent=0.05, lambda_region=0.05, temperature=0.07, hard_ratio=0.20, latent_samples=256, region_patches=256, negative_pool=1024`
- v2：`lambda=0.05, temperature=0.07, curriculum=(0.2,0.6), hard_ratio=(0.3,0.2), feature_dim=64, num_patches=256, negative_pool=1024, level_weights=0.2/0.3/0.5, refiner_channels=8, residual_scale=0.1`

### 10. GPU、CUDA、Python、PyTorch
- 训练与评估在云端完成；本地环境与云端环境不一致（本地常见为 CPU）。  
- 云端记录：RTX 4090 / PyTorch 2.5.1 / CUDA 12.4（与之前 EXP-0024 记录一致）

### 11. 最新真实测试结果（PSNR、SSIM、MAE、HFEN、Gradient MAE）
- E0 基线：MAE 0.107510 / PSNR 16.068155 / SSIM 0.529396 / HFEN 0.730762 / Gradient MAE 0.074969  
- RG-ReCL：MAE 0.108908 / PSNR 16.038064 / SSIM 0.516738 / HFEN 0.740797 / Gradient MAE 0.075903  
- 精简 RA-ReCL：MAE 0.108223 / PSNR 16.105350 / SSIM 0.520373 / HFEN 0.737152 / Gradient MAE 0.075171  
- RA-ReCL v2：MAE 0.103682 / PSNR 16.246737 / SSIM 0.486855 / HFEN 0.743232 / Gradient MAE 0.076468

### 12. 与可比基线的差值（当前-基线）
- RG-ReCL：MAE +0.001398；PSNR -0.030091；SSIM -0.012658；HFEN +0.010035；Gradient MAE +0.000934  
- 精简 RA-ReCL：MAE +0.000713；PSNR +0.037195；SSIM -0.009023；HFEN +0.006390；Gradient MAE +0.000202  
- RA-ReCL v2：MAE -0.003828；PSNR +0.178582；SSIM -0.042541；HFEN +0.012470；Gradient MAE +0.001499

### 13. 是否严格可比；不可比时原因
- 可比性：训练/测试划分一致、主干重建损失一致、seed/batch/lr/patch 一致，属于严格可比。  
- 小幅不可比点：v2 引入 residual refiner，参数量略增（+441），但不改变数据划分与优化主干，不足以成为主要可比性问题。

### 14. 训练/评估异常、失败实验、报错或数值异常
- 未见 OOM、NaN/Inf、优化器 skip step 异常；均有正常完成日志与测试输出。  
- 三线均输出 `uncertainty_results.csv` 并完成 Moment-vs-MC Dropout 方差对比。  
- v2 中 refiner warmup 阶段 `patchnce` 为 0，符合预设 curriculum 逻辑。

### 15. 遗留问题
- 无单一权重下实现“三者同时优于 E0”：
  - RG：未优于基线（all key worse）
  - 精简 RA：PSNR 有小幅提升，但 SSIM/HFEN 仍变差
  - v2：MAE/PSNR 最优，但 SSIM 明显回落、HFEN/Gradient MAE 变差
- 仍需确认是否允许牺牲 SSIM 以换取 PSNR、是否需做“结构优先”多目标 early-stop（如 val SSIM 与 val PSNR 加权）。

### 16. 本轮事实性结论
- 事实：三条方法均完成 100 epoch 并独立测试，且所有 `log.csv/test_results.csv` 可核验。  
- 事实：`RG-ReCL` 与 `精简 RA-ReCL` 在 SSIM 与 HFEN 上均低于 E0；`RA-ReCL v2` 在 MAE/PSNR 上优于 E0，但 SSIM 明显更低。  
- 解释：v2 的重建细节主导增强明显，但结构/纹理相关约束（SSIM、HFEN、gradient）可能被当前对比配置削弱。  
- 下一步建议：先固定一个“结构约束优先目标”（SSIM 与 HFEN 约束优先）再继续 ablation。

### 17. 供下一位分析者重点判断的问题
- 下一轮是否以 SSIM 为约束主指标（比如 val SSIM 不降、PSNR 允许提升），还是以 MAE/PSNR 作为主目标。  
- v2 中是否先降 `lambda` 或冻结 residual refiner 观察 SSIM 是否回升。  
- 是否保留 curriculum 的中期 0.3 hard ratio，是否应改为更小 hard_ratio。  
- 是否增加前缀对照：RA-ReCL-v2 仅开启 residual、仅开启 multi-scale、仅开启 curriculum 的三组消融，定位 SSIM 下降根因。
