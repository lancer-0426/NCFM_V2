## NCFM V3：完整分布对齐实验说明

### 一键复现实验（推荐）

在项目根目录下执行（已在 `NCFM_V3` 目录内时，可直接运行）：

```bash
bash NCFM_V3/run_ncfm_v3.sh
```

**期望输出：**
- 终端打印 4 组实验日志：`align=all`（完整消融）、`align=amplitude`、`align=phase`、`align=both`
- 在 `results_v3/` 下生成多份结果文件，其中包括：
  - `ncfm_v3_results_*.json / *.csv`：单次实验结果（MSE / MMD / NCFM 幅度 / 相位 / 复数对齐指标）
  - `ncfm_ablation_comparison_*.csv`：`align=amplitude / phase / both` 的消融对比表
  - `ncfm_additional_visualizations.png`：对齐方法的雷达图 + 相对性能热力图
- 数值层面：
  - 在 `magnitude_altered` 场景下，**NCFM 幅度对齐损失**应明显小于相位/复数对齐
  - 在 `phase_altered` 场景下，**NCFM 相位/复数对齐损失**应优于仅幅度对齐（百分比差异在 10%–200% 的量级都可以接受，侧重"方向正确"）

> 若多次运行结果有轻微波动，只要整体相对排序保持不变，即视为复现成功。

---

### 环境与依赖

建议使用 Python 3.8–3.10，对应的依赖版本在 `NCFM_V3/requirements.txt` 中已经固定：

- numpy==1.21.6  
- matplotlib==3.5.3  
- scikit-learn==1.0.2  
- pandas==1.3.5  
- argparse==1.4.0  
- tensorboard==2.10.1

安装方式（推荐在虚拟环境或 conda 环境中执行）：

```bash
pip install -r NCFM_V3/requirements.txt
```

本脚本内部已统一固定 `numpy` / `random` /（可选）`torch` 的随机种子，默认 `SEED = 42`，以减少指标波动带来的歧义。

---

### 数据准备

NCFM V3 实验支持两种数据集：

1. **合成数据**（默认）：`make_moons` + `make_blobs`，无需额外下载数据集
   - 源分布：`make_moons` 生成的双月分布
   - 目标分布 1：加入高斯噪声的幅度变换（`magnitude_altered`）
   - 目标分布 2：旋转 45° 的相位变换（`phase_altered`）
   - 目标分布 3：三簇高斯混合分布（`completely_different`）

2. **CIFAR-10 数据集**（可选）：使用 `--dataset=cifar10` 参数
   - 自动下载 CIFAR-10 数据集
   - 使用 PCA 降维提取特征
   - 可用于与 VGG-16 基线对比

运行脚本时会自动在内存中生成或加载上述数据，无需手动准备。

---

### 手动运行示例：训练 / 评估配置

脚本支持通过 `--align` 开关控制"幅度 vs 相位 vs 复数对齐"三种配置，核心参数如下：

- `--align`: `amplitude` | `phase` | `both` | `all`  
  - `amplitude`：仅对齐特征函数的幅度  
  - `phase`：仅对齐特征函数的相位  
  - `both`：对齐完整复数特征函数（幅度 + 相位）  
  - `all`：一次性跑完 `amplitude/phase/both` 三种配置做消融
- `--n_samples`: 每个分布采样点个数（默认 1000）  
- `--n_t_points`: 特征函数的 t 点数量（默认 100）  
- `--output_dir`: 结果输出目录（默认 `results_v3`）  
- `--save_csv`: 是否额外保存 CSV 结果
- `--dataset`: `synthetic` | `cifar10`（默认 `synthetic`）
  - `synthetic`：使用合成数据（make_moons）
  - `cifar10`：使用 CIFAR-10 数据集（需要 PyTorch）
- `--run_vgg_baseline`: 运行 VGG-16 基线进行对比（仅在使用 `--dataset=cifar10` 时有效）
- `--use_tensorboard` / `--no-use_tensorboard`: 启用/禁用 TensorBoard 日志（默认启用）
- `--log_dir`: TensorBoard 日志目录（默认 `logs/ncfm_v3`）

#### 基础使用示例

```bash
# 1. 跑完整的 NCFM 消融实验（推荐，包含对比表和可视化）
# 在项目根目录执行
python NCFM_V3/ncfm_complete_experiment_v3.py --align=all --save_csv --output_dir=results_v3

# 2. 仅幅度对齐
python NCFM_V3/ncfm_complete_experiment_v3.py --align=amplitude --save_csv --output_dir=results_v3

# 3. 仅相位对齐
python NCFM_V3/ncfm_complete_experiment_v3.py --align=phase --save_csv --output_dir=results_v3

# 4. 复数完整对齐
python NCFM_V3/ncfm_complete_experiment_v3.py --align=both --save_csv --output_dir=results_v3
```

#### VGG-16 基线对比使用示例（重要）

**完整对比实验（推荐）：**
```bash
# 运行完整的 NCFM 消融 + VGG-16 基线对比
python NCFM_V3/ncfm_complete_experiment_v3.py \
    --align=all \
    --dataset=cifar10 \
    --run_vgg_baseline \
    --save_csv \
    --output_dir=results_v3
```

**单个 NCFM 方法对比：**
```bash
# 仅运行 NCFM-both 方法与 VGG-16 对比
python NCFM_V3/ncfm_complete_experiment_v3.py \
    --align=both \
    --dataset=cifar10 \
    --run_vgg_baseline \
    --save_csv \
    --output_dir=results_v3
```

**说明：**
- 使用 `--run_vgg_baseline` 时，必须同时指定 `--dataset=cifar10`
- VGG-16 会训练 10 个 epoch（快速对比）
- 对比结果会保存到 `ncfm_vgg_comparison_*.csv` 文件
- 对比表格包含训练时长（秒和分钟）

**期望验证指标/对齐度量区间（仅供 sanity check）：**

- 同一场景下，MSE/MMD/NCFM 三类指标应当在 **同一数量级** 上（如 1e-4 至 1e-1 之间）
- 对"量身定制"的对齐方式（例如：幅度变换场景下的 NCFM 幅度对齐），对应指标应显著低于不匹配的对齐方式

---

### 消融实验与结果导出

当使用 `--align=all` 时，脚本会自动：

1. 分别以 `align=amplitude / phase / both` 运行 3 组实验；  
2. 对每一组实验下的 3 种场景（`magnitude_altered / phase_altered / completely_different`）计算：  
   - `mse` / `mmd` / `ncfm_amplitude` / `ncfm_phase` / `ncfm_complex`；  
3. 汇总成一张对比表 `ncfm_ablation_comparison_*.csv`，字段包括：  
   - `alignment_method` / `scenario` / 各种损失函数数值；  
4. 计算关键场景下的 **百分比差异（±%）**，用于量化"幅度 vs 相位"的优势。

> 这些结果文件可直接用于论文表格或 PPT，对齐后只需轻微排版。

---

### 日志与可视化

当前版本采用以下三类输出：

- **终端日志**：逐场景打印各损失函数数值，便于快速 eyeballing；  
- **TensorBoard 曲线**：自动记录所有指标到 TensorBoard（默认开启），可通过以下命令查看：
  ```bash
  tensorboard --logdir=logs/ncfm_v3
  ```
  然后在浏览器打开 `http://localhost:6006` 查看训练/验证曲线。  
  日志结构：`{alignment_method}/{scenario}/{metric}`，例如 `amplitude/phase_altered/ncfm_amplitude`。
- **静态可视化**：在 `results_v3/ncfm_additional_visualizations.png` 中给出：  
  - 不同对齐方法在三种场景下的雷达图（log-scale）；  
  - 同一结果集下多种度量的相对性能热力图（ranking）。

> **注意**：如需禁用 TensorBoard，使用 `--no-use_tensorboard` 参数。

---

### VGG-16 对齐基线对比

当使用 `--dataset=cifar10 --run_vgg_baseline` 时，脚本会：

1. 在 CIFAR-10 数据集上运行 NCFM 实验（使用 PCA 降维后的特征）
2. 运行 VGG-16 基线训练（10 个 epoch，用于快速对比）
3. 生成 `ncfm_vgg_comparison_*.csv` 对比表格，包含：
   - NCFM 不同对齐方法的损失指标（所有场景）
   - VGG-16 的测试准确率
   - **训练时长**（秒和分钟）
   - 训练轮数、随机种子等元信息

**对比表格字段说明：**

**简化版表格（终端打印）：**
- `方法 (Method)`: VGG-16 或 NCFM-{对齐方法}
- `最佳测试准确率 (%)`: VGG-16 的最佳测试准确率，NCFM 显示 "N/A (分布匹配任务)"
- `最终测试准确率 (%)`: VGG-16 的最终测试准确率
- `训练时长 (秒)`: 训练时间（秒）
- `训练时长 (分钟)`: 训练时间（分钟）
- `训练轮数`: VGG-16 的训练轮数，NCFM 显示 "N/A (瞬时计算)"
- `随机种子`: 使用的随机种子

**详细版表格（CSV 文件）：**
- 包含上述所有字段
- 额外包含各场景的详细损失值：
  - `幅度变换_NCFM复数损失`
  - `幅度变换_NCFM幅度损失`
  - `幅度变换_NCFM相位损失`
  - `幅度变换_MSE损失`
  - `幅度变换_MMD损失`
  - `相位变换_*`（同上）
  - `完全不同_*`（同上）

**TensorBoard 对比图：**

运行实验后，在 TensorBoard 中可以查看以下对比图表：

1. **`NCFM_vs_VGG16/Overall_Performance`**：综合性能对比
   - VGG-16 的准确率（归一化）
   - NCFM 各方法的相似度指标（基于损失转换）

2. **`NCFM_vs_VGG16/Training_Time`**：训练时长对比
   - VGG-16 的训练时间（秒和分钟）
   - NCFM 各方法的训练时间（瞬时计算，为0）

3. **`NCFM_vs_VGG16/{scenario}/{metric}`**：各场景下的详细对比
   - 例如：`NCFM_vs_VGG16/phase_altered/ncfm_complex`
   - 显示 NCFM 各方法的损失值
   - VGG-16 作为参考线

4. **`VGG16/Train/Accuracy`** 和 **`VGG16/Test/Accuracy`**：VGG-16 训练过程曲线
   - 显示每个 epoch 的训练和测试准确率

**查看方式：**
```bash
tensorboard --logdir=logs/ncfm_v3
```
然后在浏览器中打开 `http://localhost:6006`，在左侧选择上述 tag 即可查看对比图表。

> **注意**：使用 CIFAR-10 需要安装 PyTorch 和 torchvision：`pip install torch torchvision`

---



