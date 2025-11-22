# NCFM V3 使用说明

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r NCFM_V3/requirements.txt

# 如果使用 CIFAR-10 和 VGG-16 基线，还需要安装 PyTorch
pip install torch torchvision
```

### 2. 基础实验（合成数据）

```bash
# 完整消融实验（推荐）
python NCFM_V3/ncfm_complete_experiment_v3.py --align=all --save_csv

# 单个对齐方法
python NCFM_V3/ncfm_complete_experiment_v3.py --align=both --save_csv
```

### 3. VGG-16 基线对比实验（CIFAR-10）

**注意：** 如果在 `NCFM_V3` 目录下运行，直接使用脚本名；如果在项目根目录运行，使用 `NCFM_V3/` 前缀。

**完整对比实验（推荐）：**

*在 NCFM_V3 目录下：*
```bash
python ncfm_complete_experiment_v3.py --align=all --dataset=cifar10 --run_vgg_baseline --save_csv --output_dir=results_v3
```

*在项目根目录下：*
```bash
python NCFM_V3/ncfm_complete_experiment_v3.py --align=all --dataset=cifar10 --run_vgg_baseline --save_csv --output_dir=results_v3
```

**Windows 下的多行命令（使用 `^` 作为续行符）：**
```cmd
python ncfm_complete_experiment_v3.py ^
    --align=all ^
    --dataset=cifar10 ^
    --run_vgg_baseline ^
    --save_csv ^
    --output_dir=results_v3
```

**单个方法对比：**
```bash
# 在 NCFM_V3 目录下
python ncfm_complete_experiment_v3.py --align=both --dataset=cifar10 --run_vgg_baseline --save_csv --output_dir=results_v3
```

## 参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--align` | 对齐方法 | `both` | `amplitude`, `phase`, `both`, `all` |
| `--dataset` | 数据集 | `synthetic` | `synthetic`, `cifar10` |
| `--run_vgg_baseline` | 运行 VGG-16 基线 | `False` | 需要 `--dataset=cifar10` |
| `--save_csv` | 保存 CSV 结果 | `False` | - |
| `--output_dir` | 输出目录 | `results_v3` | - |
| `--n_samples` | 每个分布的样本数 | `1000` | - |
| `--n_t_points` | 特征函数 t 点数量 | `100` | - |
| `--use_tensorboard` | 启用 TensorBoard | `True` | - |
| `--log_dir` | TensorBoard 日志目录 | `logs/ncfm_v3` | - |

## 输出结果

### 1. 基础实验输出

- `ncfm_v3_results_*.json`: JSON 格式的详细结果
- `ncfm_v3_results_*.csv`: CSV 格式的结果（如果使用 `--save_csv`）
- `ncfm_ablation_comparison_*.csv`: 消融实验对比表（如果使用 `--align=all`）
- `ncfm_additional_visualizations.png`: 可视化图表

### 2. VGG-16 对比输出

- `ncfm_vgg_comparison_*.csv`: **NCFM vs VGG-16 对比表格**
  - 包含训练时长（秒和分钟）
  - 包含所有场景的损失值
  - 包含测试准确率等指标

### 3. TensorBoard 日志

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/ncfm_v3

# 在浏览器中打开
# http://localhost:6006
```

**可查看的对比图表：**
- `NCFM_vs_VGG16/Overall_Performance`: 综合性能对比
- `NCFM_vs_VGG16/Training_Time`: 训练时长对比
- `NCFM_vs_VGG16/{scenario}/{metric}`: 各场景详细对比
- `VGG16/Train/Accuracy`: VGG-16 训练准确率曲线
- `VGG16/Test/Accuracy`: VGG-16 测试准确率曲线

## 对比表格说明

### 表格内容

对比表格包含以下信息：

**VGG-16 行：**
- 方法名称：`VGG-16`
- 最佳测试准确率（%）
- 最终测试准确率（%）
- 训练时长（秒和分钟）
- 训练轮数
- 随机种子
- 数据集：`CIFAR-10`
- 评测标准：`分类准确率`

**NCFM 行（每个对齐方法一行）：**
- 方法名称：`NCFM-{对齐方法}`
- 测试准确率：`N/A (分布匹配任务)`
- 训练时长：`0.00 (瞬时计算)`
- 训练轮数：`N/A (瞬时计算)`
- 随机种子：`42`
- 数据集：`CIFAR-10`
- 评测标准：`分布匹配损失`
- **各场景的详细损失值：**
  - 幅度变换场景：NCFM复数损失、幅度损失、相位损失、MSE损失、MMD损失
  - 相位变换场景：同上
  - 完全不同场景：同上

### 表格格式

- **终端输出**：简化版表格，只显示关键指标
- **CSV 文件**：完整版表格，包含所有场景的详细损失值

## 常见问题

### Q1: 运行 VGG-16 基线时出现维度错误？

**A:** 已修复。如果仍遇到问题，请确保使用最新版本的代码。

### Q2: 如何只运行 VGG-16 基线，不运行 NCFM？

**A:** 目前不支持。VGG-16 基线是作为对比基准，需要与 NCFM 结果一起生成对比表格。

### Q3: 可以修改 VGG-16 的训练轮数吗？

**A:** 可以。修改 `run_vgg16_baseline` 函数中的 `epochs` 参数（默认 10）。

### Q4: 对比表格中的训练时长单位是什么？

**A:** 同时提供秒和分钟两种单位，方便不同场景使用。

### Q5: NCFM 的训练时长为 0，这是正常的吗？

**A:** 是的。NCFM 是瞬时计算分布匹配损失，不需要训练过程，所以训练时长为 0。

## 示例输出

运行完整对比实验后，终端会显示：

```
================================================================================
NCFM vs VGG-16 对比表格 (Comparison Table)
================================================================================
  方法 (Method)  最佳测试准确率 (%)  最终测试准确率 (%)  训练时长 (秒)  训练时长 (分钟)  训练轮数  随机种子
0         VGG-16             85.23             84.56        125.34           2.09        10        42
1    NCFM-amplitude  N/A (分布匹配任务)  N/A (分布匹配任务)         0.00         0.00 (瞬时计算)  N/A (瞬时计算)        42
2      NCFM-phase  N/A (分布匹配任务)  N/A (分布匹配任务)         0.00         0.00 (瞬时计算)  N/A (瞬时计算)        42
3       NCFM-both  N/A (分布匹配任务)  N/A (分布匹配任务)         0.00         0.00 (瞬时计算)  N/A (瞬时计算)        42

详细结果（包含所有场景的损失值）已保存到CSV文件。
================================================================================
```

CSV 文件会包含更详细的各场景损失值信息。

