# ncfm_complete_experiment_v3.py
import os
import json
import argparse
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time

try:
    import torch
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子（统一 numpy / random / torch）
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
if torch is not None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCFM V3: Complete Distribution Matching Experiment')
    parser.add_argument('--align', type=str, default='both',
                        choices=['amplitude', 'phase', 'both', 'all'],
                        help='Alignment method: amplitude, phase, both, or all (for ablation)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples per distribution')
    parser.add_argument('--n_t_points', type=int, default=100,
                        help='Number of t points for characteristic function')
    parser.add_argument('--output_dir', type=str, default='results_v3',
                        help='Output directory for results')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results to CSV file')
    parser.add_argument('--use_tensorboard', dest='use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard to log metrics (default: True)')
    parser.add_argument('--no-use_tensorboard', dest='use_tensorboard', action='store_false',
                        help='Disable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='logs/ncfm_v3',
                        help='TensorBoard log directory')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'cifar10'],
                        help='Dataset to use: synthetic (make_moons) or cifar10')
    parser.add_argument('--run_vgg_baseline', action='store_true',
                        help='Run VGG-16 baseline for comparison (only works with cifar10)')
    return parser.parse_args()


def compute_cf(samples, t_points):
    """计算特征函数"""
    cf = np.zeros(len(t_points), dtype=complex)
    for i, t in enumerate(t_points):
        cf[i] = np.mean(np.exp(1j * samples.dot(t)))
    return cf


def mse_loss(source, target):
    """MSE损失"""
    source_norm = (source - source.mean(0)) / (source.std(0) + 1e-8)
    target_norm = (target - target.mean(0)) / (target.std(0) + 1e-8)
    return np.mean((source_norm - target_norm) ** 2)


def mmd_loss(source, target, kernel_width=1.0):
    """最大均值差异损失"""

    def rbf_kernel(x, y, gamma):
        x_norm = np.sum(x ** 2, axis=1, keepdims=True)
        y_norm = np.sum(y ** 2, axis=1, keepdims=True)
        squared_dist = x_norm + y_norm.T - 2 * np.dot(x, y.T)
        return np.exp(-gamma * squared_dist)

    gamma = 1.0 / (2 * kernel_width ** 2)
    K_xx = rbf_kernel(source, source, gamma)
    K_yy = rbf_kernel(target, target, gamma)
    K_xy = rbf_kernel(source, target, gamma)

    return np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)


def ncfm_amplitude_loss(source, target, t_points):
    """NCFM幅度对齐损失"""
    cf_source = compute_cf(source, t_points)
    cf_target = compute_cf(target, t_points)
    return np.mean((np.abs(cf_source) - np.abs(cf_target)) ** 2)


def ncfm_phase_loss(source, target, t_points):
    """NCFM相位对齐损失"""
    cf_source = compute_cf(source, t_points)
    cf_target = compute_cf(target, t_points)
    # 相位差异：比较归一化特征函数的差异
    cf_source_norm = cf_source / (np.abs(cf_source) + 1e-8)
    cf_target_norm = cf_target / (np.abs(cf_target) + 1e-8)
    return np.mean(np.abs(cf_source_norm - cf_target_norm) ** 2)


def ncfm_complex_loss(source, target, t_points):
    """NCFM复数完全对齐损失"""
    cf_source = compute_cf(source, t_points)
    cf_target = compute_cf(target, t_points)
    return np.mean(np.abs(cf_source - cf_target) ** 2)


def load_cifar10_features(n_samples=1000, feature_dim=128):
    """从CIFAR-10加载数据并提取特征"""
    if torch is None:
        raise ImportError("PyTorch is required for CIFAR-10 feature extraction. Install with: pip install torch torchvision")
    
    import torchvision
    import torchvision.transforms as transforms
    
    print("Loading CIFAR-10 dataset...")
    
    # 简单的transform（不使用数据增强，用于特征提取）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 获取数据（避免 DeprecationWarning）
    train_data_list = []
    train_labels_list = []
    for img, label in trainset:
        train_data_list.append(np.asarray(img))
        train_labels_list.append(label)
    train_data = np.array(train_data_list)
    train_labels = np.array(train_labels_list)
    
    test_data_list = []
    test_labels_list = []
    for img, label in testset:
        test_data_list.append(np.asarray(img))
        test_labels_list.append(label)
    test_data = np.array(test_data_list)
    test_labels = np.array(test_labels_list)
    
    # 展平图像数据 (32x32x3 -> 3072)
    train_data_flat = train_data.reshape(len(train_data), -1)
    test_data_flat = test_data.reshape(len(test_data), -1)
    
    # 使用PCA降维到feature_dim维度
    print(f"Applying PCA to reduce dimensions to {feature_dim}...")
    pca = PCA(n_components=feature_dim, random_state=SEED)
    train_features = pca.fit_transform(train_data_flat)
    test_features = pca.transform(test_data_flat)
    
    # 采样指定数量的样本
    if n_samples < len(train_features):
        indices = np.random.RandomState(SEED).choice(len(train_features), n_samples, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]
    
    if n_samples < len(test_features):
        indices = np.random.RandomState(SEED).choice(len(test_features), n_samples, replace=False)
        test_features = test_features[indices]
        test_labels = test_labels[indices]
    
    print(f"CIFAR-10 features extracted: train={len(train_features)}, test={len(test_features)}, dim={feature_dim}")
    
    return train_features, train_labels, test_features, test_labels


def run_vgg16_baseline(epochs=10, seed=42, writer=None):
    """运行VGG-16基线训练（简化版，用于快速对比）"""
    if torch is None:
        raise ImportError("PyTorch is required for VGG-16 baseline. Install with: pip install torch torchvision")
    
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import VGG16_Weights
    
    print(f"\n{'=' * 60}")
    print(f"Running VGG-16 Baseline (seed={seed}, epochs={epochs})")
    print(f"{'=' * 60}")
    
    start_time = time.time()
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # 创建模型
    model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # 添加自适应平均池化层，将特征图固定为 1x1
    # 这样无论输入图像大小如何，都能得到固定尺寸的特征
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # 修改分类器：VGG-16 的特征通道数是 512
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )
    model = model.to(device)
    
    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        if test_acc > best_acc:
            best_acc = test_acc
        
        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar('VGG16/Train/Accuracy', train_acc, epoch)
            writer.add_scalar('VGG16/Test/Accuracy', test_acc, epoch)
            writer.add_scalar('VGG16/Train/Loss', train_loss / len(trainloader), epoch)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {100.*correct/total:.2f}%, Test Acc: {test_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    # 记录最终结果到 TensorBoard
    if writer is not None:
        writer.add_scalar('VGG16/Final/Best_Test_Accuracy', best_acc, 0)
        writer.add_scalar('VGG16/Final/Training_Time_Seconds', training_time, 0)
        writer.add_scalar('VGG16/Final/Training_Time_Minutes', training_time / 60.0, 0)
    
    result = {
        'method': 'VGG-16',
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'training_time': training_time,
        'epochs': epochs,
        'seed': seed
    }
    
    print(f"\nVGG-16 Baseline Results:")
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    return result


def run_ablation_experiment(args, writer=None):
    """运行消融实验"""
    print("=== NCFM V3: Ablation Study ===")
    print(f"Alignment method: {args.align}")
    print(f"Dataset: {args.dataset}")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {args.output_dir}")

    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('papers/NCFM-mini', exist_ok=True)

    # 根据数据集类型加载数据
    if args.dataset == 'cifar10':
        train_features, train_labels, test_features, test_labels = load_cifar10_features(
            n_samples=args.n_samples, feature_dim=128
        )
        source = train_features
        # 对于CIFAR-10，我们使用train和test作为不同的分布场景
        target_magnitude = train_features + np.random.normal(0, 0.1, train_features.shape)
        # 使用测试集作为"完全不同"的分布
        target_gaussian = test_features[:args.n_samples] if len(test_features) >= args.n_samples else test_features
        
        # 相位变换：对特征进行旋转
        angle = np.pi / 4
        # 只对前两个维度进行旋转（简化处理）
        rotation_matrix = np.eye(source.shape[1])
        rotation_matrix[0:2, 0:2] = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        target_phase = source @ rotation_matrix.T
        
        # t点维度需要匹配特征维度
        t_points = np.random.randn(args.n_t_points, source.shape[1]) * 0.1
    else:
        # 生成测试分布（合成数据）
        print("Generating test distributions...")
        source, _ = make_moons(n_samples=args.n_samples, noise=0.1, random_state=SEED)

        # 目标分布
        target_magnitude = source + np.random.normal(0, 0.1, source.shape)

        angle = np.pi / 4
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        target_phase = source @ rotation_matrix.T

        target_gaussian, _ = make_blobs(n_samples=args.n_samples, centers=3,
                                        cluster_std=0.8, random_state=SEED)

        # 生成t点
        t_points = np.random.randn(args.n_t_points, 2) * 1.5

    # 根据对齐方法选择计算哪些损失
    results = {}
    scenarios = ['magnitude_altered', 'phase_altered', 'completely_different']
    targets = [target_magnitude, target_phase, target_gaussian]

    for scenario, target in zip(scenarios, targets):
        scenario_results = {}

        # 总是计算MSE和MMD作为基线
        scenario_results['mse'] = mse_loss(source, target)
        scenario_results['mmd'] = mmd_loss(source, target)

        # 根据对齐方法计算NCFM损失
        if args.align in ['amplitude', 'both', 'all']:
            scenario_results['ncfm_amplitude'] = ncfm_amplitude_loss(source, target, t_points)

        if args.align in ['phase', 'both', 'all']:
            scenario_results['ncfm_phase'] = ncfm_phase_loss(source, target, t_points)

        if args.align in ['both', 'all']:
            scenario_results['ncfm_complex'] = ncfm_complex_loss(source, target, t_points)

        results[scenario] = scenario_results
        
        # 记录到 TensorBoard（单个实验时）
        if writer is not None and args.align != 'all':
            for method, value in scenario_results.items():
                tag = f"{args.align}/{scenario}/{method}"
                writer.add_scalar(tag, value, 0)

    return results, source, targets, scenarios


def print_results(results, args):
    """打印结果"""
    print("\n[Results] Detailed Experimental Results:")
    print("=" * 50)

    scenario_names = {
        'magnitude_altered': 'Magnitude Transform (Slight Noise)',
        'phase_altered': 'Phase Transform (45° Rotation)',
        'completely_different': 'Completely Different (Moon vs Gaussian)'
    }

    for scenario in results:
        print(f"\n{scenario_names[scenario]}:")
        for method, value in results[scenario].items():
            print(f"   • {method.upper():<15}: {value:.6f}")


def create_ablation_table(results_list, align_methods):
    """创建消融实验对比表"""
    scenarios = ['magnitude_altered', 'phase_altered', 'completely_different']
    methods = ['mse', 'mmd', 'ncfm_amplitude', 'ncfm_phase', 'ncfm_complex']

    # 创建对比数据
    comparison_data = []
    for i, (results, align_method) in enumerate(zip(results_list, align_methods)):
        for scenario in scenarios:
            row = {
                'alignment_method': align_method,
                'scenario': scenario,
            }
            for method in methods:
                if method in results[scenario]:
                    row[method] = results[scenario][method]
                else:
                    row[method] = None
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df


def create_ncfm_vgg_comparison_table(ncfm_results_list, ncfm_align_methods, vgg_result, output_dir):
    """创建NCFM与VGG-16的对比表格（包含训练时长）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备对比数据
    comparison_data = []
    
    # 添加VGG-16结果
    comparison_data.append({
        '方法 (Method)': 'VGG-16',
        '最佳测试准确率 (%)': f"{vgg_result['best_test_acc']:.2f}",
        '最终测试准确率 (%)': f"{vgg_result['final_test_acc']:.2f}",
        '训练时长 (秒)': f"{vgg_result['training_time']:.2f}",
        '训练时长 (分钟)': f"{vgg_result['training_time'] / 60.0:.2f}",
        '训练轮数 (Epochs)': vgg_result['epochs'],
        '随机种子 (Seed)': vgg_result['seed'],
        '数据集 (Dataset)': 'CIFAR-10',
        '评测标准 (Metric)': '分类准确率'
    })
    
    # 添加NCFM结果（包含所有场景的指标）
    scenarios = ['magnitude_altered', 'phase_altered', 'completely_different']
    scenario_names = {
        'magnitude_altered': '幅度变换',
        'phase_altered': '相位变换',
        'completely_different': '完全不同'
    }
    
    for results, align_method in zip(ncfm_results_list, ncfm_align_methods):
        # 为每个NCFM方法创建一个条目
        row = {
            '方法 (Method)': f'NCFM-{align_method}',
            '最佳测试准确率 (%)': 'N/A',
            '最终测试准确率 (%)': 'N/A',
            '训练时长 (秒)': '0.00',
            '训练时长 (分钟)': '0.00',
            '训练轮数 (Epochs)': 'N/A',
            '随机种子 (Seed)': SEED,
            '数据集 (Dataset)': 'CIFAR-10',
            '评测标准 (Metric)': '分布匹配损失'
        }
        
        # 添加各场景的损失值
        for scenario in scenarios:
            if scenario in results:
                scenario_name = scenario_names[scenario]
                # NCFM复数损失
                if 'ncfm_complex' in results[scenario]:
                    row[f'{scenario_name}_NCFM复数损失'] = f"{results[scenario]['ncfm_complex']:.6f}"
                # NCFM幅度损失
                if 'ncfm_amplitude' in results[scenario]:
                    row[f'{scenario_name}_NCFM幅度损失'] = f"{results[scenario]['ncfm_amplitude']:.6f}"
                # NCFM相位损失
                if 'ncfm_phase' in results[scenario]:
                    row[f'{scenario_name}_NCFM相位损失'] = f"{results[scenario]['ncfm_phase']:.6f}"
                # MSE损失
                if 'mse' in results[scenario]:
                    row[f'{scenario_name}_MSE损失'] = f"{results[scenario]['mse']:.6f}"
                # MMD损失
                if 'mmd' in results[scenario]:
                    row[f'{scenario_name}_MMD损失'] = f"{results[scenario]['mmd']:.6f}"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 保存CSV
    comparison_file = os.path.join(output_dir, f'ncfm_vgg_comparison_{timestamp}.csv')
    df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"\n[Comparison] NCFM vs VGG-16 comparison saved to: {comparison_file}")
    
    # 打印表格（使用更清晰的格式）
    print("\n" + "=" * 120)
    print("NCFM vs VGG-16 对比表格 (Comparison Table)")
    print("=" * 120)
    
    # 创建简化版表格用于打印（只显示关键指标）
    simple_data = []
    simple_data.append({
        '方法 (Method)': 'VGG-16',
        '最佳测试准确率 (%)': f"{vgg_result['best_test_acc']:.2f}",
        '最终测试准确率 (%)': f"{vgg_result['final_test_acc']:.2f}",
        '训练时长 (秒)': f"{vgg_result['training_time']:.2f}",
        '训练时长 (分钟)': f"{vgg_result['training_time'] / 60.0:.2f}",
        '训练轮数': vgg_result['epochs'],
        '随机种子': vgg_result['seed']
    })
    
    for results, align_method in zip(ncfm_results_list, ncfm_align_methods):
        simple_data.append({
            '方法 (Method)': f'NCFM-{align_method}',
            '最佳测试准确率 (%)': 'N/A (分布匹配任务)',
            '最终测试准确率 (%)': 'N/A (分布匹配任务)',
            '训练时长 (秒)': '0.00',
            '训练时长 (分钟)': '0.00',
            '训练轮数': 'N/A (瞬时计算)',
            '随机种子': SEED
        })
    
    simple_df = pd.DataFrame(simple_data)
    print(simple_df.to_string(index=False))
    
    # 打印NCFM的详细损失值
    print("\n" + "-" * 120)
    print("NCFM 详细损失值 (Detailed Loss Values):")
    print("-" * 120)
    for results, align_method in zip(ncfm_results_list, ncfm_align_methods):
        print(f"\n方法: NCFM-{align_method}")
        for scenario in scenarios:
            if scenario in results:
                scenario_name = scenario_names[scenario]
                print(f"  场景: {scenario_name}")
                if 'ncfm_complex' in results[scenario]:
                    print(f"    NCFM复数损失: {results[scenario]['ncfm_complex']:.6f}")
                if 'ncfm_amplitude' in results[scenario]:
                    print(f"    NCFM幅度损失: {results[scenario]['ncfm_amplitude']:.6f}")
                if 'ncfm_phase' in results[scenario]:
                    print(f"    NCFM相位损失: {results[scenario]['ncfm_phase']:.6f}")
                if 'mse' in results[scenario]:
                    print(f"    MSE损失: {results[scenario]['mse']:.6f}")
                if 'mmd' in results[scenario]:
                    print(f"    MMD损失: {results[scenario]['mmd']:.6f}")
    
    print("\n" + "=" * 120)
    print("说明 (Notes):")
    print("  - VGG-16: 使用CIFAR-10数据集进行图像分类任务，评测标准为测试集准确率")
    print("  - NCFM: 使用CIFAR-10特征进行分布匹配任务，评测标准为分布匹配损失（越小越好）")
    print("  - 详细结果（包含所有场景的损失值）已保存到CSV文件")
    print("=" * 120)
    
    return df


def convert_to_native_types(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def save_results(results, args, ablation_df=None):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON结果
    result_file = os.path.join(args.output_dir, f'ncfm_v3_results_{timestamp}.json')
    with open(result_file, 'w') as f:
        # 转换numpy类型为Python原生类型
        results_converted = convert_to_native_types(results)
        args_dict = vars(args)
        # 过滤掉不能序列化的对象
        args_dict_clean = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                          for k, v in args_dict.items()}
        json.dump({
            'args': args_dict_clean,
            'results': results_converted,
            'seed': SEED,
            'timestamp': timestamp
        }, f, indent=2)

    # 保存CSV结果
    if args.save_csv:
        csv_file = os.path.join(args.output_dir, f'ncfm_v3_results_{timestamp}.csv')

        # 转换结果为CSV格式
        csv_data = []
        for scenario, scenario_results in results.items():
            for method, value in scenario_results.items():
                # 转换numpy类型为Python原生类型
                value_native = convert_to_native_types(value)
                csv_data.append({
                    'scenario': scenario,
                    'method': method,
                    'value': value_native,
                    'alignment_method': args.align,
                    'seed': SEED,
                    'timestamp': timestamp
                })

        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"[CSV] Results saved to: {csv_file}")

    # 保存消融实验对比表
    if ablation_df is not None:
        ablation_file = os.path.join(args.output_dir, f'ncfm_ablation_comparison_{timestamp}.csv')
        ablation_df.to_csv(ablation_file, index=False)
        print(f"[Ablation] Comparison saved to: {ablation_file}")

    print(f"[JSON] Results saved to: {result_file}")


def log_comparison_to_tensorboard(writer, results_list, align_methods):
    """将消融实验结果记录到 TensorBoard，形成对比图表"""
    if writer is None:
        return
    
    scenarios = ['magnitude_altered', 'phase_altered', 'completely_different']
    all_metrics = ['mse', 'mmd', 'ncfm_amplitude', 'ncfm_phase', 'ncfm_complex']
    
    # 为每个场景和每个指标创建对比图表
    for scenario in scenarios:
        for metric in all_metrics:
            # 收集所有对齐方法在该场景下的该指标值
            metric_dict = {}
            for align_method, results in zip(align_methods, results_list):
                if metric in results[scenario]:
                    metric_dict[f'NCFM-{align_method}'] = results[scenario][metric]
            
            # 如果有数据，使用 add_scalars 创建对比图表
            if metric_dict:
                tag = f"Comparison/{scenario}/{metric}"
                writer.add_scalars(tag, metric_dict, 0)
    
    # 为每个对齐方法创建汇总图表（所有场景的同一指标）
    for metric in all_metrics:
        for align_method, results in zip(align_methods, results_list):
            scenario_dict = {}
            for scenario in scenarios:
                if metric in results[scenario]:
                    scenario_dict[scenario] = results[scenario][metric]
            
            if scenario_dict:
                tag = f"Method_Summary/{align_method}/{metric}"
                writer.add_scalars(tag, scenario_dict, 0)


def log_ncfm_vgg_comparison_to_tensorboard(writer, ncfm_results_list, ncfm_align_methods, vgg_result):
    """将NCFM和VGG-16的对比结果记录到TensorBoard"""
    if writer is None:
        return
    
    scenarios = ['magnitude_altered', 'phase_altered', 'completely_different']
    
    # 1. 创建综合性能对比（使用归一化的指标）
    # 将NCFM损失和VGG-16准确率放在同一个对比图中
    comparison_dict = {}
    
    # 添加VGG-16的准确率（归一化到0-1范围，用于对比）
    comparison_dict['VGG-16 (Accuracy/100)'] = vgg_result['best_test_acc'] / 100.0
    
    # 添加NCFM的复数损失（使用phase_altered场景，取倒数并归一化，使其与准确率方向一致）
    for align_method, results in zip(ncfm_align_methods, ncfm_results_list):
        if 'phase_altered' in results and 'ncfm_complex' in results['phase_altered']:
            # 将损失转换为"相似度"（越小越好 -> 越大越好）
            loss = results['phase_altered']['ncfm_complex']
            # 使用sigmoid归一化，使损失值映射到0-1范围
            similarity = 1.0 / (1.0 + loss)  # 简单的倒数归一化
            comparison_dict[f'NCFM-{align_method} (Similarity)'] = similarity
    
    if comparison_dict:
        writer.add_scalars('NCFM_vs_VGG16/Overall_Performance', comparison_dict, 0)
    
    # 2. 训练时长对比
    time_comparison = {
        'VGG-16 (seconds)': vgg_result['training_time'],
        'VGG-16 (minutes)': vgg_result['training_time'] / 60.0
    }
    # NCFM是瞬时计算，训练时间为0
    for align_method in ncfm_align_methods:
        time_comparison[f'NCFM-{align_method} (seconds)'] = 0.0
    
    writer.add_scalars('NCFM_vs_VGG16/Training_Time', time_comparison, 0)
    
    # 3. 各场景下的NCFM损失对比（VGG-16作为参考线）
    for scenario in scenarios:
        for metric in ['ncfm_complex', 'ncfm_amplitude']:
            metric_dict = {}
            for align_method, results in zip(ncfm_align_methods, ncfm_results_list):
                if scenario in results and metric in results[scenario]:
                    metric_dict[f'NCFM-{align_method}'] = results[scenario][metric]
            
            if metric_dict:
                # 添加VGG-16的参考值（使用准确率的倒数作为"损失"参考）
                # 这里使用一个固定值作为参考，表示VGG-16的性能水平
                metric_dict['VGG-16 (Reference)'] = (100.0 - vgg_result['best_test_acc']) / 100.0
                
                tag = f"NCFM_vs_VGG16/{scenario}/{metric}"
                writer.add_scalars(tag, metric_dict, 0)
    
    print("[TensorBoard] NCFM vs VGG-16 comparison logged to TensorBoard")


def calculate_percentage_differences(results_list, align_methods):
    """计算百分比差异"""
    print("\n[Analysis] Percentage Differences:")
    print("=" * 40)

    baseline_results = results_list[0]  # 第一个方法作为基线
    baseline_method = align_methods[0]

    for i in range(1, len(results_list)):
        current_method = align_methods[i]
        current_results = results_list[i]

        print(f"\n{current_method} vs {baseline_method}:")

        for scenario in ['phase_altered']:  # 重点关注相位变换场景
            print(f"  {scenario}:")
            for method in ['ncfm_amplitude', 'ncfm_complex']:
                if method in baseline_results[scenario] and method in current_results[scenario]:
                    base_val = baseline_results[scenario][method]
                    curr_val = current_results[scenario][method]
                    if base_val > 0:
                        diff_pct = ((curr_val - base_val) / base_val) * 100
                        print(f"    {method}: {diff_pct:+.1f}%")


def create_additional_visualizations(results_list, align_methods, output_dir):
    """创建额外的可视化图表"""
    # 创建性能对比雷达图
    from math import pi

    scenarios = ['Magnitude\nTransform', 'Phase\nTransform', 'Completely\nDifferent']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 雷达图
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    angles = [n / float(len(scenarios)) * 2 * pi for n in range(len(scenarios))]
    angles += angles[:1]

    for idx, (results, align_method) in enumerate(zip(results_list, align_methods)):
        values = []
        for scenario in ['magnitude_altered', 'phase_altered', 'completely_different']:
            # 使用ncfm_amplitude作为代表指标
            value = results[scenario].get('ncfm_amplitude', results[scenario].get('ncfm_complex', 0))
            values.append(np.log10(value + 1e-8))

        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=align_method, color=colors[idx])
        ax1.fill(angles, values, alpha=0.1, color=colors[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(scenarios)
    ax1.set_ylim(-8, 1)
    ax1.grid(True)
    ax1.set_title('NCFM Methods Radar Comparison\n(Log Scale)', fontweight='bold', pad=20)
    ax1.legend()

    # 相对性能热力图
    ax2 = axes[1]
    methods = ['mse', 'mmd', 'ncfm_amplitude', 'ncfm_complex']
    method_names = ['MSE', 'MMD', 'NCFM-Mag', 'NCFM-Comp']

    # 使用第一个结果集创建热力图
    results = results_list[0]
    ranking_matrix = np.zeros((len(methods), len(scenarios)))

    for j, scenario in enumerate(['magnitude_altered', 'phase_altered', 'completely_different']):
        values = [results[scenario].get(method, float('inf')) for method in methods]
        ranks = np.argsort(np.argsort(values)) + 1
        ranking_matrix[:, j] = ranks

    im = ax2.imshow(ranking_matrix, cmap='YlGnBu_r', aspect='auto')

    ax2.set_xticks(range(len(scenarios)))
    ax2.set_yticks(range(len(methods)))
    ax2.set_xticklabels(['Magnitude\nTransform', 'Phase\nTransform', 'Completely\nDifferent'])
    ax2.set_yticklabels(method_names)

    for i in range(len(methods)):
        for j in range(len(scenarios)):
            rank = int(ranking_matrix[i, j])
            color = 'white' if rank <= 2 else 'black'
            ax2.text(j, i, f'#{rank}', ha="center", va="center",
                     color=color, fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax2, label='Performance Ranking\n(#1 = Best)')
    ax2.set_title('Relative Performance Heatmap\n(Method Rankings)', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ncfm_additional_visualizations.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Viz] Additional visualizations saved to: {output_dir}/ncfm_additional_visualizations.png")


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化 TensorBoard writer
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"[TensorBoard] Logging to: {args.log_dir}")
    elif args.use_tensorboard and not TENSORBOARD_AVAILABLE:
        print("[Warning] TensorBoard requested but not available. Install with: pip install tensorboard")

    if args.align == 'all':
        # 运行完整的消融实验
        align_methods = ['amplitude', 'phase', 'both']
        all_results = []

        for align_method in align_methods:
            print(f"\n{'=' * 60}")
            print(f"Running experiment with align={align_method}")
            print(f"{'=' * 60}")

            args.align = align_method
            results, _, _, _ = run_ablation_experiment(args, writer=writer)
            all_results.append(results)
            print_results(results, args)

        # 创建对比表
        ablation_df = create_ablation_table(all_results, align_methods)

        # 记录对比数据到 TensorBoard（形成直观的对比图表）
        if writer is not None:
            log_comparison_to_tensorboard(writer, all_results, align_methods)

        # 计算百分比差异
        calculate_percentage_differences(all_results, align_methods)

        # 创建额外的可视化图表
        create_additional_visualizations(all_results, align_methods, args.output_dir)

        # 保存结果
        save_results(all_results[0], args, ablation_df)

        # 如果使用CIFAR-10且需要运行VGG基线
        if args.dataset == 'cifar10' and args.run_vgg_baseline:
            print("\n" + "=" * 80)
            print("Running VGG-16 Baseline for Comparison")
            print("=" * 80)
            try:
                vgg_result = run_vgg16_baseline(epochs=10, seed=SEED, writer=writer)
                # 创建NCFM与VGG-16的对比表格
                create_ncfm_vgg_comparison_table(all_results, align_methods, vgg_result, args.output_dir)
                # 记录到TensorBoard进行对比可视化
                if writer is not None:
                    log_ncfm_vgg_comparison_to_tensorboard(writer, all_results, align_methods, vgg_result)
            except Exception as e:
                print(f"[Warning] Failed to run VGG-16 baseline: {e}")
                print("Continuing without VGG-16 comparison...")

        print("\nAblation study completed!")
        
        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()
            print(f"[TensorBoard] Logs saved. View with: tensorboard --logdir={args.log_dir}")

    else:
        # 运行单个实验
        results, source, targets, scenarios = run_ablation_experiment(args, writer=writer)
        print_results(results, args)

        # 为单个实验也创建可视化
        create_additional_visualizations([results], [args.align], args.output_dir)

        save_results(results, args)
        
        # 如果使用CIFAR-10且需要运行VGG基线（单个实验时也支持）
        if args.dataset == 'cifar10' and args.run_vgg_baseline:
            print("\n" + "=" * 80)
            print("Running VGG-16 Baseline for Comparison")
            print("=" * 80)
            try:
                vgg_result = run_vgg16_baseline(epochs=10, seed=SEED, writer=writer)
                # 创建NCFM与VGG-16的对比表格
                create_ncfm_vgg_comparison_table([results], [args.align], vgg_result, args.output_dir)
                # 记录到TensorBoard进行对比可视化
                if writer is not None:
                    log_ncfm_vgg_comparison_to_tensorboard(writer, [results], [args.align], vgg_result)
            except Exception as e:
                print(f"[Warning] Failed to run VGG-16 baseline: {e}")
                print("Continuing without VGG-16 comparison...")

        print("\nExperiment completed!")
        
        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()
            print(f"[TensorBoard] Logs saved. View with: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()

