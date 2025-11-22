# run_ncfm_v3.sh
#!/bin/bash

# NCFM V3 实验运行脚本
echo "Running NCFM V3 Experiments..."

# 创建结果目录
mkdir -p results_v3
mkdir -p papers/NCFM-mini

# 运行完整的消融实验
echo "1. Running full ablation study..."
python ncfm_complete_experiment_v3.py --align=all --save_csv --output_dir=results_v3

# 运行单个对齐方法的实验
echo "2. Running amplitude-only experiment..."
python ncfm_complete_experiment_v3.py --align=amplitude --save_csv --output_dir=results_v3

echo "3. Running phase-only experiment..."
python ncfm_complete_experiment_v3.py --align=phase --save_csv --output_dir=results_v3

echo "4. Running complex alignment experiment..."
python ncfm_complete_experiment_v3.py --align=both --save_csv --output_dir=results_v3

echo "✅ All NCFM V3 experiments completed!"
echo "📊 Results saved to: results_v3/"
echo "📁 Ablation data saved to: papers/NCFM-mini/"





