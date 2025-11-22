@echo off
REM NCFM V3 VGG-16 基线对比实验脚本 (Windows)
echo Running NCFM V3 with VGG-16 Baseline Comparison...

REM 创建结果目录
if not exist results_v3 mkdir results_v3
if not exist papers\NCFM-mini mkdir papers\NCFM-mini

REM 运行完整的消融实验 + VGG-16 基线对比
echo Running full ablation study with VGG-16 baseline...
python ncfm_complete_experiment_v3.py --align=all --dataset=cifar10 --run_vgg_baseline --save_csv --output_dir=results_v3

echo.
echo ========================================
echo All experiments completed!
echo Results saved to: results_v3/
echo ========================================
pause



