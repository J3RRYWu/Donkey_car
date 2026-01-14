# 如何运行 Predictor 评估代码

重构后的代码**完全向后兼容**，使用方法和之前一模一样！

## 📋 前提条件

1. **Python 环境**: Python 3.11
2. **必需的包**:
   ```bash
   pip install torch numpy matplotlib imageio pillow
   ```
3. **数据文件**: NPZ 格式的轨迹数据（在 `../npz_transfer/` 目录）
4. **模型权重**: 
   - Predictor checkpoint: `checkpoints/best_model.pt`
   - VAE checkpoint: `../checkpoints_64x64/vae_epoch_300.pth`

## 🚀 快速开始

### 1. 基本评估（所有检查 + 可视化）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz traj2.npz \
    --max_horizon 30 \
    --batch_size 4
```

**输出**:
- Check 1: Baseline vs LSTM one-step prediction
- Check 2: Multi-step rollout (30 steps)
- Check 3: 可视化图像（GT vs VAE vs LSTM）
- MSE vs horizon 曲线图
- JSON/CSV 评估结果

---

### 2. 长时间评估（50 步 + 使用 NPZ GT）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --batch_size 4
```

**`--gt_from_npz` 的作用**:
- 直接从 NPZ 文件读取未来的 GT 帧（不受 `sequence_length` 限制）
- 可以评估超过窗口长度的 horizon
- 适合评估长期预测能力

---

### 3. Conformal Prediction (CP) 评估

#### 3.1 校准 + 评估覆盖率

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --cp_calibrate \
    --cp_eval \
    --cp_alpha 0.05 \
    --cp_calib_size 500
```

**输出**:
- `cp_quantiles.json`: 每个 horizon 的 q_t 值
- `cp_coverage.csv`: 每个 horizon 的覆盖率
- `cp_quantiles.png`: q_t 曲线图
- `cp_coverage.png`: 覆盖率曲线图

#### 3.2 CP 轨迹可视化（PCA 投影 + 置信带）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --cp_traj_plot \
    --cp_traj_sample_idx 0 \
    --cp_traj_horizon 50 \
    --cp_quantiles_path eval_results/cp_quantiles.json
```

**输出**:
- `cp_band_traj_sample_0.png`: 2D PCA 空间中的预测轨迹 + CP 置信带

#### 3.3 CP 边界采样可视化（解码边界点）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --cp_boundary_plot \
    --cp_boundary_step 20 \
    --cp_boundary_num 4 \
    --cp_quantiles_path eval_results/cp_quantiles.json
```

**输出**:
- `cp_boundary_decode_t20_sample_0.png`: 在 step 20 的 CP 球面边界上采样 4 个点并解码为图像

---

### 4. 生成长期预测视频

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --generate_video \
    --video_steps 100 \
    --video_fps 10 \
    --video_sample_idx 0 \
    --video_action_mode from_npz \
    --video_layout gt_pred
```

**输出**:
- `prediction_100step.mp4`: 100 步的预测视频
- `--video_layout gt_pred`: 左边 GT，右边预测（方便对比）

---

### 5. 只运行 CP（节省时间）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --only_cp \
    --cp_calibrate \
    --cp_eval \
    --cp_traj_plot \
    --cp_boundary_plot \
    --max_horizon 50 \
    --gt_from_npz
```

**`--only_cp` 的作用**:
- 跳过 Check 1, Check 2, 标准可视化
- 只运行 CP 相关的计算和可视化
- 大幅节省时间

---

### 6. 跳过特定部分（自定义运行）

```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --skip_check1 \
    --skip_visualize \
    --skip_exports
```

**跳过选项**:
- `--skip_check1`: 跳过 baseline vs LSTM 对比
- `--skip_check2`: 跳过 multi-step rollout
- `--skip_visualize`: 跳过可视化图像
- `--skip_rollout_plot`: 跳过 MSE 曲线图
- `--skip_exports`: 跳过 JSON/CSV 导出

---

## 📊 关键参数说明

### 数据相关
- `--sequence_length 16`: NPZ 窗口长度（默认 16）
- `--input_length 15`: 输入帧数量
- `--target_length 15`: 目标帧数量
- `--target_offset 1`: 目标帧起始位置
- `--gt_from_npz`: 从 NPZ 直接读取未来 GT（突破窗口限制）

### 评估相关
- `--max_horizon 50`: 最大预测步数
- `--batch_size 4`: 批次大小
- `--max_eval_batches 10`: 限制评估批次数（快速测试用）
- `--mc_samples 1`: MC-dropout 采样次数（>1 启用不确定性估计）

### CP 相关
- `--cp_alpha 0.05`: 置信度（0.05 = 95% 覆盖率）
- `--cp_norm l2`: 距离范数（l2 或 linf）
- `--cp_calib_size 500`: 校准集大小
- `--cp_seed 42`: 随机种子

### 输出相关
- `--save_dir ./eval_results`: 结果保存目录
- `--device auto`: 设备选择（auto/cuda/cpu）

---

## 📁 输出文件说明

运行后会在 `eval_results/` 目录生成：

### 标准评估
```
eval_results/
├── eval_results.json              # 完整评估结果
├── rollout_metrics.json           # 详细指标 + 有效 horizon
├── rollout_latent_mse.csv         # Latent MSE 曲线数据
├── rollout_img_mse.csv            # Image MSE 曲线数据
├── rollout_psnr.csv               # PSNR 曲线数据
├── rollout_ssim.csv               # SSIM 曲线数据
├── effective_horizon.csv          # 有效预测范围
├── rollout_mse_vs_horizon.png     # MSE 曲线图
├── prediction_sample_1.png        # 样本可视化（多个）
└── rollout_30step.png             # 30 步 rollout 可视化
```

### CP 评估
```
eval_results/
├── cp_quantiles.json              # CP 分位数
├── cp_coverage.csv                # 覆盖率数据
├── cp_quantiles.png               # q_t 曲线图
├── cp_coverage.png                # 覆盖率曲线图
├── cp_band_traj_sample_0.png      # CP 轨迹可视化
└── cp_boundary_decode_t20_sample_0.png  # 边界采样解码
```

### 视频
```
eval_results/
└── prediction_100step.mp4         # 预测视频
```

---

## 💡 常见使用场景

### 场景 1: 快速测试代码是否能跑
```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 10 \
    --max_eval_batches 2 \
    --skip_visualize
```

### 场景 2: 完整论文评估（所有指标）
```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz traj2.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --cp_calibrate \
    --cp_eval \
    --cp_traj_plot \
    --cp_boundary_plot
```

### 场景 3: 生成 PPT 演示材料
```bash
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --max_horizon 50 \
    --gt_from_npz \
    --generate_video \
    --video_steps 100 \
    --video_layout gt_pred \
    --cp_traj_plot \
    --cp_boundary_plot \
    --num_vis_samples 5
```

---

## ⚠️ 注意事项

1. **路径问题**: 确保从项目根目录运行，或调整相对路径
2. **内存占用**: `--gt_from_npz` + 大 `max_horizon` 会占用较多内存
3. **GPU 使用**: 默认自动检测，也可以手动指定 `--device cuda` 或 `--device cpu`
4. **CP 需要先校准**: 使用 `--cp_traj_plot` 或 `--cp_boundary_plot` 前需要先运行 `--cp_calibrate`
5. **视频生成**: 需要安装 `imageio` 和 `ffmpeg`

---

## 🐛 问题排查

### 问题 1: `ModuleNotFoundError`
```bash
# 确保在 predictor/ 的父目录运行
cd d:/donkey_car/Donkey_car
py -3.11 predictor/eval_predictor.py ...
```

### 问题 2: `FileNotFoundError` (找不到 NPZ)
```bash
# 检查数据路径
ls ../npz_transfer/
# 或调整 --data_dir 参数
```

### 问题 3: CP 可视化报错 `File not found: cp_quantiles.json`
```bash
# 先运行校准
py -3.11 predictor/eval_predictor.py ... --cp_calibrate
# 然后再运行可视化
py -3.11 predictor/eval_predictor.py ... --cp_traj_plot
```

---

## 📚 参考文档

- `EVAL_MODULES.md`: 模块结构说明
- `eval_predictor.py --help`: 查看所有参数
- 各模块文件顶部有详细的函数文档字符串

---

**重构后的代码 100% 向后兼容，所有原来的命令都能正常工作！** ✅
