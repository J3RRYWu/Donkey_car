# Conformal MPC 快速开始指南

## 🚀 5分钟测试MPC

### 步骤1: 确保已完成CP评估

```bash
# 应该已有以下文件:
eval_results/cp_quantiles.json          # CP分位数
eval_results/cp_safety_report.json      # 安全报告
checkpoints/best_model.pt               # LSTM模型
../vae_recon/best_model.pt              # VAE模型
```

---

### 步骤2: 运行MPC离线测试

```bash
cd ~/Donkey_car/predictor

python3 test_mpc.py \
    --lstm_path checkpoints/best_model.pt \
    --vae_path ../vae_recon/best_model.pt \
    --cp_path eval_results/cp_quantiles.json \
    --data_dir ../npz_data \
    --npz_file traj1_64x64.npz \
    --horizon 17 \
    --test_idx 100 \
    --mode offline
```

**输出**:
```
==================================================================
Testing Conformal MPC (Offline)
==================================================================
[1/5] Loading models...
✓ LSTM loaded: latent_dim=64, hidden=256
✓ VAE loaded: frozen=True

[2/5] Loading test data...
✓ Dataset loaded: 5786 sequences

[3/5] Initializing MPC controller...
✓ MPC initialized: horizon=17 steps
  CP quantiles: q_1=20.12, q_17=45.67

[4/5] Running MPC control loop...
✓ MPC optimization complete!
  Optimal action: steering=0.234, throttle=0.567
  Final cost: 12.3456
  Tracking error @ horizon: 2.456

[5/5] Generating visualizations...
✓ Saved plot: mpc_test_results/mpc_test_results.png
✓ Saved results: mpc_test_results/mpc_test_results.json
```

---

### 步骤3: 查看结果

**生成的文件**:
```
mpc_test_results/
├── mpc_test_results.png       # 4张图：成本收敛、误差、动作、CP不确定性
└── mpc_test_results.json      # 数值结果
```

**关键指标解读**:

| 指标 | 期望值 | 含义 |
|------|--------|------|
| `final_cost` | < 50 | 优化收敛良好 |
| `tracking_error_mean` | < 3.0 | 平均跟踪误差小 |
| `optimal_action` | 接近GT | MPC学到了合理策略 |

---

## 📖 理解MPC输出图表

### 图1: 成本收敛曲线
```
- Y轴: Total Cost
- X轴: Optimization Iteration
- 期望: 下降趋势并收敛（最后10次迭代应平稳）
- 如果震荡: 减小学习率 (--lr)
```

### 图2: 跟踪误差 vs Horizon
```
- Y轴: Tracking Error (L2 norm)
- X轴: Horizon (1-17步)
- 期望: < 3.0 (红色虚线)
- 如果超过: 目标太远或CP不确定性过高
```

### 图3: 优化的动作序列
```
- 蓝线: Steering (-1到1)
- 橙线: Throttle (-1到1)
- 期望: 平滑变化（无抖动）
- 如果抖动: 增大 smooth_penalty
```

### 图4: CP不确定性
```
- 橙色曲线: q_t (安全半径)
- 期望: 随horizon增长
- 含义: 远期预测自动降权
```

---

## 🎛️ 调参指南

### MPC参数 (在 `conformal_mpc.py` 中修改)

```python
self.params = {
    'tracking_weight': 1.0,      # 跟踪权重（↑更aggressive追踪）
    'action_penalty': 0.01,       # 动作惩罚（↑更保守）
    'smooth_penalty': 0.1,        # 平滑惩罚（↑减少抖动）
    'conservatism': 0.05,         # 不确定性惩罚（↑更安全）
    'uncertainty_threshold': 50.0, # 高不确定性阈值
    'lr': 0.1,                    # 优化学习率（↓更稳定）
    'n_iters': 50,                # 优化迭代次数（↑更精确）
}
```

### 调参建议

#### 问题1: MPC动作抖动
```python
# 增大平滑惩罚
'smooth_penalty': 0.5  # 从0.1 → 0.5
```

#### 问题2: MPC太保守（动作过小）
```python
# 减小动作惩罚
'action_penalty': 0.001  # 从0.01 → 0.001
```

#### 问题3: MPC追不上目标
```python
# 增大跟踪权重
'tracking_weight': 5.0  # 从1.0 → 5.0
# 或减小horizon（更激进）
--horizon 10  # 从17 → 10
```

#### 问题4: 优化不收敛
```python
# 减小学习率，增加迭代
'lr': 0.05  # 从0.1 → 0.05
'n_iters': 100  # 从50 → 100
```

---

## 🔬 高级用法

### 1. 测试不同Horizon

```bash
# 短期激进（8步）
python3 test_mpc.py ... --horizon 8

# 中期平衡（17步，推荐）
python3 test_mpc.py ... --horizon 17

# 长期保守（30步）
python3 test_mpc.py ... --horizon 30
```

### 2. 批量测试多个序列

```bash
for idx in 10 50 100 200 500; do
    python3 test_mpc.py ... --test_idx $idx --output_dir mpc_test_$idx
done
```

### 3. 在Python中使用MPC

```python
from conformal_mpc import ConformalMPC

# 初始化
mpc = ConformalMPC(
    vae_model=vae,
    lstm_model=lstm,
    cp_quantiles_path='eval_results/cp_quantiles.json',
    horizon=17
)

# 控制循环
for t in range(1000):
    # 获取当前观测
    images = get_recent_images()  # [15, 3, 64, 64]
    
    # 定义目标
    z_goal = mpc.compute_goal_latent(goal_image)
    
    # 计算最优动作
    u_opt, info = mpc.control_step(images, z_goal)
    
    # 应用动作
    apply_action(u_opt)
    
    # 日志
    if t % 10 == 0:
        print(f"Step {t}: action={u_opt}, cost={info['cost_final']}")
```

---

## 📊 期望性能指标

基于你的系统（Horizon=17, MSE<3.0）:

| 指标 | 期望范围 | 单位 |
|------|----------|------|
| **优化时间** | 50-100 | ms/step (GPU) |
| **跟踪误差** | 1.5-3.0 | latent L2 |
| **动作平滑** | Δu < 0.2 | per step |
| **成本收敛** | < 50次迭代 | iterations |

---

## 🐛 常见错误

### 错误1: `KeyError: 'rollout_from_context'`
```
原因: LSTM模型缺少rollout方法
解决: 确保使用最新版本的vae_predictor.py
```

### 错误2: `RuntimeError: Expected tensor on cuda:0 but got cpu`
```
原因: 数据/模型设备不匹配
解决: 检查所有tensor都在同一设备（GPU或CPU）
```

### 错误3: `FileNotFoundError: cp_quantiles.json`
```
原因: 未运行CP评估
解决: 先运行 eval_predictor.py --cp_calibrate --cp_eval
```

---

## 📚 下一步

1. **✅ 当前**: 离线测试单步MPC
2. **🔜 接下来**: 闭环仿真（多步rollout）
3. **🚀 最终**: 真车部署（ROS集成）

详细路线图见 `SYSTEM_ANALYSIS_AND_MPC_PLAN.md`

---

## 📧 支持

遇到问题？检查：
1. `SYSTEM_ANALYSIS_AND_MPC_PLAN.md` - 完整系统分析
2. `conformal_mpc.py` - MPC核心实现
3. `test_mpc.py` - 测试脚本

**Happy MPCing! 🎮🚗💨**
