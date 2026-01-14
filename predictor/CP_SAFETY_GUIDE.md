# Conformal Prediction Safety Evaluation Guide

## 🛡️ 目标：提供可证明的安全保证

这个模块实现了**最严格的CP安全评估**，用于safety-critical应用（如自动驾驶）。

## 🎯 核心特性

### 1. Split Conformal Prediction（理论保证）
- ✅ **Finite-sample coverage guarantee**：即使样本有限也有理论保证
- ✅ **Distribution-free**：不需要假设数据分布
- ✅ **Per-horizon quantiles**：每个预测步数都有独立的量化值

### 2. Conservative CP（额外安全边际）
- ✅ **1.2x safety factor**：量化值乘以1.2，提供20%安全边际
- ✅ **Lower effective α**：实际覆盖率 > 目标覆盖率
- ✅ **适用于高风险场景**

### 3. 多重验证
- ✅ **Independent test set**：校准集和测试集完全独立
- ✅ **Per-horizon analysis**：检查每个horizon是否达标
- ✅ **Bonferroni correction**：多重假设检验校正

### 4. Worst-case分析
- ✅ **Minimum coverage across all horizons**
- ✅ **Failure horizon identification**
- ✅ **Safety margin quantification**

---

## 🚀 使用方法

### 基础命令（最严格评估）

```bash
cd ~/Donkey_car/predictor

python3 eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../vae_recon/best_model.pt \
    --data_dir ../npz_data \
    --npz_files traj1_64x64.npz \
    --cp_safety \
    --max_horizon 50 \
    --gt_from_npz \
    --cp_alpha 0.05 \
    --cp_calib_size 500
```

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--cp_safety` | 启用严格安全评估 | **必须** |
| `--cp_alpha` | 目标错误率 | `0.05` (95%覆盖) |
| `--cp_calib_size` | 校准集大小 | `500+` |
| `--max_horizon` | 最大预测步数 | `50` |
| `--gt_from_npz` | 使用NPZ中的真实future | **推荐** |

### 不同安全级别

#### 1. 标准安全（95% coverage）
```bash
python3 eval_predictor.py ... --cp_alpha 0.05
```

#### 2. 高安全（99% coverage）
```bash
python3 eval_predictor.py ... --cp_alpha 0.01
```

#### 3. 极高安全（99.9% coverage）
```bash
python3 eval_predictor.py ... --cp_alpha 0.001
```

---

## 📊 输出文件

运行后会生成以下文件：

```
eval_results/
├── cp_safety_report.json          # 完整的JSON报告
├── cp_safety_summary.txt           # 人类可读的摘要 ⭐
├── cp_quantiles_standard.json      # 标准CP量化值
└── cp_quantiles_conservative.json  # 保守CP量化值（+20%安全边际）
```

### 重点关注：`cp_safety_summary.txt`

```text
================================================================================
CONFORMAL PREDICTION SAFETY EVALUATION SUMMARY
================================================================================

Target Coverage: 0.9500 (α=0.05)

STANDARD CP:
  Mean Coverage: 0.9534
  Min Coverage:  0.9102
  Max Coverage:  0.9801
  Safety Margin: +0.0034

CONSERVATIVE CP:
  Mean Coverage: 0.9789
  Min Coverage:  0.9456

STATISTICAL TESTS:
  Empirical α: 0.0466
  Bonferroni corrected: PASSED ✅

Horizons below target: 0

RECOMMENDATION:
  ✅ SAFE: Use standard CP quantiles
```

---

## 🔍 结果解读

### 1. 安全判定标准

| 指标 | 安全 ✅ | 边缘 ⚠️ | 不安全 ❌ |
|------|---------|---------|-----------|
| **Mean Coverage** | ≥ target | ≥ target | < target |
| **Min Coverage** | ≥ target-0.05 | ≥ target-0.10 | < target-0.10 |
| **Safety Margin** | > 0 | ≈ 0 | < 0 |
| **Bonferroni Test** | PASSED | FAILED | FAILED |

### 2. 推荐方案选择

```
if Bonferroni PASSED && Safety Margin > 0:
    ✅ 使用 STANDARD CP
    → 最优性能，理论保证满足
    
elif Safety Margin ≥ 0 && Mean Coverage ≥ target:
    ⚠️ 使用 CONSERVATIVE CP
    → 性能略保守，但更安全
    
else:
    ❌ 使用 CONSERVATIVE CP + 重新校准
    → 当前模型不足以提供安全保证
    → 建议：增加校准集大小，或重新训练模型
```

---

## 🔬 理论基础

### Split Conformal Prediction Theorem

给定：
- 校准集 $\{(X_i, Y_i)\}_{i=1}^n$ i.i.d.
- 目标错误率 $\alpha \in (0, 1)$
- 非一致性分数 $s_i = ||f(X_i) - Y_i||_2$

定义量化值：
$$
\hat{q} = \text{Quantile}_{(1-\alpha)(1+1/n)}(\{s_i\}_{i=1}^n)
$$

预测集：
$$
C(X_{n+1}) = \{y : ||f(X_{n+1}) - y||_2 \leq \hat{q}\}
$$

**保证**：
$$
P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha
$$

**关键特性**：
1. ✅ **Finite-sample valid**：对任何 $n$ 都成立
2. ✅ **Distribution-free**：不需要假设分布
3. ✅ **Tight**：几乎是最优的（不能做得更好）

### Per-Horizon Extension

对于序列预测，在每个 horizon $t$ 独立计算：
$$
\hat{q}_t = \text{Quantile}_{(1-\alpha)(1+1/n)}(\{s_{i,t}\}_{i=1}^n)
$$

其中 $s_{i,t} = ||z_{i,t}^{pred} - z_{i,t}^{true}||_2$

**Coverage保证**（per horizon）：
$$
P(z_{t+1}^{true} \in C_t(X)) \geq 1 - \alpha, \quad \forall t
$$

### Conservative CP (Extra Safety)

定义：
$$
\hat{q}_t^{cons} = \beta \cdot \hat{q}_t, \quad \beta > 1
$$

**效果**：
$$
P(Y \in C^{cons}(X)) \geq 1 - \alpha/\beta > 1 - \alpha
$$

本实现使用 $\beta = 1.2$（20% safety margin）

---

## 🧪 实验建议

### 1. 基础安全检查（快速）

```bash
python3 eval_predictor.py \
    --cp_safety \
    --max_horizon 30 \
    --cp_calib_size 300 \
    --max_eval_batches 100 \
    ...
```

**用时**：~5分钟  
**目的**：快速验证CP是否可行

### 2. 完整安全评估（严格）

```bash
python3 eval_predictor.py \
    --cp_safety \
    --max_horizon 50 \
    --cp_calib_size 1000 \
    --gt_from_npz \
    ...
```

**用时**：~20分钟  
**目的**：论文级严格评估

### 3. 高安全要求（极严格）

```bash
python3 eval_predictor.py \
    --cp_safety \
    --cp_alpha 0.01 \
    --max_horizon 50 \
    --cp_calib_size 2000 \
    --gt_from_npz \
    ...
```

**用时**：~30分钟  
**目的**：safety-critical应用（自动驾驶）

---

## ❓ FAQ

### Q1: 为什么我的coverage低于target？

**可能原因**：
1. 校准集太小（增大 `--cp_calib_size`）
2. 模型预测质量太差（重新训练）
3. 数据分布偏移（检查train/test分布）

**解决方案**：
- 使用 **CONSERVATIVE CP** quantiles
- 增大校准集到 1000+
- 检查模型的有效预测horizon

### Q2: STANDARD vs CONSERVATIVE，用哪个？

| 场景 | 推荐 |
|------|------|
| **研究/论文** | STANDARD（展示最优性能） |
| **原型系统** | CONSERVATIVE（安全第一） |
| **生产部署** | CONSERVATIVE + 额外验证 |
| **Safety-critical** | CONSERVATIVE + 人工监督 |

### Q3: Per-horizon coverage为什么不同？

**正常现象**！
- 短期预测（1-10步）：通常 coverage > target
- 中期预测（10-30步）：coverage ≈ target
- 长期预测（30+步）：可能 coverage < target（模型能力不足）

**建议**：
- 识别 **effective horizon**（coverage开始下降的点）
- 只在有效范围内使用CP保证
- 超出范围需要其他安全机制（人工接管等）

### Q4: 如何提高coverage？

**方法1**：减小 α（更宽松的目标）
```bash
--cp_alpha 0.10  # 90% coverage（更容易达到）
```

**方法2**：使用Conservative CP（自动）
```bash
--cp_safety  # 自动提供conservative版本
```

**方法3**：改进模型
- 增加训练数据
- 使用更好的架构
- 改进训练策略

---

## 🎯 总结

### 为什么这个方案"滴水不漏"？

1. ✅ **理论保证**：Split CP有严格的数学证明
2. ✅ **Finite-sample**：即使数据有限也成立
3. ✅ **Distribution-free**：不依赖强假设
4. ✅ **保守估计**：1.2x safety factor
5. ✅ **多重验证**：Bonferroni correction
6. ✅ **Per-horizon**：每步独立检查
7. ✅ **Worst-case**：关注最差情况

### 使用建议

```bash
# 第一步：运行安全评估
python3 eval_predictor.py --cp_safety ...

# 第二步：查看结果
cat eval_results/cp_safety_summary.txt

# 第三步：根据推荐选择quantiles
# - 如果PASSED：用 cp_quantiles_standard.json
# - 如果FAILED：用 cp_quantiles_conservative.json

# 第四步：可视化（使用推荐的quantiles）
python3 eval_predictor.py \
    --only_cp \
    --cp_traj_plot \
    --cp_quantiles_path eval_results/cp_quantiles_conservative.json \
    ...
```

**这就是最安全、最严格的CP评估方案！** 🛡️
