# CP 可视化调优指南

## 问题说明

CP band 在 2D PCA 投影中看起来很大的原因：
- **q_t** 是在高维 latent 空间（64D）计算的 L2 距离
- PCA 投影到 2D 后，距离的尺度发生了变化
- 直接用 64D 的 q_t 作为 2D 圆的半径会导致可视化失真

## 解决方案

### 方案 1: 自动缩放（已实现）✅

**代码位置**: `eval_conformal.py` 的 `visualize_cp_trajectory_band()`

**原理**:
```python
# 计算 2D 空间中的实际距离
dists_2d = ||pred_2d - gt_2d||

# 计算原始高维空间的距离
dists_highd = ||pred_64d - gt_64d||

# 缩放因子 = 2D距离 / 高维距离
scale = median(dists_2d / dists_highd)

# 应用到可视化
qt_2d = qt_highd * scale
```

**效果**:
- 自动将 q_t 从 ~88 缩放到合适的 2D 尺度（通常 0.1-10）
- 保持相对关系正确
- 无需手动调参

---

### 方案 2: 手动缩放因子

如果需要更精细的控制，可以添加命令行参数：

#### 修改 1: 添加参数

在 `eval_predictor.py` 中添加：
```python
parser.add_argument('--cp_traj_scale', type=float, default=None,
                   help='Manual scaling factor for CP band in 2D PCA space. '
                        'If None, auto-compute from data. Typical values: 0.01-0.1')
```

#### 修改 2: 传递参数

在调用 `visualize_cp_trajectory_band()` 时：
```python
visualize_cp_trajectory_band(
    ...
    scale_factor=args.cp_traj_scale,  # 新增
)
```

#### 修改 3: 使用参数

在 `eval_conformal.py` 中：
```python
def visualize_cp_trajectory_band(
    ...
    scale_factor: Optional[float] = None,  # 新增
):
    ...
    if scale_factor is None:
        # 自动计算（现有代码）
        scale = float(np.median(scale_factors))
    else:
        # 使用手动指定的值
        scale = float(scale_factor)
        print(f"[CP Vis] Using manual 2D scaling factor: {scale:.6f}")
```

#### 使用方式

```bash
python3 eval_predictor.py \
    --cp_traj_plot \
    --cp_traj_scale 0.05 \  # 手动指定缩放
    --cp_quantiles_path eval_results/cp_quantiles.json
```

**调参建议**:
- 如果圆圈太大：减小 `--cp_traj_scale`（如 0.01, 0.02）
- 如果圆圈太小：增大 `--cp_traj_scale`（如 0.1, 0.2）
- 通常范围：0.01 - 0.1

---

### 方案 3: 在 2D 空间重新校准 CP

**原理**: 完全在 2D PCA 空间中重新计算 CP 的 q_t

**优点**:
- 理论上最严格
- 2D 空间的真实 CP 保证

**缺点**:
- 需要重新收集 2D 空间的分数
- 计算开销大
- 对于可视化目的过于复杂

**实现**（如果需要）:
```python
# 1. 投影所有校准集样本到 2D
# 2. 在 2D 空间计算预测-GT 距离
# 3. 在 2D 空间计算 quantile
# 4. 保存 2D 专用的 cp_quantiles_2d.json
# 5. 可视化时使用 2D 量化值
```

**适用场景**: 研究论文需要严格的 2D CP 保证

---

## 方案选择建议

### 一般使用 → 方案 1（自动缩放）✅
- 零配置
- 效果好
- 已实现

### 需要微调 → 方案 2（手动缩放）
- 演示 PPT 时想要更美观的图
- 不同 alpha 值需要不同缩放

### 研究需要 → 方案 3（2D 重新校准）
- 论文中需要严格的 2D CP 理论保证
- 不推荐用于一般可视化

---

## 理解缩放的必要性

### 为什么需要缩放？

#### 高维空间（64D）
```
q_t = 88  →  这是 64 维空间中的 L2 球半径
              ||z_pred - z_gt||_2 ≤ 88 (在 64 维)
```

#### 2D PCA 投影
```
投影后: z_64d → z_2d (通过 PCA)
```

**关键问题**: 
- PCA 只保留了前 2 个主成分（解释方差最大的方向）
- 投影会"压缩"距离
- 64D 空间中的 88 单位，投影到 2D 后可能只对应 1-5 单位

**类比**:
- 就像把 3D 地球仪（半径 10cm）投影到 2D 地图（宽 20cm）
- 你不能说地图上画 10cm 半径的圆就等于地球仪的大小
- 需要缩放！

### 缩放因子的含义

```python
scale = 0.05  # 典型值
```

**含义**:
- 64D 空间中 1 单位距离 ≈ 2D 空间中 0.05 单位
- q_t = 88 (64D) → qt_2d = 88 × 0.05 = 4.4 (2D)
- 这样在 2D 图中画半径 4.4 的圆才合理

---

## 验证效果

运行后检查输出：

```bash
[CP Vis] 2D scaling factor: 0.0234 (high-D to 2D PCA space)
[CP] Saved trajectory band plot: eval_results/cp_band_traj_sample_0.png
```

**期望结果**:
- 缩放因子通常在 0.01 - 0.1
- CP band 应该能看清 GT 轨迹
- 大部分圆圈应该包含相应的 GT 点

---

## 常见问题

### Q1: 为什么不直接减小 alpha？

**A**: 
- alpha = 0.05 是目标覆盖率（95%）的统计保证
- 减小 alpha（如 0.01）会增大 q_t，使圆圈更大
- 这不解决可视化问题，反而更糟

### Q2: 缩放会破坏 CP 的理论保证吗？

**A**:
- 不会！
- 原始 64D 空间的 CP 保证依然有效
- 缩放只是为了 2D 可视化，不影响实际预测

### Q3: 自动缩放不准确怎么办？

**A**:
- 使用方案 2 手动指定 `--cp_traj_scale`
- 通常 0.01-0.1 之间调整
- 或者使用不同的 `--cp_traj_sample_idx` 看其他样本

---

## 总结

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| 方案 1 (自动) | 零配置，自动计算 | 可能不完美 | ⭐⭐⭐ |
| 方案 2 (手动) | 可精确控制 | 需要调参 | ⭐⭐ |
| 方案 3 (重校准) | 理论严格 | 复杂，计算量大 | ⭐ |

**推荐**: 直接使用方案 1，如有需要再考虑方案 2。
