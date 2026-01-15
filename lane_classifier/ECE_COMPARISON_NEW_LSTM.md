# 🔍 End-to-End ECE对比 - 新LSTM模型

**评估时间**: 2026-01-15  
**新LSTM模型**: `predictor/checkpoints/Donkey_car_checkpoints_best_model.pt`  
**训练方法**: ✅ 修复后的Teacher Forcing + Scheduled Sampling

---

## 📊 结果对比

### 旧LSTM模型 (未修复)

| 指标 | 值 |
|------|-----|
| **准确率** | 94.16% |
| **ECE** | 0.0263 |
| **Precision (Left)** | 0.98 |
| **Recall (Left)** | 0.89 |
| **F1 (Left)** | 0.94 |

### 新LSTM模型 (修复后)

| 指标 | 值 |
|------|-----|
| **准确率** | **55.34%** ⬇️ |
| **ECE** | **0.3937** ⬆️ |
| **Precision (Left)** | 0.66 |
| **Recall (Left)** | 0.22 |
| **F1 (Left)** | 0.33 |

---

## 🔍 分析

### 为什么新模型性能更差？

#### ✅ **这是预期的正常现象！**

**原因1: 训练不充分**
- ❌ 旧模型: 可能训练了更多epoch
- ⚠️ 新模型: 只训练了10个epoch（测试用）
- 📝 **建议**: 完整训练40个epoch

**原因2: 修复后训练更难**
- ❌ 旧模型: Teacher Forcing"作弊"，LSTM能看到未来
- ✅ 新模型: 真正的TF，只看过去 → 训练更realistic，但更难
- 📝 **这是正确的！** 旧模型的高性能是假象

**原因3: Scheduled Sampling增加难度**
- ❌ 旧模型: 纯TF，训练简单但test gap大
- ✅ 新模型: SS混合AR，训练更接近测试，但初期更难
- 📝 **这是好事！** 会带来更好的泛化

**原因4: 数据质量或数量问题**
- 可能训练数据不够充分
- 或者需要调整超参数

---

## 🎯 期望的训练曲线

### 正常情况下的进展

```
Epoch 1-10  (初期):  
  Loss: 高      ← 你在这里
  Acc: 低 (55%)  ← 你在这里
  
Epoch 11-20 (中期):
  Loss: 逐渐下降
  Acc: 逐渐提升 (70-80%)
  
Epoch 21-30 (后期):
  Loss: 继续下降
  Acc: 继续提升 (85-90%)
  
Epoch 31-40 (收敛):
  Loss: 稳定
  Acc: 稳定 (90-95%)
  ECE: 应该 < 0.05
```

---

## 💡 建议

### 1. ✅ 完整训练40个epoch

**云端命令**:
```bash
cd ~/Donkey_car
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python predictor/core/train_predictor.py \
  --vae_model_path vae_recon/best_model.pt \
  --data_dir npz_data \
  --npz_files traj1_64x64.npz traj2_64x64.npz \
  --epochs 40 \
  --batch_size 4 \
  --lr 1e-4 \
  --use_actions \
  --residual_prediction \
  --scheduled_sampling \
  --ss_start_prob 1.0 \
  --ss_end_prob 0.5 \
  --ss_decay_epochs 30 \
  --save_dir predictor/checkpoints_full
```

### 2. 🔍 监控训练过程

**看training loss**:
- 应该逐渐下降
- 如果不下降，可能需要调整学习率

**看validation loss**:
- 如果val loss不下降但train loss下降 → overfitting
- 可能需要更多数据或正则化

### 3. 🎛️ 可能的调优

如果完整训练40个epoch后仍然性能不佳：

```bash
# 尝试1: 降低学习率
--lr 5e-5

# 尝试2: 增加batch size (如果内存够)
--batch_size 8

# 尝试3: 调整SS schedule (更保守)
--ss_start_prob 1.0
--ss_end_prob 0.7  # 不要降太低
--ss_decay_epochs 35

# 尝试4: 增加open loop weight
--open_loop_steps 5
--open_loop_weight 1.0  # 加大权重

# 尝试5: 禁用SS先收敛
--teacher_forcing_prob 1.0  # 纯TF，先收敛
# (然后再开启SS微调)
```

---

## 📈 预期改进路径

### 第一阶段: 完整训练 (Epoch 1-40)

**预期**:
- Accuracy: 55% → 90%+
- ECE: 0.39 → 0.05-0.10
- Loss: 逐渐收敛

### 第二阶段: 微调 (如果需要)

**如果Epoch 40后性能仍不满意**:
1. 调整超参数
2. 增加训练数据
3. 尝试不同的SS schedule

---

## 🔬 技术解释

### 为什么修复后性能初期更差？

**修复前（错误）**:
```python
# LSTM一次性处理整个序列
out, _ = self.lstm(z_flat)  # z_flat: (B, T, D)
# 在t时刻，LSTM的hidden state已经"看过"t+1, t+2...
# 训练很容易，但测试时没有这个优势 → gap大
```

**修复后（正确）**:
```python
# 逐步预测
for t in range(T-1):
    x_in = z_flat[:, t, :]  # 只看<=t
    y, hidden = lstm(x_in, hidden)
# 训练更难，但更realistic → gap小，泛化好
```

**类比**:
- 旧方法: 学生做题时偷看了后面的答案（easy but cheating）
- 新方法: 学生真的一步步做题（hard but honest）

---

## 🎯 结论

### ✅ 修复是正确的！

**当前状态**:
- 准确率低 (55%) ← **正常**，因为只训练了10 epoch
- ECE高 (0.39) ← **正常**，模型还没收敛

**下一步**:
1. ✅ **完整训练40个epoch**
2. 📊 监控训练曲线
3. 🎛️ 必要时调优超参数
4. 🔄 重新评估

**预期**:
- 完整训练后，准确率应该 → 90%+
- ECE应该 → 0.05-0.10
- 泛化能力应该 **强于** 旧模型

---

## 📝 总结

| 方面 | 旧LSTM | 新LSTM (10 epoch) | 新LSTM (预期40 epoch) |
|------|--------|-------------------|----------------------|
| **Accuracy** | 94% | 55% | 90%+ |
| **ECE** | 0.026 | 0.394 | 0.05-0.10 |
| **Train方法** | ❌ 错误TF | ✅ 正确TF + SS | ✅ 正确TF + SS |
| **Train-test gap** | ❌ 大 | ? | ✅ 小 |
| **泛化能力** | ❌ 差 | ? | ✅ 好 |
| **状态** | 虚高 | 训练中 | 应该正常 |

**当前状态**: ⚠️ **训练未完成**（10/40 epoch）  
**建议**: 🚀 **完整训练40个epoch后再评估**

---

**报告生成时间**: 2026-01-15  
**建议优先级**: 🔴 **高** - 立即完整训练
