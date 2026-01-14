# DonkeyCar Predictor with Conformal MPC

## 📂 项目结构

```
predictor/
├── core/              # 核心模型和训练
├── evaluation/        # 评估模块
├── mpc/               # MPC控制器
├── conformal/         # Conformal Prediction
├── docs/              # 完整文档
├── tests/             # 测试工具
└── checkpoints/       # 训练好的模型
```

## 🚀 快速开始

### 1. 训练模型
```bash
python core/train_predictor.py --help
```

### 2. 评估模型
```bash
python evaluation/eval_predictor.py --help
```

### 3. 测试MPC
```bash
python mpc/test_mpc.py --help
```

## 📚 文档

详细文档请查看 `docs/` 目录：

- `docs/RUN_GUIDE.md` - 运行指南
- `docs/MPC_QUICKSTART.md` - MPC快速开始
- `docs/SYSTEM_ANALYSIS_AND_MPC_PLAN.md` - 系统分析
- `docs/CP_SAFETY_GUIDE.md` - CP安全评估

## 📊 主要功能

✅ VAE + LSTM 轨迹预测
✅ Conformal Prediction 不确定性量化
✅ Gradient-based MPC 控制器
✅ 完整的评估和可视化框架

## 🎓 引用

如果使用了这个代码，请引用：

```bibtex
@misc{donkeycar_conformal_mpc,
  title={Conformal Model Predictive Control for Autonomous Driving},
  author={Your Name},
  year={2026}
}
```
