# Predictor 代码重组方案

## 📂 新结构

```
predictor/
├── core/                    # 核心模型和训练
│   ├── vae_predictor.py    # VAE + LSTM模型
│   ├── train_predictor.py  # 训练脚本
│   └── __init__.py
│
├── evaluation/              # 评估模块
│   ├── eval_predictor.py   # 主评估脚本
│   ├── eval_metrics.py     # 指标计算
│   ├── eval_utils.py       # 工具函数
│   ├── eval_visualization.py  # 可视化
│   ├── eval_conformal.py   # CP可视化
│   ├── eval_cp_2d.py       # 2D CP评估
│   ├── eval_cp_safety.py   # 严格CP安全评估
│   └── __init__.py
│
├── mpc/                     # MPC控制器
│   ├── conformal_mpc.py    # 核心MPC类
│   ├── test_mpc.py         # MPC测试脚本
│   ├── test_mpc_closer_goal.py      # 更近目标测试
│   ├── test_mpc_ultra_conservative.py  # 超保守测试
│   └── __init__.py
│
├── conformal/               # Conformal Prediction
│   ├── conformal.py        # CP核心函数
│   └── __init__.py
│
├── docs/                    # 文档
│   ├── README.md           # 总览
│   ├── EVAL_MODULES.md     # 评估模块说明
│   ├── RUN_GUIDE.md        # 运行指南
│   ├── CP_SAFETY_GUIDE.md  # CP安全评估指南
│   ├── CP_VIS_TUNING.md    # CP可视化调优
│   ├── MPC_QUICKSTART.md   # MPC快速开始
│   ├── SYSTEM_ANALYSIS_AND_MPC_PLAN.md  # 系统分析
│   ├── FILES_TO_COPY.txt   # 文件清单
│   └── MPC_FILES_LIST.txt  # MPC文件清单
│
├── tests/                   # 测试和辅助
│   └── test_import.py      # 导入测试
│
├── checkpoints/            # 训练好的模型权重
│   └── (现有checkpoints)
│
├── __init__.py
└── README.md               # 项目总README

## 🚚 迁移命令

### Windows (PowerShell)
```powershell
cd D:\donkey_car\Donkey_car\predictor

# 创建子目录
New-Item -ItemType Directory -Force -Path core
New-Item -ItemType Directory -Force -Path evaluation
New-Item -ItemType Directory -Force -Path mpc
New-Item -ItemType Directory -Force -Path conformal
New-Item -ItemType Directory -Force -Path docs
New-Item -ItemType Directory -Force -Path tests

# 移动核心文件
Move-Item -Path vae_predictor.py -Destination core\
Move-Item -Path train_predictor.py -Destination core\

# 移动评估文件
Move-Item -Path eval_*.py -Destination evaluation\

# 移动MPC文件
Move-Item -Path conformal_mpc.py -Destination mpc\
Move-Item -Path test_mpc*.py -Destination mpc\

# 移动CP文件
Move-Item -Path conformal.py -Destination conformal\

# 移动文档
Move-Item -Path *.md -Destination docs\
Move-Item -Path *.txt -Destination docs\

# 移动测试文件
Move-Item -Path test_import.py -Destination tests\

# 创建__init__.py
New-Item -ItemType File -Path core\__init__.py
New-Item -ItemType File -Path evaluation\__init__.py
New-Item -ItemType File -Path mpc\__init__.py
New-Item -ItemType File -Path conformal\__init__.py
New-Item -ItemType File -Path tests\__init__.py
```

### Linux/macOS
```bash
cd ~/Donkey_car/predictor

# 创建子目录
mkdir -p core evaluation mpc conformal docs tests

# 移动核心文件
mv vae_predictor.py train_predictor.py core/

# 移动评估文件
mv eval_*.py evaluation/

# 移动MPC文件
mv conformal_mpc.py mpc/
mv test_mpc*.py mpc/

# 移动CP文件
mv conformal.py conformal/

# 移动文档
mv *.md *.txt docs/

# 移动测试文件
mv test_import.py tests/

# 创建__init__.py
touch core/__init__.py
touch evaluation/__init__.py
touch mpc/__init__.py
touch conformal/__init__.py
touch tests/__init__.py
```

## ⚠️ 需要更新的导入路径

重组后，需要更新以下文件的导入语句：

### 1. `mpc/test_mpc*.py`
```python
# 旧
from vae_predictor import VAEPredictor, load_model
from conformal_mpc import ConformalMPC

# 新
from predictor.core.vae_predictor import VAEPredictor, load_model
from predictor.mpc.conformal_mpc import ConformalMPC
```

### 2. `mpc/conformal_mpc.py`
```python
# 旧
from conformal import CPQuantiles

# 新
from predictor.conformal.conformal import CPQuantiles
```

### 3. `evaluation/eval_*.py`
```python
# 旧
from vae_predictor import VAEPredictor
from conformal import conformal_quantile

# 新
from predictor.core.vae_predictor import VAEPredictor
from predictor.conformal.conformal import conformal_quantile
```

## 📝 优点

✅ 清晰的模块划分
✅ 易于维护和扩展
✅ 文档独立管理
✅ 符合Python包的最佳实践
✅ 便于CI/CD集成
