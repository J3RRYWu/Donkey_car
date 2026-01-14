# 📂 代码重组完整指南

## 🎯 重组目标

将混乱的平铺结构：
```
predictor/
├── vae_predictor.py
├── train_predictor.py
├── eval_predictor.py
├── eval_metrics.py
├── ... (20+ 个文件)
```

重组为清晰的模块结构：
```
predictor/
├── core/              # 核心模型
├── evaluation/        # 评估模块
├── mpc/               # MPC控制
├── conformal/         # CP工具
├── docs/              # 文档
└── tests/             # 测试
```

---

## 🚀 执行步骤（3步完成）

### Step 1: 运行重组脚本（1分钟）

```bash
cd D:\donkey_car\Donkey_car\predictor  # Windows
# cd ~/Donkey_car/predictor             # Linux

python reorganize.py
```

**输出示例**：
```
======================================================================
开始重组predictor文件夹
======================================================================
✓ 创建目录: core/
✓ 创建目录: evaluation/
✓ 创建目录: mpc/
...
✅ 重组完成！
```

**完成后的结构**：
```
predictor/
├── core/
│   ├── vae_predictor.py
│   ├── train_predictor.py
│   └── __init__.py
├── evaluation/
│   ├── eval_predictor.py
│   ├── eval_*.py (7个文件)
│   └── __init__.py
├── mpc/
│   ├── conformal_mpc.py
│   ├── test_mpc*.py (3个文件)
│   └── __init__.py
├── conformal/
│   ├── conformal.py
│   └── __init__.py
├── docs/
│   └── *.md, *.txt (10个文档)
├── tests/
│   └── test_import.py
├── checkpoints/
├── __init__.py
└── README.md
```

---

### Step 2: 更新导入路径（1分钟）

```bash
python update_imports.py
```

**输出示例**：
```
======================================================================
更新导入路径
======================================================================
[mpc/]
  ✓ conformal_mpc.py
  ✓ test_mpc.py
  ...
✅ 已更新 8 个文件
```

**自动更新的导入**：
```python
# 之前
from vae_predictor import VAEPredictor
from conformal import conformal_quantile

# 更新后
from predictor.core.vae_predictor import VAEPredictor
from predictor.conformal.conformal import conformal_quantile
```

---

### Step 3: 验证和提交（2分钟）

```bash
# 1. 查看变更
git status

# 2. 测试导入（可选）
python -c "from predictor.core.vae_predictor import VAEPredictor; print('✓ 导入成功')"
python -c "from predictor.mpc.conformal_mpc import ConformalMPC; print('✓ 导入成功')"

# 3. 暂存所有变更
git add -A

# 4. 提交
git commit -m "refactor: reorganize predictor into modular structure

- Split into core, evaluation, mpc, conformal, docs modules
- Move all documentation to docs/
- Update import paths automatically
- Add module-level __init__.py files
- Add comprehensive README.md"

# 5. 推送（可选）
git push origin main
```

---

## 📊 重组前后对比

### 文件数量

| 目录 | 之前（根目录） | 之后（子目录） |
|------|---------------|---------------|
| **Python文件** | 20个 | 20个（不变） |
| **文档文件** | 10个 | 10个 |
| **组织方式** | 平铺 | 5个子模块 |

### 可维护性

| 指标 | 之前 | 之后 |
|------|------|------|
| **清晰度** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **文档管理** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **新手友好** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔧 手动调整（如果自动脚本失败）

### 如果`reorganize.py`失败

**手动创建目录**：
```bash
mkdir core evaluation mpc conformal docs tests
```

**手动移动文件**（Windows PowerShell）：
```powershell
Move-Item vae_predictor.py core\
Move-Item train_predictor.py core\
Move-Item eval_*.py evaluation\
# ... 依此类推
```

### 如果`update_imports.py`失败

**手动更新关键文件**：

**1. `mpc/conformal_mpc.py` (约第1行)**
```python
# 添加
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 或者直接改导入
# from conformal import ... 
# 改为
from predictor.conformal.conformal import ...
```

**2. `mpc/test_mpc.py` (约第10行)**
```python
from predictor.core.vae_predictor import load_model
from predictor.mpc.conformal_mpc import ConformalMPC
```

**3. `evaluation/eval_predictor.py` (约第5行)**
```python
from predictor.core.vae_predictor import VAEPredictor
from predictor.conformal.conformal import conformal_quantile
```

---

## ✅ 验证清单

重组完成后，检查以下项：

- [ ] 所有文件都移动到了正确的子目录
- [ ] 每个子目录都有`__init__.py`
- [ ] 根目录有`README.md`
- [ ] 文档都在`docs/`目录下
- [ ] 可以成功导入核心模块：
  ```bash
  python -c "from predictor.core.vae_predictor import VAEPredictor"
  python -c "from predictor.mpc.conformal_mpc import ConformalMPC"
  ```
- [ ] Git status显示文件移动（不是删除+新增）
- [ ] 提交信息清晰

---

## 🆘 常见问题

### Q1: 导入报错 `ModuleNotFoundError`

**解决**：
```python
# 在脚本开头添加
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Q2: Git显示大量删除/新增而不是移动

**解决**：
```bash
# 使用git mv而不是直接move
git mv vae_predictor.py core/vae_predictor.py
```

### Q3: 想恢复原来的结构

**解决**：
```bash
git reset --hard HEAD  # 撤销所有未提交的更改
```

---

## 📝 重组后的使用方式

### 训练模型
```bash
# 之前
python train_predictor.py --help

# 现在
python core/train_predictor.py --help
# 或者
python -m predictor.core.train_predictor --help
```

### 评估模型
```bash
# 之前
python eval_predictor.py --help

# 现在
python evaluation/eval_predictor.py --help
# 或者
python -m predictor.evaluation.eval_predictor --help
```

### 测试MPC
```bash
# 之前
python test_mpc.py --help

# 现在
python mpc/test_mpc.py --help
# 或者
python -m predictor.mpc.test_mpc --help
```

### 查看文档
```bash
# 之前
cat MPC_QUICKSTART.md

# 现在
cat docs/MPC_QUICKSTART.md
```

---

## 🎯 完成！

重组后，你的代码结构将更加专业和易于维护！

**下一步**：
1. ✅ 提交代码
2. ✅ 通知团队成员（如果有）
3. ✅ 更新CI/CD配置（如果有）
4. ✅ 开始写论文！📚

---

**需要帮助？** 查看 `docs/` 目录下的其他文档。
