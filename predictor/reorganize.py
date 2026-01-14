"""
自动重组predictor文件夹结构
运行: python reorganize.py
"""

import os
import shutil
from pathlib import Path

def reorganize_predictor():
    """重组predictor文件夹"""
    
    # 获取当前目录
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("开始重组predictor文件夹")
    print("="*70)
    
    # 1. 创建新的子目录
    subdirs = ['core', 'evaluation', 'mpc', 'conformal', 'docs', 'tests']
    for subdir in subdirs:
        subdir_path = base_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"✓ 创建目录: {subdir}/")
    
    # 2. 定义文件移动映射
    file_moves = {
        # 核心文件 -> core/
        'core': [
            'vae_predictor.py',
            'train_predictor.py'
        ],
        # 评估文件 -> evaluation/
        'evaluation': [
            'eval_predictor.py',
            'eval_metrics.py',
            'eval_utils.py',
            'eval_visualization.py',
            'eval_conformal.py',
            'eval_cp_2d.py',
            'eval_cp_safety.py'
        ],
        # MPC文件 -> mpc/
        'mpc': [
            'conformal_mpc.py',
            'test_mpc.py',
            'test_mpc_closer_goal.py',
            'test_mpc_ultra_conservative.py'
        ],
        # CP文件 -> conformal/
        'conformal': [
            'conformal.py'
        ],
        # 文档文件 -> docs/
        'docs': [
            'EVAL_MODULES.md',
            'RUN_GUIDE.md',
            'CP_SAFETY_GUIDE.md',
            'CP_VIS_TUNING.md',
            'MPC_QUICKSTART.md',
            'SYSTEM_ANALYSIS_AND_MPC_PLAN.md',
            'FILES_TO_COPY.txt',
            'MPC_FILES_LIST.txt',
            'REORGANIZE_STRUCTURE.md'  # 把刚创建的也移过去
        ],
        # 测试文件 -> tests/
        'tests': [
            'test_import.py'
        ]
    }
    
    # 3. 移动文件
    print("\n" + "="*70)
    print("移动文件...")
    print("="*70)
    
    for target_dir, files in file_moves.items():
        print(f"\n[{target_dir}/]")
        for filename in files:
            src = base_dir / filename
            dst = base_dir / target_dir / filename
            
            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ {filename}")
                except Exception as e:
                    print(f"  ✗ {filename}: {e}")
            else:
                print(f"  - {filename} (不存在，跳过)")
    
    # 4. 创建 __init__.py 文件
    print("\n" + "="*70)
    print("创建 __init__.py...")
    print("="*70)
    
    init_files = {
        'core/__init__.py': '"""Core models and training."""\n',
        'evaluation/__init__.py': '"""Evaluation modules."""\n',
        'mpc/__init__.py': '"""Model Predictive Control with Conformal Prediction."""\n',
        'conformal/__init__.py': '"""Conformal Prediction utilities."""\n',
        'tests/__init__.py': '"""Test utilities."""\n'
    }
    
    for init_path, content in init_files.items():
        init_file = base_dir / init_path
        if not init_file.exists():
            init_file.write_text(content, encoding='utf-8')
            print(f"  ✓ {init_path}")
    
    # 5. 创建主README
    print("\n" + "="*70)
    print("创建 README.md...")
    print("="*70)
    
    readme_content = """# DonkeyCar Predictor with Conformal MPC

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
"""
    
    readme_file = base_dir / 'README.md'
    readme_file.write_text(readme_content, encoding='utf-8')
    print("  ✓ README.md")
    
    # 6. 完成
    print("\n" + "="*70)
    print("✅ 重组完成！")
    print("="*70)
    print("\n⚠️  重要提示：")
    print("1. 需要更新导入路径（详见 docs/REORGANIZE_STRUCTURE.md）")
    print("2. 建议先在测试分支运行，确认无误后再合并")
    print("3. Git会自动检测文件移动（git mv）")
    print("\n下一步：")
    print("  cd predictor")
    print("  git status  # 查看变更")
    print("  git add -A  # 暂存所有变更")
    print("  git commit -m 'refactor: reorganize predictor structure'")


if __name__ == '__main__':
    try:
        reorganize_predictor()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
