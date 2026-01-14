"""
自动更新重组后的导入路径
运行: python update_imports.py
"""

import re
from pathlib import Path

def update_imports_in_file(file_path: Path, import_rules: dict):
    """更新单个文件的导入语句"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # 应用所有导入规则
        for old_import, new_import in import_rules.items():
            # 支持多种导入格式
            patterns = [
                (f'from {old_import} import', f'from {new_import} import'),
                (f'import {old_import}', f'import {new_import}'),
            ]
            
            for old, new in patterns:
                if old in content:
                    content = content.replace(old, new)
        
        # 如果有修改，写回文件
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def update_all_imports():
    """更新所有文件的导入路径"""
    
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("更新导入路径")
    print("="*70)
    
    # 定义导入替换规则
    import_rules = {
        # 核心模块
        'vae_predictor': 'predictor.core.vae_predictor',
        'train_predictor': 'predictor.core.train_predictor',
        
        # 评估模块
        'eval_predictor': 'predictor.evaluation.eval_predictor',
        'eval_metrics': 'predictor.evaluation.eval_metrics',
        'eval_utils': 'predictor.evaluation.eval_utils',
        'eval_visualization': 'predictor.evaluation.eval_visualization',
        'eval_conformal': 'predictor.evaluation.eval_conformal',
        'eval_cp_2d': 'predictor.evaluation.eval_cp_2d',
        'eval_cp_safety': 'predictor.evaluation.eval_cp_safety',
        
        # MPC模块
        'conformal_mpc': 'predictor.mpc.conformal_mpc',
        
        # CP模块
        'conformal': 'predictor.conformal.conformal',
    }
    
    # 需要更新的目录
    target_dirs = ['mpc', 'evaluation', 'core']
    
    updated_files = []
    
    for target_dir in target_dirs:
        dir_path = base_dir / target_dir
        if not dir_path.exists():
            print(f"\n⚠️  目录不存在，跳过: {target_dir}/")
            continue
        
        print(f"\n[{target_dir}/]")
        
        # 遍历所有Python文件
        for py_file in dir_path.glob('*.py'):
            if py_file.name.startswith('__'):
                continue
            
            if update_imports_in_file(py_file, import_rules):
                print(f"  ✓ {py_file.name}")
                updated_files.append(str(py_file.relative_to(base_dir)))
            else:
                print(f"  - {py_file.name} (无需更新)")
    
    # 总结
    print("\n" + "="*70)
    if updated_files:
        print(f"✅ 已更新 {len(updated_files)} 个文件:")
        for f in updated_files:
            print(f"  - {f}")
    else:
        print("ℹ️  没有文件需要更新")
    print("="*70)
    
    print("\n⚠️  重要提示：")
    print("1. 请手动验证关键文件的导入是否正确")
    print("2. 运行 python -m pytest tests/ 测试")
    print("3. 如果有问题，查看 docs/REORGANIZE_STRUCTURE.md")


if __name__ == '__main__':
    try:
        update_all_imports()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
