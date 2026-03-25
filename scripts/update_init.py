import os
import ast

def generate_init_for_dir(target_dir):
    """
    为指定的目录生成 __init__.py 文件
    """
    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        print(f"⚠️ 警告: 目录不存在，跳过 -> {target_dir}")
        return

    init_file_path = os.path.join(target_dir, "__init__.py")
    # 提取最后的文件夹名作为包名 (比如从 "models/layers" 提取出 "layers")
    package_name = os.path.basename(os.path.normpath(target_dir))
    
    import_statements = []
    all_exports = []
    
    print(f"🔍 正在扫描 [{package_name}] 包目录: {target_dir}...")
    for filename in sorted(os.listdir(target_dir)):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = filename[:-3]
            filepath = os.path.join(target_dir, filename)
            
            # 使用 AST 静态解析 Python 文件
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
                
            # 查找文件中的 __all__ = [...] 列表
            module_all = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, ast.List):
                                module_all = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            
            if module_all:
                import_statements.append(f"from .{module_name} import {', '.join(module_all)}")
                all_exports.extend(module_all)
                # print(f"  └── 提取自 {module_name}.py: {module_all}")

    if not all_exports:
        print(f"  └── ℹ️ 没有在该目录下找到任何带有 __all__ 的文件。")
        return

    # 开始写入 __init__.py
    print(f"📝 正在生成 {init_file_path}...")
    with open(init_file_path, "w", encoding="utf-8") as f:
        f.write(f'"""\n这是由脚本自动生成的 {package_name} 初始化文件。\n请勿手动修改此文件，如需更新请运行 scripts/generate_inits.py\n"""\n\n')
        
        f.write("\n".join(import_statements) + "\n\n")
        
        f.write("__all__ = [\n")
        for name in sorted(all_exports):
            f.write(f"    '{name}',\n")
        f.write("]\n")
        
    print(f"🎉 [{package_name}] 生成完毕！\n")


if __name__ == "__main__":
    print(f"{'='*50}")
    print("🚀 开始批量生成 __init__.py 文件")
    print(f"{'='*50}\n")
    
    # ==========================================
    # 在这里配置你需要自动生成 __init__.py 的所有目录
    # ==========================================
    TARGET_DIRS = [
        "layers",
        # 随时可以往这里添加新的包目录
    ]
    
    for directory in TARGET_DIRS:
        generate_init_for_dir(directory)
        
    print(f"{'='*50}")
    print("🏁 所有目录扫描与生成任务完成！")