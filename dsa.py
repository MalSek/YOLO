import os
from pathlib import Path

def print_project_tree(root_path=".", max_depth=3, output_file="project_tree.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(root_path):
            level = root.replace(str(Path(root_path)), "").count(os.sep)
            if level > max_depth:
                continue
            indent = "    " * level
            f.write(f"{indent}├── {os.path.basename(root)}/\n")
            sub_indent = "    " * (level + 1)
            for file in files[:5]:  # Показываем первые 5 файлов
                f.write(f"{sub_indent}├── {file}\n")
            if len(files) > 5:
                f.write(f"{sub_indent}└── ... ({len(files)} файлов)\n")

print_project_tree("E:/Project")