import os
import re
from pathlib import Path

def get_imports():
    imports = set()
    project_root = Path.cwd()
    
    for py_file in project_root.rglob("*.py"):
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
            
        with open(py_file, "r", encoding="utf-8") as f:
            for line in f:
                # from x import y
                match_from = re.match(r"^from\s+([a-zA-Z0-9_]+)", line)
                if match_from:
                    imports.add(match_from.group(1))
                
                # import x
                match_import = re.match(r"^import\s+([a-zA-Z0-9_]+)", line)
                if match_import:
                    imports.add(match_import.group(1))
                    
    return sorted(list(imports))

print("Found imports:")
print("\n".join(get_imports()))
