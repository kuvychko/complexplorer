#!/usr/bin/env python3
"""Fix imports in legacy files to use relative imports."""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace imports from complexplorer.X to relative imports
    patterns = [
        (r'from complexplorer\.domain', 'from .domain'),
        (r'from complexplorer\.cmap', 'from .cmap'),
        (r'from complexplorer\.funcs', 'from .funcs'),
        (r'from complexplorer\.plots_2d', 'from .plots_2d'),
        (r'from complexplorer\.plots_3d', 'from .plots_3d'),
        (r'from complexplorer\.plots_3d_pyvista', 'from .plots_3d_pyvista'),
        (r'from complexplorer\.utils', 'from .utils'),
        (r'from complexplorer\.mesh_utils', 'from .mesh_utils'),
        (r'from complexplorer\.stl_export', 'from .stl_export'),
    ]
    
    for old, new in patterns:
        content = re.sub(old, new, content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed imports in {filepath}")
    else:
        print(f"No changes needed in {filepath}")

def main():
    legacy_dir = Path('complexplorer/legacy')
    
    # Fix all Python files in legacy directory
    for py_file in legacy_dir.glob('**/*.py'):
        fix_imports_in_file(py_file)

if __name__ == '__main__':
    main()