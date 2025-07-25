#!/usr/bin/env python3
"""Fix imports in legacy test files to use top-level imports."""

import os
import re
from pathlib import Path

def fix_test_imports(filepath):
    """Fix imports in a single test file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace direct module imports with top-level imports
    patterns = [
        # Direct module imports
        (r'from complexplorer\.domain import', 'from complexplorer import'),
        (r'from complexplorer\.cmap import', 'from complexplorer import'),
        (r'from complexplorer\.funcs import', 'from complexplorer import'),
        (r'from complexplorer\.plots_2d import', 'from complexplorer import'),
        (r'from complexplorer\.plots_3d import', 'from complexplorer import'),
        (r'from complexplorer\.plots_3d_pyvista import', 'from complexplorer import'),
        (r'from complexplorer\.utils import', 'from complexplorer import'),
        (r'from complexplorer\.mesh_utils import', 'from complexplorer import'),
        (r'from complexplorer\.stl_export import', 'from complexplorer import'),
        # Import module itself
        (r'import complexplorer\.domain', 'import complexplorer'),
        (r'import complexplorer\.cmap', 'import complexplorer'),
        (r'import complexplorer\.funcs', 'import complexplorer'),
        (r'import complexplorer\.plots_2d', 'import complexplorer'),
        (r'import complexplorer\.plots_3d', 'import complexplorer'),
        (r'import complexplorer\.plots_3d_pyvista', 'import complexplorer'),
        (r'import complexplorer\.utils', 'import complexplorer'),
        (r'import complexplorer\.mesh_utils', 'import complexplorer'),
        (r'import complexplorer\.stl_export', 'import complexplorer'),
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
    test_dir = Path('tests/legacy')
    
    # Fix all Python test files
    for py_file in test_dir.glob('test_*.py'):
        fix_test_imports(py_file)

if __name__ == '__main__':
    main()