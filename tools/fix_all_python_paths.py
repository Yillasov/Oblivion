#!/usr/bin/env python3
"""
Fix Python Path in All Oblivion Project Files

This script adds the necessary Python path configuration to all Python files
in the project to ensure they can be run directly without module import errors.
"""

import os
import re
import sys

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Python path fix to insert at the beginning of each file
PATH_FIX = '''#!/usr/bin/env python3
"""
{docstring}
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '{rel_path}'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

'''

def fix_file(filepath):
    """Add Python path fix to a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already fixed
        if 'project_root = os.path.abspath(' in content:
            print(f"Skipping already fixed file: {filepath}")
            return False
        
        # Skip if it's not a Python module (no imports from src)
        if 'from src.' not in content and 'import src.' not in content:
            return False
        
        # Extract docstring
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else "Module description"
        
        # Remove original docstring
        if docstring_match:
            content = content.replace(docstring_match.group(0), "", 1)
        
        # Calculate relative path to project root
        rel_path = os.path.relpath(PROJECT_ROOT, os.path.dirname(filepath))
        
        # Create fixed content
        fixed_content = PATH_FIX.format(docstring=docstring, rel_path=rel_path) + content.lstrip()
        
        # Write fixed content back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed file: {filepath}")
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return False

def main():
    """Fix all Python files in the project."""
    fixed_count = 0
    total_count = 0
    skipped_dirs = ['.git', '.github', 'venv', 'env', '__pycache__']
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip directories that should be ignored
        dirs[:] = [d for d in dirs if d not in skipped_dirs]
        
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                total_count += 1
                if fix_file(filepath):
                    fixed_count += 1
    
    print(f"Fixed {fixed_count} out of {total_count} Python files")

if __name__ == "__main__":
    main()