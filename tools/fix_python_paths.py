#!/usr/bin/env python3
"""
Fix Python Path in Oblivion Project Files

This script adds the necessary Python path configuration to Python files
in specified directories to ensure they can be run directly without module import errors.
"""

import os
import re
import sys

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Directories to process
DIRECTORIES = [
    os.path.join(PROJECT_ROOT, 'src/core/hardware/optimizations'),
    os.path.join(PROJECT_ROOT, 'src/core/sdk'),
    os.path.join(PROJECT_ROOT, 'src/core/utils'),
    os.path.join(PROJECT_ROOT, 'src/core/hardware/sensors'),
    os.path.join(PROJECT_ROOT, 'src/core/neuromorphic')
]

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
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already fixed
    if 'project_root = os.path.abspath(' in content:
        print(f"Skipping already fixed file: {filepath}")
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
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed file: {filepath}")
    return True

def main():
    """Fix all Python files in the specified directories."""
    fixed_count = 0
    total_count = 0
    
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        print(f"Processing directory: {directory}")
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    total_count += 1
                    if fix_file(filepath):
                        fixed_count += 1
    
    print(f"Fixed {fixed_count} out of {total_count} Python files")

if __name__ == "__main__":
    main()