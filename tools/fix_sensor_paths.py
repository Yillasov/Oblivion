#!/usr/bin/env python3
"""
Fix Python Path in Sensor Module Files

This script adds the necessary Python path configuration to all sensor module files
to ensure they can be run directly without module import errors.
"""

import os
import re
import sys

# Directory containing sensor module files
sensors_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/simulation/sensors'))

# Python path fix to insert at the beginning of each file
PATH_FIX = '''#!/usr/bin/env python3
"""
{docstring}
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
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
        return
    
    # Extract docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    docstring = docstring_match.group(1).strip() if docstring_match else "Module description"
    
    # Remove original docstring
    if docstring_match:
        content = content.replace(docstring_match.group(0), "", 1)
    
    # Add imports if they don't exist
    imports_to_add = []
    if 'import sys' not in content:
        imports_to_add.append('import sys')
    if 'import os' not in content:
        imports_to_add.append('import os')
    
    # Create fixed content
    fixed_content = PATH_FIX.format(docstring=docstring) + content.lstrip()
    
    # Write fixed content back to file
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed file: {filepath}")

def main():
    """Fix all Python files in the sensors directory."""
    count = 0
    for root, _, files in os.walk(sensors_dir):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                fix_file(filepath)
                count += 1
    
    print(f"Fixed {count} files in {sensors_dir}")

if __name__ == "__main__":
    main()