#!/bin/bash
# Configuration Migration Tool
# Simple script to migrate hardware configurations between different hardware types

# Set the project root directory
PROJECT_ROOT="/Users/yessine/Oblivion"

# Run the migration tool
python3 "${PROJECT_ROOT}/src/tools/config/migrate_config.py" "$@"