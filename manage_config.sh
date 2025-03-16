#!/bin/bash

# Simple configuration management script
# Usage: ./manage_config.sh [command] [options]

# Check for required packages
if ! python -c "import yaml" &> /dev/null; then
    echo "Installing required packages..."
    pip install pyyaml
fi

# Run the configuration manager
python -m src.tools.config.config_manager "$@"