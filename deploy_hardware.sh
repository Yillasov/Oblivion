#!/bin/bash

# Simple hardware deployment script
# Usage: ./deploy_hardware.sh [hardware_type] [config_name] [algorithm_files...]

# Set default values
HARDWARE_TYPE=${1:-"loihi"}
CONFIG_NAME=${2:-"default"}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we have algorithm files
if [ $# -gt 2 ]; then
    ALGORITHM_FILES="${@:3}"
    echo "Deploying algorithms: $ALGORITHM_FILES"
else
    # Default to all neuromorphic algorithms
    ALGORITHM_FILES="$SCRIPT_DIR/src/core/neuromorphic/*.py"
    echo "No algorithms specified, using default neuromorphic algorithms"
fi

# Create deployment package
echo "Creating deployment package for $HARDWARE_TYPE with config $CONFIG_NAME..."
python -m src.tools.deployment.deploy create-package $HARDWARE_TYPE $CONFIG_NAME --algorithms $ALGORITHM_FILES

# Ask if user wants to deploy
read -p "Deploy to hardware now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Find the most recent deployment package
    LATEST_PACKAGE=$(ls -t $SCRIPT_DIR/deploy/${HARDWARE_TYPE}_${CONFIG_NAME}_*.zip | head -1)
    
    if [ -z "$LATEST_PACKAGE" ]; then
        echo "Error: No deployment package found"
        exit 1
    fi
    
    # Ask for hardware address
    read -p "Enter hardware address (leave blank for default): " HARDWARE_ADDRESS
    
    # Deploy to hardware
    if [ -n "$HARDWARE_ADDRESS" ]; then
        echo "Deploying to hardware at $HARDWARE_ADDRESS..."
        python -m src.tools.deployment.deploy deploy "$LATEST_PACKAGE" --address "$HARDWARE_ADDRESS"
    else
        echo "Deploying to hardware with default address..."
        python -m src.tools.deployment.deploy deploy "$LATEST_PACKAGE"
    fi
else
    echo "Deployment package created but not deployed"
fi

echo "Done"