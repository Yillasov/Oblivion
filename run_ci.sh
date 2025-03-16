#!/bin/bash

# Simple CI runner script
echo "Running Oblivion CI Pipeline"
python -m src.core.testing.ci_pipeline --config ci_config.json
exit $?