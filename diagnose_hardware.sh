#!/bin/bash

# Hardware Diagnostic Script
# Simple script to run hardware diagnostics

# Add project root to Python path
export PYTHONPATH=/Users/yessine/Oblivion:$PYTHONPATH

# Default hardware type (empty for auto-detection)
HARDWARE_TYPE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --hardware)
      HARDWARE_TYPE="$2"
      shift 2
      ;;
    --json)
      JSON_OUTPUT="--json"
      shift
      ;;
    --output)
      OUTPUT_FILE="--output $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--hardware TYPE] [--json] [--output FILE]"
      exit 1
      ;;
  esac
done

# Run the diagnostic tool
if [ -n "$HARDWARE_TYPE" ]; then
  python3 /Users/yessine/Oblivion/src/tools/diagnostics/run_diagnostics.py --hardware "$HARDWARE_TYPE" $JSON_OUTPUT $OUTPUT_FILE
else
  python3 /Users/yessine/Oblivion/src/tools/diagnostics/run_diagnostics.py $JSON_OUTPUT $OUTPUT_FILE
fi

# Exit with the same code as the Python script
exit $?