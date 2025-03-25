#!/bin/bash

# Enhanced CI runner script
echo "Running Oblivion CI Pipeline"

# Parse command line arguments
TEST_TYPE=""
CONFIG_FILE="ci_config.json"
RESULTS_DIR="test_results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --unit)
      TEST_TYPE="unit"
      shift
      ;;
    --integration)
      TEST_TYPE="integration"
      shift
      ;;
    --hardware)
      TEST_TYPE="hardware"
      shift
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --unit          Run only unit tests"
      echo "  --integration   Run only integration tests"
      echo "  --hardware      Run only hardware tests"
      echo "  --config FILE   Use specified config file"
      echo "  --results-dir DIR  Store results in specified directory"
      echo "  --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build command
CMD="python -m src.core.testing.ci_pipeline --config $CONFIG_FILE --results-dir $RESULTS_DIR"

if [ -n "$TEST_TYPE" ]; then
  CMD="$CMD --test-type $TEST_TYPE"
fi

# Run the command
echo "Executing: $CMD"
$CMD
exit $?