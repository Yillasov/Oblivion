#!/bin/bash

# Run Sensor Simulation Environment
# This script provides a convenient way to run the sensor simulation environment

# Default values
SCENARIO="example_scenario"
CREATE_EXAMPLE=false
REAL_TIME=false
OUTPUT_DIR="output/sensor_sim"
SCENARIOS_DIR="configs/scenarios"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --scenario)
      SCENARIO="$2"
      shift 2
      ;;
    --create-example)
      CREATE_EXAMPLE=true
      shift
      ;;
    --real-time)
      REAL_TIME=true
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --scenarios-dir)
      SCENARIOS_DIR="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --scenario SCENARIO      Name of scenario to run (default: example_scenario)"
      echo "  --create-example         Create example scenario"
      echo "  --real-time              Run simulation in real-time"
      echo "  --output-dir DIR         Directory to save output data (default: output/sensor_sim)"
      echo "  --scenarios-dir DIR      Directory containing scenario files (default: configs/scenarios)"
      echo "  --log-level LEVEL        Logging level (default: INFO)"
      echo "  --help                   Show this help message"
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
CMD="python3 src/tools/run_sensor_sim.py --scenario $SCENARIO --output-dir $OUTPUT_DIR --scenarios-dir $SCENARIOS_DIR --log-level $LOG_LEVEL"

if $CREATE_EXAMPLE; then
  CMD="$CMD --create-example"
fi

if $REAL_TIME; then
  CMD="$CMD --real-time"
fi

# Run the command
echo "Running sensor simulation: $CMD"
$CMD

exit $?