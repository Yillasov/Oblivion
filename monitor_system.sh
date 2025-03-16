#!/bin/bash

# Simple system monitoring script
# Usage: ./monitor_system.sh [--report] [--interval SECONDS]

# Parse arguments
REPORT=false
INTERVAL=5

while [[ $# -gt 0 ]]; do
  case $1 in
    --report)
      REPORT=true
      shift
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./monitor_system.sh [--report] [--interval SECONDS]"
      exit 1
      ;;
  esac
done

# Install required packages if needed
if ! python -c "import psutil" &> /dev/null; then
    echo "Installing required packages..."
    pip install psutil
fi

# Run the monitor
if [ "$REPORT" = true ]; then
    echo "Generating system diagnostic report..."
    python -m src.tools.monitoring.system_monitor --report
else
    echo "Starting system monitoring with interval ${INTERVAL}s..."
    python -m src.tools.monitoring.system_monitor --monitor --interval "$INTERVAL"
fi