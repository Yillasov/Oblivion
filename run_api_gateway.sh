#!/bin/bash

# Simple API gateway script
# Usage: ./run_api_gateway.sh [--port PORT] [--example]

# Parse arguments
PORT=8080
EXAMPLE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --example)
      EXAMPLE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_api_gateway.sh [--port PORT] [--example]"
      exit 1
      ;;
  esac
done

# Install required packages if needed
if ! python -c "import requests" &> /dev/null; then
    echo "Installing required packages..."
    pip install requests
fi

# Run the API gateway
if [ "$EXAMPLE" = true ]; then
    echo "Starting API gateway on port ${PORT} with example handlers..."
    python -m src.tools.api.gateway --port "$PORT" --example
else
    echo "Starting API gateway on port ${PORT}..."
    python -m src.tools.api.gateway --port "$PORT"
fi