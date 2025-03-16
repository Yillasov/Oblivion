#!/bin/bash

# Simple documentation generator script
# Usage: ./generate_docs.sh [--format FORMAT] [--output-dir DIR] MODULE [MODULE...]

# Parse arguments
FORMAT="markdown"
OUTPUT_DIR="/Users/yessine/Oblivion/docs"
MODULES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      MODULES+=("$1")
      shift
      ;;
  esac
done

# Check if we have modules
if [ ${#MODULES[@]} -eq 0 ]; then
  echo "No modules specified"
  echo "Usage: ./generate_docs.sh [--format FORMAT] [--output-dir DIR] MODULE [MODULE...]"
  echo "Example: ./generate_docs.sh --format html src/core src/tools"
  exit 1
fi

# Install required packages if needed
if ! python -c "import ast" &> /dev/null; then
    echo "Required packages not found"
    exit 1
fi

# Run the documentation generator
echo "Generating ${FORMAT} documentation for ${MODULES[@]} in ${OUTPUT_DIR}..."
python -m src.tools.docs.doc_generator --format "$FORMAT" --output-dir "$OUTPUT_DIR" "${MODULES[@]}"

# Check if documentation was generated
if [ $? -eq 0 ]; then
    echo "Documentation generated successfully"
    
    # Open documentation if it's HTML
    if [ "$FORMAT" = "html" ]; then
        echo "Opening documentation in browser..."
        open "${OUTPUT_DIR}/index.html"
    else
        echo "Documentation available in ${OUTPUT_DIR}"
    fi
else
    echo "Failed to generate documentation"
    exit 1
fi