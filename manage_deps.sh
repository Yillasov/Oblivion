#!/bin/bash

# Simple dependency management system
# Usage: ./manage_deps.sh [command] [options]

# Set default values
DEPS_FILE="/Users/yessine/Oblivion/dependencies.json"
VENV_DIR="/Users/yessine/Oblivion/.venv"
REQUIREMENTS_FILE="/Users/yessine/Oblivion/requirements.txt"

# Function to initialize dependency tracking
init_deps() {
  if [ -f "$DEPS_FILE" ]; then
    echo "Dependencies file already exists"
  else
    # Create initial dependencies file
    cat > "$DEPS_FILE" << EOF
{
  "python": {
    "packages": [],
    "version": "$(python3 --version | cut -d' ' -f2)"
  },
  "system": {
    "libraries": []
  },
  "neuromorphic": {
    "hardware": [],
    "simulators": []
  }
}
EOF
    echo "Initialized dependencies file at $DEPS_FILE"
  fi
  
  # Create virtual environment if it doesn't exist
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
  fi
}

# Function to list dependencies
list_deps() {
  if [ ! -f "$DEPS_FILE" ]; then
    echo "Dependencies file not found. Run './manage_deps.sh init' first."
    exit 1
  fi
  
  echo "=== Oblivion Dependencies ==="
  
  # Use Python to parse and display JSON nicely
  python3 -c "import json; import sys; f=open('$DEPS_FILE'); data=json.load(f); print(json.dumps(data, indent=2)); f.close()"
  
  # Show installed Python packages
  echo -e "\nInstalled Python packages:"
  if [ -d "$VENV_DIR" ]; then
    "$VENV_DIR/bin/pip" list
  else
    pip list
  fi
}

# Function to add a dependency
add_dep() {
  if [ ! -f "$DEPS_FILE" ]; then
    echo "Dependencies file not found. Run './manage_deps.sh init' first."
    exit 1
  fi
  
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments"
    echo "Usage: ./manage_deps.sh add [type] [name] [version]"
    echo "Example: ./manage_deps.sh add python numpy 1.21.0"
    exit 1
  fi
  
  TYPE="$1"
  NAME="$2"
  VERSION="${3:-latest}"
  
  # Add to dependencies file
  case "$TYPE" in
    python)
      # Add Python package
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
if '$NAME' not in [p.split('==')[0] for p in data['python']['packages']]:
    data['python']['packages'].append('$NAME==$VERSION' if '$VERSION' != 'latest' else '$NAME')
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Added Python dependency: $NAME $VERSION')
else:
    print('Dependency already exists: $NAME')
"
      # Install the package
      if [ -d "$VENV_DIR" ]; then
        if [ "$VERSION" == "latest" ]; then
          "$VENV_DIR/bin/pip" install "$NAME"
        else
          "$VENV_DIR/bin/pip" install "$NAME==$VERSION"
        fi
      else
        if [ "$VERSION" == "latest" ]; then
          pip install "$NAME"
        else
          pip install "$NAME==$VERSION"
        fi
      fi
      ;;
    
    system)
      # Add system library
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
if '$NAME' not in data['system']['libraries']:
    data['system']['libraries'].append('$NAME')
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Added system dependency: $NAME')
else:
    print('Dependency already exists: $NAME')
"
      echo "Note: System libraries must be installed manually"
      ;;
    
    neuromorphic)
      # Add neuromorphic dependency
      if [ -z "$3" ]; then
        echo "Error: Missing hardware/simulator argument"
        echo "Usage: ./manage_deps.sh add neuromorphic [hardware|simulator] [name]"
        exit 1
      fi
      
      SUBTYPE="$2"
      NAME="$3"
      
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
if '$SUBTYPE' in data['neuromorphic'] and '$NAME' not in data['neuromorphic']['$SUBTYPE']:
    data['neuromorphic']['$SUBTYPE'].append('$NAME')
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Added neuromorphic $SUBTYPE dependency: $NAME')
else:
    print('Invalid subtype or dependency already exists: $SUBTYPE/$NAME')
"
      ;;
    
    *)
      echo "Error: Unknown dependency type: $TYPE"
      echo "Valid types: python, system, neuromorphic"
      exit 1
      ;;
  esac
}

# Function to remove a dependency
remove_dep() {
  if [ ! -f "$DEPS_FILE" ]; then
    echo "Dependencies file not found. Run './manage_deps.sh init' first."
    exit 1
  fi
  
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments"
    echo "Usage: ./manage_deps.sh remove [type] [name]"
    echo "Example: ./manage_deps.sh remove python numpy"
    exit 1
  fi
  
  TYPE="$1"
  NAME="$2"
  
  # Remove from dependencies file
  case "$TYPE" in
    python)
      # Remove Python package
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
packages = data['python']['packages']
filtered = [p for p in packages if not p.startswith('$NAME==') and p != '$NAME']
if len(filtered) < len(packages):
    data['python']['packages'] = filtered
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Removed Python dependency: $NAME')
else:
    print('Dependency not found: $NAME')
"
      # Uninstall the package
      if [ -d "$VENV_DIR" ]; then
        "$VENV_DIR/bin/pip" uninstall -y "$NAME"
      else
        pip uninstall -y "$NAME"
      fi
      ;;
    
    system)
      # Remove system library
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
libraries = data['system']['libraries']
if '$NAME' in libraries:
    libraries.remove('$NAME')
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Removed system dependency: $NAME')
else:
    print('Dependency not found: $NAME')
"
      ;;
    
    neuromorphic)
      # Remove neuromorphic dependency
      if [ -z "$3" ]; then
        echo "Error: Missing hardware/simulator argument"
        echo "Usage: ./manage_deps.sh remove neuromorphic [hardware|simulator] [name]"
        exit 1
      fi
      
      SUBTYPE="$2"
      NAME="$3"
      
      python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
if '$SUBTYPE' in data['neuromorphic'] and '$NAME' in data['neuromorphic']['$SUBTYPE']:
    data['neuromorphic']['$SUBTYPE'].remove('$NAME')
    with open('$DEPS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Removed neuromorphic $SUBTYPE dependency: $NAME')
else:
    print('Dependency not found: $SUBTYPE/$NAME')
"
      ;;
    
    *)
      echo "Error: Unknown dependency type: $TYPE"
      echo "Valid types: python, system, neuromorphic"
      exit 1
      ;;
  esac
}

# Function to export dependencies to requirements.txt
export_deps() {
  if [ ! -f "$DEPS_FILE" ]; then
    echo "Dependencies file not found. Run './manage_deps.sh init' first."
    exit 1
  fi
  
  # Extract Python packages to requirements.txt
  python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
with open('$REQUIREMENTS_FILE', 'w') as f:
    for package in data['python']['packages']:
        f.write(package + '\n')
print('Exported Python dependencies to $REQUIREMENTS_FILE')
"
}

# Function to install all dependencies
install_deps() {
  if [ ! -f "$DEPS_FILE" ]; then
    echo "Dependencies file not found. Run './manage_deps.sh init' first."
    exit 1
  fi
  
  # Export to requirements.txt first
  export_deps
  
  # Install Python packages
  if [ -d "$VENV_DIR" ]; then
    echo "Installing Python dependencies in virtual environment..."
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
  else
    echo "Installing Python dependencies..."
    pip install -r "$REQUIREMENTS_FILE"
  fi
  
  # List system dependencies that need manual installation
  echo -e "\nSystem dependencies (install manually):"
  python3 -c "
import json
with open('$DEPS_FILE', 'r') as f:
    data = json.load(f)
for lib in data['system']['libraries']:
    print('- ' + lib)
"
}

# Parse command
case "$1" in
  init)
    init_deps
    ;;
  list)
    list_deps
    ;;
  add)
    shift
    add_dep "$@"
    ;;
  remove)
    shift
    remove_dep "$@"
    ;;
  export)
    export_deps
    ;;
  install)
    install_deps
    ;;
  *)
    echo "Oblivion Dependency Manager"
    echo "Usage: ./manage_deps.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  init                Initialize dependency tracking"
    echo "  list                List all dependencies"
    echo "  add [type] [name]   Add a dependency"
    echo "  remove [type] [name] Remove a dependency"
    echo "  export              Export to requirements.txt"
    echo "  install             Install all dependencies"
    echo ""
    echo "Dependency Types:"
    echo "  python              Python packages"
    echo "  system              System libraries"
    echo "  neuromorphic        Neuromorphic hardware/simulators"
    ;;
esac