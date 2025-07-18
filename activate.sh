#!/usr/bin/env bash

# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed!"
    echo "Please run: source ./ct_activate.sh"
    exit 1
fi
# Combined setup and run script for CryoTEMPO LI
old_opts=$(set +o | grep errexit)
set -e

# Activate Poetry virtual environment
VENV_PATH=$(poetry env info --path)
if [ -z "$VENV_PATH" ]; then
    echo "Poetry virtual environment not found. Did you set it up?"
    exit 1
fi
source "$VENV_PATH/bin/activate"

export CPOM_SOFTWARE_DIR=/home/willisc3/luna/CPOM/willisc3/cpom_software2
export PYTHONPATH=/home/willisc3/luna/CPOM/willisc3/cpom_software2/src
export PATH=/home/willisc3/luna/CPOM/willisc3/cpom_software2/src/clev2er/tools:${PATH}
export CPDATA_DIR=/home/willisc3/luna/CPOM/archive/

# Check if specified paths exist
declare -A path_env_map=(
    [$CPDATA_DIR]=CPDATA_DIR
)

missing_paths=()
for path in "${!path_env_map[@]}"; do
    if [ ! -d "$path" ]; then
        missing_paths+=("${path_env_map[$path]}: $path")
    fi
done

if [ ${#missing_paths[@]} -gt 0 ]; then
    echo "WARNING: The following environment variables have paths that do not exist:" >&2
    for missing_path in "${missing_paths[@]}"; do
        echo "  - $missing_path" >&2
    done
fi

ulimit -n 8192
eval "set +o errexit"

echo "Environment setup complete. You are now in the CPOM Software v2 virtual environment."
