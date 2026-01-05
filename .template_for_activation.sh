#!/usr/bin/env bash

# Must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: This script must be sourced (e.g., 'source ./activate.sh')." >&2
  return 1 2>/dev/null || exit 1
fi

# Save caller options and relax locally
__old_opts="$(set +o)"
set +euo pipefail

# --- begin activation logic ---

# Require Poetry
if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found on PATH." >&2
  eval "$__old_opts"; unset __old_opts
  return 1
fi

PROJECT_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Project venv not found at $VENV_PATH. Run 'poetry install'." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

export CPOM_SOFTWARE_DIR="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PROJECT_ROOT/src/clev2er/tools:$PATH"
export CPDATA_DIR="/cpdata"

if [ ! -d $CPDATA_DIR ]; then 
echo "WARNING: Your CPDATA_DIR is set to $CPDATA_DIR which does not exist"; 
echo "Edit this activate.sh file to set the line export CPDATA_DIR=/actual/cpdata"
fi

# Check paths
missing_paths=()
for var in CPDATA_DIR; do
  val="${!var-}"        # avoid -u issues if unset
  [[ -n "$val" && ! -d "$val" ]] && missing_paths+=("$var: $val")
done

if (( ${#missing_paths[@]} > 0 )); then
  echo "WARNING: The following environment variables have paths that do not exist:" >&2
  for m in "${missing_paths[@]}"; do echo "  - $m" >&2; done
fi

# Optional: bump ulimit (ignore errors)
ulimit -n 8192 2>/dev/null || true

export PATH=${CPOM_SOFTWARE_DIR}/src/cpom/altimetry/tools:$PATH

echo "Environment setup complete. CPOM Software v2 virtual environment is active."

# --- end activation logic ---

# Restore caller options
eval "$__old_opts"
unset __old_opts