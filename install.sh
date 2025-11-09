#!/usr/bin/env bash

# Installation script for CPOM Software v2 on linux/macos

# Save the current -e setting
old_opts=$(set +o | grep errexit)

set -e  # Exit on any error

setup_and_run_file=./activate.sh

export CPOM_SOFTWARE_DIR=$PWD

echo -e "\nCPOM Software v2 Installation\n"

cp ./.template_for_activation.sh $setup_and_run_file

# Ensure the output script is executable
chmod +x $setup_and_run_file

if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is already installed."
else
    echo "Please first install Python 3.12 on your system"
    exit 1
fi

curl -sSL https://install.python-poetry.org | python3 -


# Make sure we’re not inside any active venv/conda env
[[ -n "${VIRTUAL_ENV-}" ]] && deactivate || true
if [[ -n "${CONDA_PREFIX-}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda deactivate || true
fi
unset VIRTUAL_ENV

# Force in-project .venv for Poetry 2.x
export POETRY_VIRTUALENVS_CREATE=true
export POETRY_VIRTUALENVS_IN_PROJECT=true
poetry config --local virtualenvs.in-project true

# Remove any previously associated env(s) for this project
poetry env remove --all >/dev/null 2>&1 || true

# Create a fresh env with the exact interpreter you want
poetry env use "$(command -v python3.12)"

# Install deps into the new .venv
poetry lock
poetry install

# Sanity check: should now be the project .venv
VENV_PATH="$(poetry env info --path)"
echo "Using virtualenv: $VENV_PATH"
if [[ "$VENV_PATH" != "$PWD/.venv" ]]; then
  echo "ERROR: expected $PWD/.venv but got $VENV_PATH" >&2
  exit 1
fi

export ppath=$(poetry env info --path)

echo "export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:\$PATH" >> $setup_and_run_file

export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:$PATH

# Get the version of git
git_version=$(git --version | awk '{print $3}')

# Convert the version to a comparable number
# Major version gets padded normally, while minor and patch versions are three digits
version_number=$(echo "$git_version" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')

# Define the required version (2.20.0)
required_version=$(echo "2.20.0" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')

# Debugging output to ensure proper values
echo "Detected Git version: $git_version "
echo "Required Git version: >= 2.20.0"

# Compare the versions
if [[ "$version_number" -gt "$required_version" ]]; then
    echo "Git version is greater than 2.20. Performing the task..."
    # Place your task commands here
    # Install pre-commit if not already installed
    if ! command -v pre-commit &>/dev/null; then
        echo "Installing pre-commit..."
        pip install pre-commit
    fi

    pre-commit install
    pre-commit autoupdate
else
    echo "Git version is not greater than 2.20. Skipping the task."
    echo "WARNING: git version is < 2.20. Can not install pre-commit hooks"
    echo "Please upgrade git on your system and re-run the install"
fi

# Restore the original -e setting
eval "$old_opts"

echo ""
echo "-----------------------"
echo "Installation complete. "
echo "-----------------------"
echo "Use \"source $setup_and_run_file\" to set up and activate the environment."
echo "or \". $setup_and_run_file\" is an alternative syntax."
