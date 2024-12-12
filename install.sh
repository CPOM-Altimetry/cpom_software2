#!/usr/bin/env bash

set -e  # Exit on any error

setup_file=./setup_env.sh

export CPOM_SOFTWARE_DIR=$PWD

echo "export CPOM_SOFTWARE_DIR=$PWD" > $setup_file

if [ -z "${PYTHONPATH}" ]; then
    echo "export PYTHONPATH=$CPOM_SOFTWARE_DIR/src" >> $setup_file
else
    echo "export PYTHONPATH=$CPOM_SOFTWARE_DIR/src:$PYTHONPATH" >> $setup_file
fi

conda_used=0

# Check if python3.12 is installed
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed at:"
    command -v python3.12
    python3.12 -V
else
    echo "Python 3.12 is not installed"
    # Check if conda is installed
    if command -v conda &>/dev/null; then
        echo "Conda is already installed, creating a new environment with Python 3.12..."
        conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."
        source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure conda is initialized
        conda activate py312
        conda_used=1
    else
        echo "Conda is not installed, installing Miniconda..."
        
        # Download Miniconda installer for Linux or macOS (adjust URL for your platform)
        MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        
        # For macOS, use the following line:
        # MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        
        # Download Miniconda installer
        wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
        
        # Make the installer executable
        chmod +x $MINICONDA_INSTALLER
        
        # Install Miniconda (non-interactively)
        ./$MINICONDA_INSTALLER -b -p $HOME/miniconda
        
        # Clean up the installer
        rm $MINICONDA_INSTALLER
        
        echo "Miniconda installed successfully."

        # Initialize Conda
        $HOME/miniconda/bin/conda init

        # Reload shell configuration without exiting the script
        source $HOME/.bashrc || source $HOME/.zshrc || true

        # Create the environment with Python 3.12
        $HOME/miniconda/bin/conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."

        source $HOME/miniconda/etc/profile.d/conda.sh
        conda activate py312
        conda_used=1
    fi
fi

# Install/reinstall Poetry using the official installer
curl -sSL https://install.python-poetry.org | python3 -

# Make sure that Poetry creates its own venv and doesn't reuse Conda
poetry config virtualenvs.create true

if [ $conda_used -eq 1 ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate py312
fi

# Set Poetry to use Python 3.12
poetry env use python3.12

# Lock and install dependencies
poetry lock
poetry install

export ppath=$(poetry env info --path)

echo "export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:\$PATH" >> $setup_file

export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:$PATH

pre-commit install
pre-commit autoupdate

echo "Installation complete!"
echo "To set up to use the CPOM Software v2:"
echo "-------------------------------------"
echo "cd $PWD"
echo "poetry shell"
echo ". setup_env.sh"
