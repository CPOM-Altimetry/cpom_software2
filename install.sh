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

# Check if Python 3.12 is installed
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed at:"
    command -v python3.12
    python3.12 -V
else
    echo "Python 3.12 is not installed"

    if command -v conda &>/dev/null; then
        echo "Conda is installed. Creating a new environment with Python 3.12..."
        conda create -n py312 python=3.12 -y
        conda_used=1
    else
        echo "Conda is not installed. Installing Miniconda..."
        
        MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
        chmod +x $MINICONDA_INSTALLER
        ./$MINICONDA_INSTALLER -b -p $HOME/miniconda
        rm $MINICONDA_INSTALLER

        # Initialize Conda
        $HOME/miniconda/bin/conda init bash
        exec bash  # Restart the shell

        # Create the environment
        $HOME/miniconda/bin/conda create -n py312 python=3.12 -y
        conda_used=1
    fi
fi

# Activate the Conda environment if used
if [ "$conda_used" -eq 1 ]; then
    activate py312
fi

# Install Poetry
curl -sSL https://install.python-poetry.org | python3.12 -

# Configure Poetry
poetry config virtualenvs.create true
poetry env use python3.12

# Lock and install dependencies
poetry lock
poetry install

# Set up the PATH
ppath=$(poetry env info --path)
echo "export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:\$PATH" >> $setup_file
export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:${ppath}/bin:$PATH

# Pre-commit setup
pre-commit install
pre-commit autoupdate

# Final message
echo "Installation complete!"
echo "To use the CPOM Software v2:"
echo "-------------------------------------"
echo "cd $PWD"
echo "poetry shell"
echo ". setup_env.sh"
