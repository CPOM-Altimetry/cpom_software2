#!/usr/bin/env bash

setup_file=/tmp/test.sh


echo "export CPOM_SOFTWARE_DIR=$PWD" > $setup_file
if [ -z "${PYTHONPATH}" ]; then
    echo "export PYTHONPATH=$CPOM_SOFTWARE_DIR/src:$PYTHON_PATH" >> $setup_file
else
    echo "export PYTHONPATH=$CPOM_SOFTWARE_DIR/src" >> $setup_file
fi

echo "export PATH=$CPOM_SOFTWARE_DIR/src/cpom/altimetry/tools:$PATH" >> $setup_file


# Check if python3.12 is installed
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed at "
    command -v python3.12
    python3.12 -V
else
    echo "Python 3.12 is not installed"
    # Check if conda is installed
    if command -v conda &>/dev/null; then
        echo "Conda is already installed, creating a new environment with Python 3.12..."
        conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."
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
        
        # Initialize Conda
        $HOME/miniconda/bin/conda init
        
        # Clean up the installer
        rm $MINICONDA_INSTALLER
        
        echo "Miniconda installed successfully."

        # Create the environment with Python 3.12
        $HOME/miniconda/bin/conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."

        conda activate py312
    fi
fi


# Install Poetry using the official installer
curl -sSL https://install.python-poetry.org | python3 -

poetry config virtualenvs.create true