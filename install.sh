#!/usr/bin/env bash

# Installation script for CPOM Software v2 on linux/macos

# Save the current -e setting
old_opts=$(set +o | grep errexit)

set -e  # Exit on any error

setup_and_run_file=./activate.sh

export CPOM_SOFTWARE_DIR=$PWD

# Generate the setup_and_run.sh script
echo "#!/usr/bin/env bash" > $setup_and_run_file
echo "" >> $setup_and_run_file

echo "# Check if the script is being sourced or executed" >> $setup_and_run_file
echo 'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then' >> $setup_and_run_file
echo '    echo "ERROR: This script must be sourced, not executed!"' >> $setup_and_run_file
echo '    echo "Please run: source ./ct_activate.sh"' >> $setup_and_run_file
echo '    exit 1' >> $setup_and_run_file
echo "fi" >> $setup_and_run_file

echo "# Combined setup and run script for CryoTEMPO LI" >> $setup_and_run_file
echo 'old_opts=$(set +o | grep errexit)' >> $setup_and_run_file
echo "set -e" >> $setup_and_run_file
echo "" >> $setup_and_run_file
echo "# Activate Poetry virtual environment" >> $setup_and_run_file
echo "VENV_PATH=\$(poetry env info --path)" >> $setup_and_run_file
echo "if [ -z \"\$VENV_PATH\" ]; then" >> $setup_and_run_file
echo "    echo \"Poetry virtual environment not found. Did you set it up?\"" >> $setup_and_run_file
echo "    exit 1" >> $setup_and_run_file
echo "fi" >> $setup_and_run_file
echo "source \"\$VENV_PATH/bin/activate\"" >> $setup_and_run_file
echo "" >> $setup_and_run_file

# Export environment variables to the script
echo "export CPOM_SOFTWARE_DIR=$PWD" >> $setup_and_run_file
echo "export PYTHONPATH=$CPOM_SOFTWARE_DIR/src" >> $setup_and_run_file
echo "export PATH=${CPOM_SOFTWARE_DIR}/src/clev2er/tools:\${PATH}" >> $setup_and_run_file

echo "export CPDATA_DIR=/cpdata" >> $setup_and_run_file

# Special handling for hostname "lec-cpom"
current_hostname=$(hostname)
if [[ "$current_hostname" == "lec-cpom" ]]; then
    echo "export CPDATA_DIR=/media/luna/archive" >> $setup_and_run_file
    echo "export CPOM_SOFTWARE_DIR=/media/luna/shared/software/cpom_software" >> $setup_and_run_file
fi

# Add path existence checks
# Add path and environment variable existence checks
echo "" >> $setup_and_run_file
echo "# Check if specified paths exist" >> $setup_and_run_file
echo "declare -A path_env_map=(" >> $setup_and_run_file
echo "    [\$CPDATA_DIR]=CPDATA_DIR" >> $setup_and_run_file
echo ")" >> $setup_and_run_file
echo "" >> $setup_and_run_file
echo "missing_paths=()" >> $setup_and_run_file
echo "for path in \"\${!path_env_map[@]}\"; do" >> $setup_and_run_file
echo "    if [ ! -d \"\$path\" ]; then" >> $setup_and_run_file
echo "        missing_paths+=(\"\${path_env_map[\$path]}: \$path\")" >> $setup_and_run_file
echo "    fi" >> $setup_and_run_file
echo "done" >> $setup_and_run_file
echo "" >> $setup_and_run_file
echo "if [ \${#missing_paths[@]} -gt 0 ]; then" >> $setup_and_run_file
echo "    echo \"WARNING: The following environment variables have paths that do not exist:\" >&2" >> $setup_and_run_file
echo "    for missing_path in \"\${missing_paths[@]}\"; do" >> $setup_and_run_file
echo "        echo \"  - \$missing_path\" >&2" >> $setup_and_run_file
echo "    done" >> $setup_and_run_file
echo "fi" >> $setup_and_run_file

# Set ulimit
echo "" >> $setup_and_run_file
echo "ulimit -n 8192" >> $setup_and_run_file

echo "eval \"$old_opts\"" >> $setup_and_run_file

# Notify user the environment is ready
echo "" >> $setup_and_run_file
echo "echo \"Environment setup complete. You are now in the CPOM Software v2 virtual environment.\"" >> $setup_and_run_file

# Ensure the output script is executable
chmod +x $setup_and_run_file

# Install Python and dependencies
conda_used=0

if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is already installed."
else
    if command -v conda &>/dev/null; then
        echo "Conda is available, creating Python 3.12 environment..."
        conda create -n py312 python=3.12 -y
        conda_used=1
    else
        echo "Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
        export PATH=$HOME/miniconda/bin:$PATH
        conda init
        conda create -n py312 python=3.12 -y
        conda_used=1
    fi
fi

if [ $conda_used -eq 1 ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate py312
fi

curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create true
poetry env use python3.12
poetry lock
poetry install

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
