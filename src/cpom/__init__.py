"""
Documentation for the CPOM Software Package, hosted on GitHub at
[github.com/CPOM-Altimetry/cpom_software2](https://github.com/CPOM-Altimetry/cpom_software2).

# Installation

## Installation of the CPOM Software Package

Note that the package has been tested on Linux and MacOS systems. Use on
other operating systems is possible but may require additional install steps.

Make sure you have *git* installed on your target system.

Clone the git public repository in to a suitable directory on your system.
This will create a directory called **/cpom_software2** in your current directory.

with https:
`git clone https://github.com/CPOM-Altimetry/cpom_software2.git`

or with ssh:
`git clone git@github.com:CPOM-Altimetry/cpom_software2.git`

or with the GitHub CLI:
`gh repo clone CPOM-Altimetry/cpom_software2`

## Shell Environment Setup

The following shell environment variables need to be set to support package
operations.

In a bash shell this might be done by adding export lines to your 
$HOME/.bashrc or $HOME/.bash_profile file.

- Set the *CPOM_SOFTWARE_DIR* environment variable to the root of the cpom software package.
- Add $CPOM_SOFTWARE_DIR/src to *PYTHONPATH*.
- Add ${CPOM_SOFTWARE_DIR}/src/cpom/tools to the *PATH*.
- Set the shell's *ulimit -n* to allow enough file descriptors to be available for
    multi-processing.

An example environment setup is shown below (the path in the first line should be
adapted for your specific directory path):

```script
export CPOM_SOFTWARE_DIR=/Users/someuser/software/cpom_software2
export PYTHONPATH=$PYTHONPATH:$CPOM_SOFTWARE_DIR/src
export PATH=${CPOM_SOFTWARE_DIR}/src/cpom/tools:${PATH}
# for multi-processing/shared mem support set ulimit
# to make sure you have enough file descriptors available
ulimit -n 8192
```

## Python Requirement

python v3.11 must be installed or available before proceeding.
A recommended minimal method of installation of python 3.11 is using Miniconda.

To install Python 3.11 using Miniconda, select the appropriate link for your operating system from:

https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/

For example, for **Linux** (select different installer for other operating systems),
download the installer and install a minimal python 3.11 installation using:

```script
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
chmod +x Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
./Miniconda3-py311_24.1.2-0-Linux-x86_64.sh

Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no] yes
```
You may need to start a new shell to refresh your environment before
checking that python 3.11 is in your path.

Check that python v3.11 is now available, by typing:

```
python -V
```

## Virtual Environment and Package Requirements

This project uses *poetry* (a dependency manager, see: https://python-poetry.org/) to manage
package dependencies and virtual envs.

First, you need to install *poetry* on your system using instructions from
https://python-poetry.org/docs/#installation. Normally this just requires running:

`curl -sSL https://install.python-poetry.org | python3 -`

You should also then ensure that poetry is in your path, such that the command

`poetry --version`

returns the poetry version number. You may need to modify your
PATH variable in order to achieve this.

To make sure poetry is setup to use Python 3.11 virtual env when in the CLEV2ER base directory

```
cd $CPOM_SOFTWARE_DIR
poetry env use $(which python3.11)
```

### Install Required Python packages using Poetry

Run the following command to install python dependencies for this project
(for info, it uses settings in pyproject.toml to know what to install)

```
cd $CPOM_SOFTWARE_DIR
poetry install
```

### Load the Virtual Environment

Now you are all setup to go. Whenever you want to run any cpom software you
must first load the virtual environment using the `poetry shell` commands.

```
cd $CPOM_SOFTWARE_DIR
poetry shell
```

# Test Development

Each module should have an associated pytest unit or integration test. Place these in
a **tests/** directory inside your module directory. ie:

```
mymodule.py
tests/test_mymodule.py
```

If your test accesses data outside the repository then you need to exclude
it from running on GitHub Actions CI. To do this just include one of the following
at the top of your test code:

`pytestmark = pytest.mark.requires_external_data`

or

`pytestmark = pytest.mark.non_core`

# Documentation

Documentation is automatically generated to this page 
https://cpom-altimetry.github.io/cpom_software2 
from docstrings in the code when the **main** branch is updated. A few things to note:

- docstrings should use Markdown syntax for basic formatting. See 
  [markdownguide.org/basic-syntax](https://www.markdownguide.org/basic-syntax).
- within each directory there should be a __init__.py file. The docstring in these files
    are displayed as the introduction page of that module or set of modules.
- the top level page (ie this page) is in **src/cpom/__init__.py**
- you can display images within docstrings by putting the images in 
    **docs/images/**some_image.png and then in the module's docstring put
    `![](/cpom_software2/images/some_image.png "")`. Note you should do a `git add docs/images`
    within your branch as well so that the image is included. See the docstring example in
    `altimetry.tools.plot_map` or `cpom.altimetry.tools.plot_map`
- you can create diagrams using **mermaid** syntax within the docstring. 
  See this 
  [link](https://github.blog/developer-skills/github/include-diagrams-markdown-files-mermaid/).
- **documentation is only included in the web page when the main branch is updated**.

"""
