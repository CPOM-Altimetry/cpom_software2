"""
Automated documentation for the CPOM Software Package, hosted on GitHub at
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

## Quick Setup

This quick setup runs a script to do all the setup work.
It is recommended in most cases (tested on macos, linux)

Run the install script:

```
cd cpom_software2
./install.sh
```

This will install 

- python 3.12
- poetry
- project packages (using poetry)
- create a file ./setup_env.sh which is used to setup the correct environment variables


### Load the Virtual Environment

Now you are all setup to go. Whenever you want to run any cpom software tools you
must first activate the virtual environment as follows:

```
cd $CPOM_SOFTWARE_DIR
source activate.sh
```
or the identical
```
. activate.sh
```

# Tool List

This section provides a list of current tools available in the
cpom software v2. This is in addition to in-code usage of the modules.

| Tool | Purpose |
| plot_map.py | generic program to plot parameters from netcdf 
files(s) on cryosphere maps | 
| find_files_in_area.py | identify files containing locations within 
a cpom area mask or within a radius of a point |

# Development

## Development Process

This section details the main development processes to contribute to the CPOM software.

- Create a new feature branch

  ```
  git checkout -b yourinitials_featurename
  ```
- Create new local commits as you develop your feature

  ```
  git commit -a -m "commit description"
  ```
  
  During the git commit the automated code checks (lint, mypy, ruff, etc) should run 
  (using the pre-installed **pre-commit** tool). These must pass in order for the 
  commit to succeed. 
  If these checks do not run then your **pre-commit** setup (during installation) is not correct.
  
- push your branch to GitHub
  ```
  git push
  ```

- Create a **Pull Request (PR)** on GitHub for your new branch.
  
  This just starts a dialog on your new feature. You can continue to develop the feature with 
  additional commits and pushes.

  This will also automatically run the **GitHub Actions** tests on your full branch. 
  This runs the static code checks (as per pre-commit and also all the pytests in the full branch). 
  The success or failure (including reasons) will be reported in the PR page.

- Finally, once your feature is fully tested request a review on GitHub in your Pull Request page.

- Once the review is passed, the feature will be merged in to the main branch.

## Test Development

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

# Automatic Documentation

Documentation is automatically generated to this page 
https://cpom-altimetry.github.io/cpom_software2 
from docstrings in the code when the **main** branch is updated on GitHub. 
A few things to note:

- docstrings should use Markdown syntax for basic formatting. See 
  [markdownguide.org/basic-syntax](https://www.markdownguide.org/basic-syntax).
- within each directory there should be a __init__.py file. The docstring in these files
    are displayed as the introduction page of that module or set of modules.
- the top level page (ie this page) is in **src/cpom/__init__.py**
- you can display images within docstrings by putting the images in 
    **docs/images/**some_image.png and then in the module's docstring put
    `![](/cpom_software2/images/some_image.png "")`. Note you should do a `git add docs/images`
    within your branch as well so that the image is included. See the docstring example in
    `cpom.altimetry.tools.plot_map`
- you can create diagrams using **mermaid** syntax within the docstring. 
  See this 
  [link](https://github.blog/developer-skills/github/include-diagrams-markdown-files-mermaid/).
- **documentation is only included in the web page when the main branch is updated** so
  you won't see any update when you modify a separate branch. Your docstrings will only be
  processed after a successful pull request to the main branch.
- further info on how auto-documentation works from docstrings and more advanced syntax
  at the **pdoc** documentation: https://pdoc.dev/docs/pdoc.html#what-is-pdoc

"""
