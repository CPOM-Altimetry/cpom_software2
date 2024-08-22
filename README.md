# CPOM Software (v2)

CPOM Software (v2.x) aims to build/refactor the original CPOM Software Repository (https://github.com/CPOM-Altimetry/cpom_software), 
but now use more advanced software engineering techniques for testing, 
static code analysis, documentation and continous integration.

This is a software package for :

- Altimetry processing (focusing on the Cryosphere)
- Cryosphere plotting
- DEMs
- Masks
- ...
  
# What's New 

All code in the cpom (v2) repository has to be refactored to pass the automated static code analysis and pytest tests.

## Package and Dependency Management

Now uses **poetry** for dependency management instead of **conda**

## Automated Static Code Analysis

Static code analysis (linting, type checking, etc) is built in at the pre-commit, commit and GitHub CI
stages (after a pull request, or push to the main branch).

## Automated Testing

Testing is performed using the **pytest** framework. Tests are automatically run using GitHub actions during a pull request
or push to the main branch.

## Automatic Documentation

Documentation is automatically produced from docstrings in the code, and published here: 
[cpom-altimetry.github.io/cpom_software2](https://cpom-altimetry.github.io/cpom_software2/)
