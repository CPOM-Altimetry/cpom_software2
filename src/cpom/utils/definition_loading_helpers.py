"""Helper functions for loading definition files for areas, masks, etc."""

import importlib.util
import os
import sys


def import_module_from_file(file_path):
    """Imports a Python module from a specified file path.

    The function dynamically loads a Python module from the given file path.
    The module name is derived from the file name, excluding its extension
    and directory path. The module is also added to `sys.modules` for
    standard import access.

    Args:
        file_path (str): The file path to the Python module to be imported.

    Raises:
        ImportError: If the module cannot be imported, either due to an
                     invalid file path or a failure during execution.

    Returns:
        tuple: A tuple containing:
            - module (module): The imported module object.
            - module_name (str): The name of the imported module.
    """
    # Calculate the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    print(module_name)
    print(file_path)

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules
        spec.loader.exec_module(module)  # Execute the module
        return module, module_name

    raise ImportError(f"Cannot import module from file path: {file_path}")
