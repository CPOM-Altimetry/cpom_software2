"""cpom.logging_funcs.logging.py

Providing standardised functions for logging functionality.

Example :

from cpom.logging_funcs.logging import get_logger

log = get_logger(
        log_file_info=logfile,
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=logging.INFO,
        )

"""

import logging
from types import TracebackType
from typing import Type


# pylint: disable = R0913, R0917
def set_loggers(
    log_format: str = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    log_name: str = "",
    log_file_info: str = "info.log",
    log_file_error: str = "error.log",
    log_file_warning: str = "warning.log",
    log_file_debug: str = "debug.log",
    default_log_level: int = logging.INFO,
):
    """
    Setup Logging handlers
    - direct log.ERROR messages -> separate log file
    - direct log.WARNING -> separate log file
    - direct log.INFO (including log.ERROR, log.WARNING) -> separate log file
    - direct log.DEBUG (including log.ERROR, log.WARNING, log.INFO) -> separate log file
    - direct all allowed levels to stout
    - set maximum allowed log level (applies to all outputs, default is log.INFO,
    - ie no log.DEBUG messages will be included by default)

    Args:
        log_format (str) : format string to use in logger
        log_name (str) : log name, default is ""
        log_file_info (str) : file name of INFO log, default is "info.log"
        log_file_warning (str) : file name of WARNING log, default is "warning.log"
        log_file_error (str) : file name of ERROR log, default is "error.log"
        log_file_debug (str) : file name of DEBUG log, default is "debug.log"
        default_log_level(int): default log level, default is logging.INFO
    """

    log = logging.getLogger(log_name)
    log_formatter = logging.Formatter(log_format, datefmt="%d/%m/%Y %H:%M:%S")

    # log messages -> stdout (include all depending on log.setLevel(), at end of function)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(default_log_level)
    log.addHandler(stream_handler)

    # include all allowed log levels up to INFO (ie ERROR, WARNING, INFO, not DEBUG)
    file_handler_info = logging.FileHandler(log_file_info, mode="w")
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    # include all allowed log levels up to INFO (ie ERROR, WARNING, INFO, not DEBUG)
    file_handler_warning = logging.FileHandler(log_file_warning, mode="w")
    file_handler_warning.setFormatter(log_formatter)
    file_handler_warning.setLevel(logging.WARNING)
    log.addHandler(file_handler_warning)

    # only includes ERROR level messages
    file_handler_error = logging.FileHandler(log_file_error, mode="w")
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    # include all allowed log levels up to DEBUG
    if default_log_level == logging.DEBUG:
        file_handler_debug = logging.FileHandler(log_file_debug, mode="w")
        file_handler_debug.setFormatter(log_formatter)
        file_handler_debug.setLevel(logging.DEBUG)
        log.addHandler(file_handler_debug)

    log.setLevel(default_log_level)

    print("log file (INFO) :", log_file_info)
    print("log file (WARNING) :", log_file_warning)
    print("log file (ERROR):", log_file_error)
    print("log file (DEBUG):", log_file_debug)

    return log


def exception_hook(
    exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: TracebackType
) -> None:
    """Logs exception traceback output to the error log, instead of just to the console.

    Without this, these errors can be missed if the console is not checked.

    Args:
        exc_type (Type[BaseException]): The exception type.
        exc_value (BaseException): The exception instance.
        exc_traceback (TracebackType): The traceback object.

    Returns:
        None: This function does not return a value.
    """
    log = logging.getLogger("")
    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
