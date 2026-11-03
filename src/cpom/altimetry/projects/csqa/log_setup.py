"""cpom.altimetry.projects.csqa.log_setup

Logging configuration of the CSQA tools, shared by the main process and the cycle and plot
worker processes, and the initialisation of worker processes (init_worker).
"""

import logging
import os
import threading
import time

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s"


def setup_logging(level: int, log_file: str | None = None):
    """configure logging to stderr and optionally a log file

    Args:
        level (int): logging level
        log_file (str|None): also log to this file (appended)
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    # quieten verbose third party / plotting modules
    for name in ("matplotlib", "PIL", "cpom.areas", "cpom.backgrounds", "fiona", "rasterio"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


def exit_with_parent(poll_seconds: float = 5.0):
    """End this (worker) process when its parent process ends, so workers do not carry on
    processing (and writing outputs) after the main process is killed. Starts a daemon thread
    checking the parent process id every poll_seconds

    Args:
        poll_seconds (float): seconds between checks
    """
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(poll_seconds)
            if os.getppid() != parent:  # re-parented: the parent ended
                os._exit(1)  # pylint: disable=protected-access

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()


def init_worker(level: int, log_file: str | None = None):
    """initializer of worker processes: log like the main process (setup_logging) and end
    with the parent process (exit_with_parent)"""
    setup_logging(level, log_file)
    exit_with_parent()


def current_logging_config() -> tuple[int, str | None]:
    """(level, log file) of the current process's root logger, to configure worker processes
    the same way with setup_logging()"""
    root = logging.getLogger()
    log_file = next(
        (h.baseFilename for h in root.handlers if isinstance(h, logging.FileHandler)), None
    )
    return root.level, log_file
