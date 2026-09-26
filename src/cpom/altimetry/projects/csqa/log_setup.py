"""cpom.altimetry.projects.csqa.log_setup

Logging configuration of the CSQA tools, shared by the main process and the cycle and plot
worker processes.
"""

import logging

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


def current_logging_config() -> tuple[int, str | None]:
    """(level, log file) of the current process's root logger, to configure worker processes
    the same way with setup_logging()"""
    root = logging.getLogger()
    log_file = next(
        (h.baseFilename for h in root.handlers if isinstance(h, logging.FileHandler)), None
    )
    return root.level, log_file
