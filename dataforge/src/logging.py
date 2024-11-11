import logging
import os
import dataforge
from typing import Optional

from dataforge.src.generic import get_package_root

def get_logger(log_filename: Optional[str] = None, level = logging.INFO) -> logging.Logger:
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-7.7s]  %(message)s")
    rootLogger = logging.getLogger("dataforge")
    if rootLogger.hasHandlers():
        rootLogger.handlers.clear()
    rootLogger.setLevel(level)

    if log_filename is not None:
        log_dir = os.path.join(
            get_package_root(module_file_path = dataforge.__file__),
            'logs',
        )
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(
            log_dir,
            log_filename,
        )
        fileHandler = logging.FileHandler(log_filename, mode="w")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger