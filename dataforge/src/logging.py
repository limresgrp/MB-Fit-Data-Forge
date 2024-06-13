import logging
from typing import Optional

def get_logger(log_filename: Optional[str] = None) -> logging.Logger:
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger("dataforge")
    if rootLogger.hasHandlers():
        rootLogger.handlers.clear()
    rootLogger.setLevel(logging.INFO)

    if log_filename is not None:
        fileHandler = logging.FileHandler(log_filename, mode="w")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger