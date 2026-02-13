import logging
import os
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR


def get_logger(logger_name: str, log_file: str):

    """
    Create and return a configured logger instance.
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    log_format = (
        "%(asctime)s | %(name)s | %(levelname)s | "
        "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
    )

    formatter = logging.Formatter(log_format)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
