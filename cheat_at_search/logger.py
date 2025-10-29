import logging
import sys
from typing import Optional, Union, Literal

LogLevelType = Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]


def log_at(level):
    """Enable INFO level logging for the cheat_at_search package."""
    logger = logging.getLogger("cheat_at_search")
    logger.setLevel(level)
    if not logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    base_name = "cheat_at_search"
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith(base_name + "."):
            # Remove existing handlers to avoid duplicate logs
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.setLevel(level)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
    return logger


def log_to_stdout(
    logger_name: Optional[str] = None,
    level: LogLevelType = "ERROR",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Configure a logger to output to stdout with the specified log level.

    Args:
        logger_name: The name of the logger to configure. If None, the root logger is configured.
        level: The logging level. Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               or the corresponding integer values from the logging module.
        format_string: The format string for the log messages.

    Returns:
        The configured logger instance.
    """
    if logger_name and not logger_name.startswith("cheat_at_search"):
        logger_name = f"cheat_at_search.{logger_name}" if logger_name else "cheat_at_search"
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get the logger (root logger if name is None)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger
