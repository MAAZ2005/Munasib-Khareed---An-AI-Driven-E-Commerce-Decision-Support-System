"""Logging configuration and utilities for Munasib Khareed."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "munasib.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> None:
    """Configure root logger with both console and rotating file handlers.

    Args:
        level: Logging level name (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Path to log file. If None, defaults to 'logs/munasib.log'.
        log_format: Logging format string.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated backup files to retain.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    target_path = Path(log_file) if log_file else DEFAULT_LOG_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=target_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance with the given module name.

    Args:
        name: Name of module or component requesting logger.

    Returns:
        Configured logging.Logger instance.
    """
    return logging.getLogger(name)
