"""Structured and rotating logging system for Project NOIR."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class NoirFormatter(logging.Formatter):
    """Custom formatter with timestamp, module, and level formatting."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)
        module = record.name.split(".")[-1]
        msg = record.getMessage()
        if record.exc_info:
            text = super().format(record)
            return f"[{timestamp}] [{level}] [{module}] {text}"
        return f"[{timestamp}] [{level}] [{module}] {msg}"


def setup_logging(
    log_dir: str | Path = "logs",
    log_level: str = "INFO",
    log_file: str = "noir.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """Configure structured console and rotating file logging.

    Args:
        log_dir: Directory where log files are stored.
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Base name of the rotating log file.
        max_bytes: Maximum size in bytes before rotating.
        backup_count: Number of rotated log archives to keep.
        console: Whether to attach a stdout console handler.

    Returns:
        The root 'noir' logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger("noir")
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = NoirFormatter()

    # Rotating file handler
    full_log_file = log_path / log_file
    file_handler = RotatingFileHandler(
        full_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    root_logger.info("Project NOIR logging initialized. File: %s", full_log_file)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger namespaced under 'noir.<name>'."""
    if name.startswith("noir.") or name == "noir":
        return logging.getLogger(name)
    return logging.getLogger(f"noir.{name}")
