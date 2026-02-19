#!/usr/bin/env python3
"""
logger.py - Clean logging system for the BMD/Muscle CT analysis pipeline

Produces timestamped logs similar to the ONH_Analysis project format:
    HH:MM:SS - message
    HH:MM:SS -   Indented sub-message
    HH:MM:SS -     INFO: Status message
    HH:MM:SS -     WARN: Warning message
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class PipelineLogger:
    """
    Custom logger producing clean, timestamped output.

    Output format matches ONH_Analysis style:
    - HH:MM:SS - prefix for all messages
    - Indentation via leading spaces in message
    - INFO:/WARN: markers for status messages
    """

    def __init__(self, log_path: Optional[Path] = None, console: bool = True):
        """
        Initialize the logger.

        Args:
            log_path: Path to log file (optional)
            console: Whether to also print to console
        """
        self.log_file: Optional[TextIO] = None
        self.console = console
        self.log_path = log_path

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, "w")

    def _timestamp(self) -> str:
        """Get current timestamp in HH:MM:SS format."""
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, message: str, timestamp: bool = True):
        """Write message to log file and/or console."""
        if timestamp:
            line = f"{self._timestamp()} - {message}"
        else:
            line = message

        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

        if self.console:
            print(line)

    def log(self, message: str, indent: int = 0):
        """
        Log a message with optional indentation.

        Args:
            message: The message to log
            indent: Indentation level (each level = 2 spaces)
        """
        prefix = "  " * indent
        self._write(f"{prefix}{message}")

    def info(self, message: str, indent: int = 2):
        """Log an INFO status message."""
        prefix = "  " * indent
        self._write(f"{prefix}INFO: {message}")

    def warn(self, message: str, indent: int = 2):
        """Log a WARN status message."""
        prefix = "  " * indent
        self._write(f"{prefix}WARN: {message}")

    def error(self, message: str, indent: int = 0):
        """Log an ERROR message."""
        prefix = "  " * indent
        self._write(f"{prefix}ERROR: {message}")

    def header(self, title: str, char: str = "=", width: int = 60):
        """Log a section header with separator lines."""
        separator = char * width
        self._write(separator)
        self._write(title)
        self._write(separator)

    def subheader(self, title: str, char: str = "-"):
        """Log a sub-section header."""
        self._write(f"\n{char*3} {title} {char*3}")

    def blank(self):
        """Log a blank line."""
        self._write("", timestamp=False)

    def section(self, title: str):
        """Log a major section start."""
        self._write(f"\n--- {title} ---")

    def subject_start(self, subject_id: str, session: str = None):
        """Log start of subject processing."""
        if session:
            self.log(f"\n{subject_id}: {session}")
        else:
            self.log(f"\n{subject_id}:")

    def metric(self, name: str, value, unit: str = "", indent: int = 1):
        """Log a metric value."""
        prefix = "  " * indent
        if unit:
            self._write(f"{prefix}{name}: {value} {unit}")
        else:
            self._write(f"{prefix}{name}: {value}")

    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def setup_pipeline_logging(output_dir: Path, prefix: str = "pipeline") -> PipelineLogger:
    """
    Set up pipeline logging with timestamped log file.

    Args:
        output_dir: Base output directory (log goes in output_dir/Log/)
        prefix: Log file prefix (e.g., "pipeline", "extraction")

    Returns:
        Configured PipelineLogger instance
    """
    log_dir = output_dir / "Log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_log_{timestamp}.txt"

    logger = PipelineLogger(log_path=log_path, console=True)

    # Log header
    logger.header("ERAP CT Analysis Pipeline (BMD + Muscle)")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_path}")
    logger.header("=", width=60)

    return logger


# Convenience function for quick logging from modules
_global_logger: Optional[PipelineLogger] = None

def get_logger() -> Optional[PipelineLogger]:
    """Get the global logger instance (if set)."""
    return _global_logger

def set_logger(logger: PipelineLogger):
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger
