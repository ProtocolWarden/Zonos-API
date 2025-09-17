import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from config import config as cfg

LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Define a custom logger class that extends the default logger
class CustomLogger(logging.getLoggerClass()):

    def __init__(
        self,
        name: str,
        level: str = logging.DEBUG,
    ):
        super().__init__(name, level)

    def log_message(
        self,
        message: str,
        level: str = "DEBUG",
    ):
        # Log at the appropriate level
        if level.upper() == "DEBUG":
            self.debug(message)
        elif level.upper() == "INFO":
            self.info(message)
        elif level.upper() == "WARNING":
            self.warning(message)
        elif level.upper() == "ERROR":
            self.error(message)
        elif level.upper() == "CRITICAL":
            self.critical(message)
        else:
            self.debug(message)  # Default to DEBUG if level is not recognized

    def format_block_message(
        self,
        header: str,
        level: str = "DEBUG",
        **variables,
    ):
        char = "-"
        half = 25
        whole = len(header) + (half * 2) + 2

        log_content = (f"\n\n" + (char * half) + f" {header} " + (char * half))
        for var_name, value in variables.items():
            var_title = (var_name.title())  # Capitalize the variable name
            log_content += f"\n{var_title}: {value}"
        log_content += "\n" + (char * whole) + "\n"

        self.log_message(message=log_content, level=level)

    def format_divider_message(
        self,
        level: str = "DEBUG",
        **variables,
    ):
        char = "="
        length = 75

        log_content = f"\n\n{char * length}\n"
        for var_name, value in variables.items():
            log_content += (f"{var_name.upper()}: {value}\t|\t")
        log_content = log_content.rstrip('|\t')  # Remove trailing separator
        log_content += f"\n{char * length}\n"

        self.log_message(message=log_content, level=level)

    def log_json(
        self,
        json_data: dict[str, Any],
        level: str = "DEBUG",
    ) -> None:
        """
        Pretty-print and log the provided JSON object.
        """
        try:
            # Pretty-print the JSON data
            formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
            self.log_message(message=f"\n{formatted_json}", level=level)

        except (TypeError, ValueError) as e:
            # If json_data is not serializable, log the error
            self.error(f"Failed to serialize JSON data: {e}")


# Set CustomLogger as the logger class
logging.setLoggerClass(CustomLogger)


def get_logger(
    logger_name: Optional[str] = None,
) -> logging.Logger:
    if logger_name is None:
        logger_name = Path(__file__).stem  # Use helper file name if not provided

    logger = logging.getLogger(logger_name)
    logger.setLevel(LOGGING_LEVELS[cfg.LOG_LEVEL])
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
    )

    # # Console Handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)

    # Console Handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.stream = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        buffering=1,
    )
    logger.addHandler(console_handler)

    if cfg.LOGGING_DEBUG:
        # Dynamically set log file names for each driver
        log_dir = Path(cfg.LOG_DIR)  # Directory for log files, set in config
        # Ensure the directory exists so file handlers do not fail
        log_dir.mkdir(parents=True, exist_ok=True)
        info_log_file = log_dir / f"{logger_name}.info"
        error_log_file = log_dir / f"{logger_name}.err"

        # Guarantee the log files exist even before the first write attempt.
        # While RotatingFileHandler will create them on demand, touching them
        # explicitly avoids confusion when troubleshooting logging setup.
        info_log_file.touch(exist_ok=True)
        error_log_file.touch(exist_ok=True)

        # Info File Rotating Handler (rotation after 5MB, keeping 3 backups)
        info_file_handler = RotatingFileHandler(
            info_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB file size
            backupCount=3,  # Keep 3 backup log files
            encoding="utf-8",
        )
        info_file_handler.setFormatter(log_format)
        logger.addHandler(info_file_handler)

        # Error File Rotating Handler (rotation after 2MB, keeping 5 backups)
        error_file_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=2 * 1024 * 1024,  # 2MB file size
            backupCount=5,  # Keep 5 backup log files
            encoding="utf-8",
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(log_format)
        logger.addHandler(error_file_handler)

    return logger
