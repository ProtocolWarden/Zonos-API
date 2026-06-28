import os
from pathlib import Path

# Assuming this script is in the src/config directory
BASE_DIR = (
    Path(__file__).resolve().parent.parent
)  # Go up two levels to reach the root direct

# ///// Logging
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")

LOG_DIR = os.environ.get(
    "LOG_DIR",
    str(
        Path(
            BASE_DIR,
            "log",
            "local",
        )
    ),
)

LOGGING_DEBUG = os.environ.get("LOGGING_DEBUG", "True").lower() in ("true", "1", "yes")
