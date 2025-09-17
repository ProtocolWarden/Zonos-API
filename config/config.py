import os
from pathlib import Path

# Assuming this script is in the src/config directory
BASE_DIR = (
    Path(__file__).resolve().parent.parent
)  # Go up two levels to reach the root direct

# ///// Logging
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")

DEFAULT_LOG_DIR = Path(BASE_DIR, "log", "local")
LEGACY_LOG_DIR = Path(BASE_DIR, "logs", "local")

_env_log_dir = os.environ.get("LOG_DIR")
if _env_log_dir:
    LOG_DIR = str(Path(_env_log_dir))
else:
    # Prefer the corrected "log/" directory structure while preserving
    # backwards compatibility with deployments that still have "logs/".
    LOG_DIR = str(DEFAULT_LOG_DIR)
    if LEGACY_LOG_DIR.exists() and not DEFAULT_LOG_DIR.exists():
        LOG_DIR = str(LEGACY_LOG_DIR)

LOGGING_DEBUG = os.environ.get("LOGGING_DEBUG", "True").lower() in ("true", "1", "yes")
