"""The logger package must not set every logger in the process to DEBUG.

logger_helper.py installs CustomLogger as the logging class for the whole process, so a logger that any library
creates after the import is a CustomLogger with CustomLogger's default level. At DEBUG, torch's compile loggers wrote
287,762 lines into two days of zonos-tts-api's log (VideoFoundry, 2026-11-03). At NOTSET a new logger takes its
parent's level, so a library's own level, or the root's, decides.
"""

import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logger.logger_helper import CustomLogger  # noqa: E402  (the import installs CustomLogger as the logging class)


def test_a_logger_created_after_the_import_is_notset():
    log = logging.getLogger(f"third_party_{uuid.uuid4().hex}")
    assert isinstance(log, CustomLogger)
    assert log.level == logging.NOTSET


def test_a_library_level_holds_for_a_child_created_after_the_import():
    parent = logging.getLogger(f"library_{uuid.uuid4().hex}")
    parent.setLevel(logging.ERROR)
    child = logging.getLogger(f"{parent.name}.compile")
    assert child.getEffectiveLevel() == logging.ERROR
    assert not child.isEnabledFor(logging.DEBUG)
