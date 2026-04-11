"""Compatibility helpers for older dependencies on newer Python versions."""
from __future__ import annotations

import sys
import types
import typing


def install_typing_compat_shims() -> None:
    """Backfill typing submodules removed in newer CPython releases.

    antlr4-python3-runtime 4.7.x imports ``typing.io`` directly, which breaks on
    Python 3.13. Install a minimal shim before importing antlr4-dependent code.
    """
    if "typing.io" not in sys.modules:
        typing_io = types.ModuleType("typing.io")
        typing_io.TextIO = typing.TextIO
        typing_io.BinaryIO = typing.BinaryIO
        sys.modules["typing.io"] = typing_io

