"""Doppel copy-move forgery detection package."""

from detector.detect import detect
from detector.fusion import fuse, FusionResult

__all__ = ["detect", "fuse", "FusionResult"]
