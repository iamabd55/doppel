"""Block-based DCT copy-move forgery detection."""

from detector.dct.detect import detect
from detector.dct.features import extract_features
from detector.dct.matching import find_matches

__all__ = ["detect", "extract_features", "find_matches"]
