"""Shared foundations: every tunable, and locale-free number parsing.

Nothing here imports a pipeline stage, so importing `core` is cheap and cannot pull in
PyMuPDF, OpenCV or a network call. `pipeline` stays a submodule for that reason - it is the
orchestrator, not a foundation.
"""
from .config import (
    LLM_MODEL,
    LLM_TIMEOUT,
    NUM_PREDICT,
    OCR_LONG_EDGE,
    OCR_MODEL,
    OCR_TIMEOUT,
    OLLAMA_URL,
    RENDER_LONG_EDGE,
    UNSHARP_PERCENT,
    UNSHARP_RADIUS,
    UNSHARP_THRESHOLD,
)
from .numbers import (
    extract_currency,
    is_blank,
    is_dash,
    parse_number,
    split_quantity_unit,
)

__all__ = [
    "LLM_MODEL", "LLM_TIMEOUT", "NUM_PREDICT", "OCR_LONG_EDGE", "OCR_MODEL", "OCR_TIMEOUT",
    "OLLAMA_URL", "RENDER_LONG_EDGE", "UNSHARP_PERCENT", "UNSHARP_RADIUS", "UNSHARP_THRESHOLD",
    "extract_currency", "is_blank", "is_dash", "parse_number", "split_quantity_unit",
]
