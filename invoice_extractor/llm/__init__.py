"""LLM stage - a whole markdown document to structured JSON, in one model call.

The prompt is model-agnostic and assumes what a vision model actually emits: a whole page,
prose included, possibly several tables, in pipe or HTML syntax. `columns` is whatever the
document prints, so the output shape adapts per invoice with nothing hardcoded.
"""
from .markdown_to_json import ENVELOPE_FIELDS, extract

__all__ = ["ENVELOPE_FIELDS", "extract"]
