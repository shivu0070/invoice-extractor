"""OCR stage - a page image to markdown, via LightOnOCR-2 on Ollama.

One call, and whatever the model returns is passed downstream **verbatim**. There is no table
parser between this stage and the LLM: finding the line-item table is the language model's job,
which is what makes the pipeline layout-independent. Parsing the markdown first was measured
and loses - see `llm.extract`.
"""
from .lightonocr import read_page

__all__ = ["read_page"]
