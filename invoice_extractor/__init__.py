"""Invoice line-item extraction: PDF -> JSON, layout-independent.

    LightOnOCR-2 (vision)  ->  markdown  ->  llama3.2 (text)  ->  JSON

No table detection, no column geometry, no layout repair. See `core.pipeline` for why that
absence is the design.

    core           tunables and number parsing, plus the orchestrator
    preprocessing  PDF page -> the exact pixels the vision model sees
    ocr            those pixels -> markdown
    llm            that markdown -> JSON
    app            the two front-ends, `cli` and `frontend`
"""
from .core.pipeline import PageResult, process_page, process_pdf

__all__ = ["PageResult", "process_page", "process_pdf"]
__version__ = "0.1.0"
