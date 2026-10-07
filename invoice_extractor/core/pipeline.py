"""PDF -> JSON. Four steps, no branches.

    render -> preprocess -> LightOnOCR -> llama3.2 -> JSON

There is no table detection, no column geometry, no layout repair and no arbitration between
candidate grids. That absence is the design, not an omission: the previous pipeline reached a
slightly higher score with 51 functions of layout repair whose development record names
individual PDFs 303 times, and each new layout that broke it needed new code. Here a new
layout needs nothing - the vision model reads the page and the language model finds the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..llm import markdown_to_json as _extract
from ..ocr import lightonocr as _ocr
from ..preprocessing import preprocess as _pre
from ..preprocessing import page_render as _render
from .config import LLM_MODEL, OCR_MODEL, OLLAMA_URL


@dataclass
class PageResult:
    key: str                       #: "<filename>#p<n>", 1-based
    markdown: str                  #: exactly what the vision model returned
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    envelope: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        d = {"page": self.key, "columns": self.columns, "rows": self.rows,
             "envelope": self.envelope}
        if self.error:
            d["error"] = self.error
        return d


def process_page(key: str, img, host: str = OLLAMA_URL, ocr_model: str = OCR_MODEL,
                 llm_model: str = LLM_MODEL) -> PageResult:
    """One page, end to end. Failures are captured per page rather than raised, so one bad
    page in a 40-page document does not lose the other 39."""
    try:
        md = _ocr.read_page(_pre.prepare(img), host=host, model=ocr_model)
    except Exception as e:                                        # noqa: BLE001
        return PageResult(key=key, markdown="", error=f"ocr: {e.__class__.__name__}: {e}")
    try:
        got = _extract.extract(md, host=host, model=llm_model)
    except Exception as e:                                        # noqa: BLE001
        return PageResult(key=key, markdown=md,
                          error=f"extract: {e.__class__.__name__}: {e}")
    return PageResult(key=key, markdown=md, columns=got["columns"],
                      rows=got["rows"], envelope=got["envelope"])


def process_pdf(pdf: Path, host: str = OLLAMA_URL, ocr_model: str = OCR_MODEL,
                llm_model: str = LLM_MODEL, on_page=None, on_ocr=None) -> list[PageResult]:
    """Every page of one PDF, **batched by stage**: all the OCR first, then all the extraction.

    Not page-by-page, and the reason is the target hardware. The two models together are ~3.5 GB
    against a 4 GB card, so they cannot both stay resident: interleaving them makes Ollama evict
    and reload on every single call - 2N loads for N pages, each tens of seconds, dwarfing the
    ~10s of actual work. Running each stage to completion costs one load per stage regardless of
    page count.

    The cost of batching is memory: every page's markdown is held at once. That is a few KB per
    page, so a 500-page document is still single-digit megabytes.

    `on_ocr(key, markdown)` fires as each page is read; `on_page(result)` as each is extracted.
    """
    pages = _render.page_images(pdf)

    read: list[tuple[str, str, str]] = []                  # (key, markdown, error)
    for key, img in pages:
        try:
            md = _ocr.read_page(_pre.prepare(img), host=host, model=ocr_model)
            err = ""
        except Exception as e:                                    # noqa: BLE001
            md, err = "", f"ocr: {e.__class__.__name__}: {e}"
        read.append((key, md, err))
        if on_ocr:
            on_ocr(key, md)

    out = []
    for key, md, err in read:
        if err:
            res = PageResult(key=key, markdown=md, error=err)
        else:
            try:
                got = _extract.extract(md, host=host, model=llm_model)
                res = PageResult(key=key, markdown=md, columns=got["columns"],
                                 rows=got["rows"], envelope=got["envelope"])
            except Exception as e:                                # noqa: BLE001
                res = PageResult(key=key, markdown=md,
                                 error=f"extract: {e.__class__.__name__}: {e}")
        out.append(res)
        if on_page:
            on_page(res)
    return out
