"""Page image -> markdown, via LightOnOCR-2 on Ollama.

The output is handed to the LLM stage **verbatim**. There is deliberately no table parser
between them: the previous pipeline located the line-item table in code and re-rendered it as
a clean pipe table before the model saw it, and dropping that step is the whole point of this
one. Finding the table is now the language model's job, which is why it works on a layout
nobody wrote code for.

LightOnOCR emits HTML `<table>` markup as often as pipe tables, sometimes with prose, a
letterhead and several tables on one page. All of that is passed through untouched.
"""
from __future__ import annotations

import base64
import json
import urllib.request

from PIL import Image

from ..core.config import NUM_PREDICT, OCR_MODEL, OCR_TIMEOUT, OLLAMA_URL

PROMPT = ("Read this invoice page and return its line-item table as a GitHub-flavored markdown "
          "pipe table. Include the column header row exactly as printed. Copy every value "
          "verbatim - do not compute, reformat or omit anything. Output only the table.")


def _png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_page(img: Image.Image, host: str = OLLAMA_URL, model: str = OCR_MODEL,
              prompt: str = PROMPT, timeout: int = OCR_TIMEOUT,
              num_predict: int = NUM_PREDICT) -> str:
    """One vision call. Returns whatever markdown the model produced, unedited.

    `temperature=0` for reproducibility within a build - note it is NOT reproducible ACROSS
    llama.cpp builds, so a number measured on one host may shift by a cell or two on another.
    """
    opts: dict = {"temperature": 0}
    if num_predict:
        opts["num_predict"] = num_predict
    body = {"model": model, "prompt": prompt, "stream": False,
            "images": [base64.b64encode(_png_bytes(img)).decode("ascii")],
            "options": opts}
    req = urllib.request.Request(host.rstrip("/") + "/api/generate",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")
