"""Markdown document -> JSON, in one model call.

The prompt is deliberately **model-agnostic**. The previous pipeline's three-pass sequence was
written against `qwen2.5:3b-instruct` and every one of its prompts opened "You will be shown a
markdown table", with pass 2 and 3 instructing the model to "skip the ... --- separator row".
That assumes a single, already-isolated, pipe-syntax table - true only because a parser had
built one first. Benchmarking any other model on those prompts measures how well it imitates
the incumbent, not how well it reads an invoice.

This prompt assumes instead what a vision model actually emits: a whole page, prose included,
possibly several tables, pipes or HTML. Finding the line-item table is the model's job.
`columns` is whatever the document prints, so the shape adapts per invoice with nothing
hardcoded - no required `qty`, no required `description`.

Two things are deliberately NOT in the prompt:

* **Money stays a string.** A schema that typed amounts as numbers would turn "$1,500.00" into
  1500.0 and destroy the verbatim property the whole design rests on.
* **HTML stripping.** It was tried, as an explicit instruction, and it did not work: llama3.2
  returned the markup anyway, because the absolute "copy every value exactly as printed" rule
  above it wins the conflict. Worse, adding the clause cost 23 correct cells and introduced 9
  blanks (395/29/0 -> 372/43/9). Prompt text is not free on a 3B - a new rule competes with the
  existing ones rather than stacking on them. `_detag` does the job in code instead.
"""
from __future__ import annotations

import html
import json
import re
import urllib.request

from ..core.config import LLM_MODEL, LLM_TIMEOUT, OLLAMA_URL

ENVELOPE_FIELDS = ["invoice_number", "invoice_date", "subtotal", "discount",
                   "tax", "total"]

SYSTEM = (
    "You are given one invoice, as markdown. It may contain headings, addresses, notes and "
    "more than one table, and a table may be written with markdown pipes or as HTML.\n"
    "Find the LINE-ITEM table - the one listing the goods or services being billed, one row "
    "per item - and ignore every other table on the page, such as delivery addresses, "
    "summary boxes or payment slips.\n"
    "Return these keys:\n"
    "  columns - that table's column headings, exactly as printed, left to right. Use the "
    "headings the document itself uses; never rename, translate or standardise them, and "
    "never add a column the table does not have.\n"
    "  rows - one array per line-item row, holding that row's cells in the same left-to-right "
    "order as columns. Each array must have exactly as many items as columns.\n"
    "  invoice_number, invoice_date, subtotal, discount, tax, total - taken from anywhere "
    "in the document, or an empty string where the document does not print one.\n"
    "Copy every value exactly as printed: keep currency symbols, thousands separators, "
    "decimal points, dashes and letter case. Do not compute, round, reformat or infer any "
    "value. Use an empty string for a cell that is blank. Never merge, deduplicate, reorder "
    "or drop a row, even when two rows are identical."
)

#: `rows` is an array of arrays rather than an array of objects because Ollama compiles the
#: schema to a GBNF grammar and cannot express per-row dynamic keys (`prefixItems` is
#: rejected). Positional cells are the only way to keep the column names free.
SCHEMA = {
    "type": "object",
    "properties": {
        **{f: {"type": "string"} for f in ENVELOPE_FIELDS},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
    "required": ["columns", "rows"] + ENVELOPE_FIELDS,
}

_HTML_TAG = re.compile(r"<[^<>]{0,120}>")
_HTML_BR = re.compile(r"<br\s*/?>", re.I)


def _detag(v: str) -> str:
    """Strip a cell's HTML formatting down to the text a reader would see.

    The vision model emits real HTML tables and inline formatting rides along inside the
    cells: `<td>Services<br><em>Cost of various services.</em></td>`. This is a regex's job -
    deterministic, free, and identical on every layout - where the prompt rule that tried to do
    it was none of those (see the module docstring).

    `<br>` becomes a space rather than vanishing, because it separates two real words that
    would otherwise fuse into "ServicesCost". The tag pattern requires a closing `>` with no
    angle bracket inside, so an ordinary cell reading "5 < 10 pcs" survives untouched.
    """
    v = _HTML_BR.sub(" ", v)
    v = _HTML_TAG.sub("", v)
    v = html.unescape(v)
    return " ".join(v.split())


def _chat(system: str, user: str, schema: dict, host: str, model: str, timeout: int) -> str:
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False, "options": {"temperature": 0}, "format": schema}
    req = urllib.request.Request(host.rstrip("/") + "/api/chat",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def extract(md_text: str, host: str = OLLAMA_URL, model: str = LLM_MODEL,
            timeout: int = LLM_TIMEOUT) -> dict:
    """`{columns: [...], rows: [{col: value}, ...], envelope: {...}}` from one call."""
    raw = _chat(SYSTEM,
                f"INVOICE DOCUMENT:\n{md_text}\n\n"
                "Extract the line-item table and the invoice details.",
                SCHEMA, host, model, timeout)
    obj = _parse(raw)
    columns = [_detag(str(c)) for c in (obj.get("columns") or [])]
    rows = []
    for r in obj.get("rows") or []:
        if not isinstance(r, list):
            continue
        cells = [("" if v is None else _detag(str(v))) for v in r]
        # Pad or trim to the column count so a short row never silently shifts its cells left.
        cells = (cells + [""] * len(columns))[:len(columns)]
        rows.append({c: cells[i] for i, c in enumerate(columns)})
    return {"columns": columns, "rows": rows,
            "envelope": {f: _detag(str(obj.get(f, "") or "")) for f in ENVELOPE_FIELDS}}
