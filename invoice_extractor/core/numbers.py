"""Turning a printed cell into a number, deterministically.

Every rule here answers a fact seen on a real page in this corpus, not a hypothesis: a
thousands separator that is a comma on one supplier's invoice, a space on another's and an
apostrophe on a third's OCR'd receipt; a currency symbol glued to the figure with no space or
sitting a few hundred pixels away from it; a lone dash that means zero sitting beside a blank
cell that means "no data was printed here at all". Confusing the last two is exactly the
failure the project's archive spent a session chasing (a dash read as "$" instead of zero) -
so `is_dash` and `is_blank` are kept as two different questions with two different answers,
never collapsed into one "falsy" cell.

No locale is ever assumed from who printed the page. The separator convention is read off the
punctuation actually present in the string, per docs/PROJECT.md's hard rule against
hardcoding for one document.
"""

from __future__ import annotations

import re
from typing import Optional

DASH_CHARS = {"-", "–", "—"}  # hyphen, en dash, em dash

_DASH_RE = re.compile(r"^[A-Za-z$€£]*\s*[-–—]$")
_KEEP_NUMERIC = re.compile(r"[^0-9.,'\s-]")
_CURRENCY_TOKEN = re.compile(r"[A-Za-z$€£][A-Za-z$€£]*")
_QTY_UNIT = re.compile(r"^([0-9][0-9.,'\s-]*[0-9]|[0-9])\s*([A-Za-z%]+)?$")


def is_blank(cell: Optional[str]) -> bool:
    """No text at all - as opposed to a dash, which is text that means zero."""
    return not (cell or "").strip()


def is_dash(cell: Optional[str]) -> bool:
    """A cell that is *only* a dash, optionally beside a currency symbol - `"-"`, `"$-"`,
    `"$ -"`, `"TSh -"`. The accounting convention for zero. `"-5"` and `"12-34"` are not this;
    they still carry a number and must reach `parse_number` untouched."""
    stripped = (cell or "").strip()
    return stripped in DASH_CHARS or bool(_DASH_RE.match(stripped))


def parse_number(text: Optional[str]) -> Optional[float]:
    """A printed figure -> float, or `None` when the cell carries no figure at all.

    A dash resolves to `0.0` (see `is_dash`); a blank cell resolves to `None` because it is a
    different fact about the page - absence of data, not a stated zero. The decimal mark is
    whichever of `,` or `.` last appears followed by exactly one or two digits to the end of
    the string; apostrophes and spaces are always thousands separators, never decimal marks,
    because no supplier in this corpus uses either as one.
    """
    if text is None:
        return None
    if is_dash(text):
        return 0.0
    body = _KEEP_NUMERIC.sub("", text).strip()
    if not body or not any(ch.isdigit() for ch in body):
        return None

    negative = body.startswith("-") or body.endswith("-")
    body = body.strip("-").strip()
    if not body:
        return None
    body = body.replace("'", "")

    # A space is only a thousands separator when the groups it separates really are groups: one
    # to three digits, then threes. Collapsing every space unconditionally is how a misread
    # currency symbol becomes a wrong number rather than a blank - measured 2026-09-01 on the
    # synthetic corpus, PP-OCR returns `$ 1,214.25` as `5 1,214.25` (§3d records the same `$` ->
    # `5`/`6`/`S` confusion), and `"5 1,214.25".replace(" ", "")` is 51,214.25. Forty-two times
    # the real figure, emitted with no sign that anything went wrong. `1 335,00` is a genuine
    # space-grouped 1335.00 and still parses as one; `5 1,214.25` is two tokens, and the answer
    # is the one carrying the figure.
    parts = [p for p in body.split() if any(ch.isdigit() for ch in p)]
    if len(parts) > 1 and not (re.fullmatch(r"\d{1,3}", parts[0])
                               and all(re.fullmatch(r"\d{3}([.,]\d{1,2})?", p)
                                       for p in parts[1:])):
        # Most digits wins, last on a tie: the figure is the long token, and a stray symbol
        # misread as a digit is the short one, whichever side of it it landed on.
        body = max(parts, key=lambda p: (sum(ch.isdigit() for ch in p), parts.index(p)))
    body = body.replace(" ", "")

    last_sep = max(body.rfind(","), body.rfind("."))
    if last_sep == -1:
        digits = body
    else:
        frac = body[last_sep + 1:]
        if 1 <= len(frac) <= 2 and frac.isdigit():
            digits = f"{re.sub(r'[.,]', '', body[:last_sep])}.{frac}"
        else:
            digits = re.sub(r"[.,]", "", body)

    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def extract_currency(text: Optional[str]) -> Optional[str]:
    """The currency-like token in a money cell - `"$"`, `"TSh"`, `"USD"` - wherever it sits
    relative to the figure. Only meaningful when called on a cell already known to hold a
    money value; a quantity cell's unit-of-measure ("Pc") would match the same pattern, and
    telling the two apart is the caller's job (it knows which column it is looking at)."""
    if not text:
        return None
    m = _CURRENCY_TOKEN.search(text)
    return m.group(0) if m else None


def split_quantity_unit(text: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    """A quantity cell that may carry its unit inline - `"20.00 Pc"` -> `(20.0, "Pc")` - split
    so `qty` stays numeric and the unit is not lost. Dash and blank follow the same convention
    as `parse_number`/`is_blank`; a cell with no trailing letters returns `(value, None)`."""
    if is_blank(text):
        return None, None
    if is_dash(text):
        return 0.0, None
    m = _QTY_UNIT.match((text or "").strip())
    if not m:
        return parse_number(text), None
    return parse_number(m.group(1)), (m.group(2) or None)
