"""Score the pipeline against the hand-checked truth tables.

    python scripts/score.py --truth ../corpus/md --pdfs ../corpus
    python scripts/score.py --truth ../corpus/md --json out/     # score saved runs

Uses the same `cell_eq` and row alignment as the previous pipeline's scorer, so a number here
is directly comparable with every number in the project record. Reference, same 16 pages, same
grader, against the truth tables as corrected on 2026-09-10:

    geometry pipeline (the old src2 tree, 51 layout functions)   414/10/0   97.6%   12/16
    LightOnOCR raw markdown, no LLM                 406/38/1   91.2%   11/16

A cell counts three ways: `right` (matches), `wrong` (a value, but the wrong one) and `blank`
(truth has a value, we returned nothing). `blank` is scored against you - it is a miss, not a
free pass - and a page is `perfect` only with zero wrong AND zero blank.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invoice_extractor.core.numbers import parse_number


def parse_md_table(path: Path) -> tuple:
    lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for l in lines:
        if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$", l):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        rows.append([re.sub(r"<br\s*/?>", " ", c, flags=re.I).strip() for c in cells])
    return (rows[0], rows[1:]) if rows else ([], [])


def norm(s) -> str:
    s = re.sub(r"<br\s*/?>", " ", str(s or ""), flags=re.I)
    s = s.replace("\\n", " ").replace("\\r", " ")
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def cell_eq(a, b) -> bool:
    na, nb = norm(a), norm(b)
    if not na and not nb:
        return True
    if na == nb:
        # Exact after norm wins outright - never let the numeric coercion override a real
        # match. `parse_number` will read a float out of "FUEL(7841AB04, 7718AB04)-REIMBURSEMENT"
        # letters and all, and a different one out of the same text without the comma-space,
        # which would flip two identical strings to "wrong".
        return True
    va, vb = parse_number(str(a or "")), parse_number(str(b or ""))
    if va is not None and vb is not None:
        return abs(va - vb) < 0.005
    if va is not None or vb is not None:
        return False
    return na == nb


def align_rows(src_rows, got_rows) -> list:
    """Needleman-Wunsch on row signatures, so a missing or spurious row shifts nothing.

    Positional comparison (row i to row i) is the trap here: one extra header row absorbed into
    the body makes every later row mismatch and reports dozens of failures for one defect.
    """
    a = [norm(" ".join(str(c) for c in r)) for r in src_rows]
    b = [norm(" ".join(str(c) for c in r)) for r in got_rows]
    n, m, GAP = len(a), len(b), 1.0
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * GAP; back[i][0] = "u"
    for j in range(1, m + 1):
        cost[0][j] = j * GAP; back[0][j] = "l"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 1 - SequenceMatcher(None, a[i - 1], b[j - 1]).ratio() \
                if (a[i - 1] or b[j - 1]) else 0.0
            cost[i][j], back[i][j] = min(
                [(cost[i - 1][j - 1] + sub, "d"), (cost[i - 1][j] + GAP, "u"),
                 (cost[i][j - 1] + GAP, "l")], key=lambda x: x[0])
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        d = back[i][j]
        if d == "d":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d == "u":
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


def score_page(src_header, src_rows, got_columns, got_row_dicts) -> tuple:
    got_rows = [[rd.get(c, "") for c in got_columns] if isinstance(rd, dict)
                else [""] * len(got_columns) for rd in got_row_dicts]
    if len(got_columns) == len(src_header):
        col_map = list(range(len(src_header)))
    else:
        col_map = []
        for h in src_header:
            best_j, best_r = None, 0.35
            for j, g in enumerate(got_columns):
                r = SequenceMatcher(None, norm(h), norm(g)).ratio()
                if r > best_r:
                    best_r, best_j = r, j
            col_map.append(best_j)

    right = wrong = blank = 0
    detail = []
    for si, gi in align_rows(src_rows, got_rows):
        if si is None:
            continue
        got_row = got_rows[gi] if gi is not None else []
        for c, want in enumerate(src_rows[si]):
            if not str(want).strip():
                continue
            gj = col_map[c] if c < len(col_map) else None
            got = got_row[gj] if (gj is not None and gj < len(got_row)) else None
            col = src_header[c] if c < len(src_header) else f"col{c}"
            if got is None or not str(got).strip():
                blank += 1
                detail.append((si, col, want, "<blank>" if gi is not None else "<no row>"))
            elif cell_eq(want, got):
                right += 1
            else:
                wrong += 1
                detail.append((si, col, want, got))
    return right, wrong, blank, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", type=Path, required=True, help="directory of <page key>.md")
    ap.add_argument("--json", type=Path, default=None,
                    help="score a directory of already-produced <page_key>.json files")
    ap.add_argument("--pdfs", type=Path, default=None,
                    help="run the pipeline live over the PDFs in this directory instead")
    ap.add_argument("--host", default=None)
    ap.add_argument("--verbose", action="store_true", help="print every wrong cell")
    args = ap.parse_args(argv)

    if not (args.json or args.pdfs):
        print("give --json (score saved output) or --pdfs (run live)", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    if args.pdfs:
        from invoice_extractor.core.config import OLLAMA_URL
        from invoice_extractor.core.pipeline import process_pdf
        host = args.host or OLLAMA_URL
        for pdf in sorted(args.pdfs.glob("*.pdf")):
            if "synthetic" in pdf.name.lower():
                continue                       # no truth tables for those
            for res in process_pdf(pdf, host=host):
                results[res.key] = res.to_dict()
                print(f"  ran {res.key}", flush=True)
    else:
        for f in sorted(args.json.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            results[d.get("page") or f.stem.replace("_p", "#p")] = d

    tot_r = tot_w = tot_b = perfect = scored = missing = 0
    print(f"\n{'page':46s} {'result':>10s} {'rows':>8s}")
    print("-" * 70)
    for md in sorted(args.truth.glob("*.md")):
        key = md.stem
        if "synthetic" in key.lower():
            continue
        h, rows = parse_md_table(md)
        if not h or not rows:
            continue
        got = results.get(key)
        if got is None:
            print(f"{key[:46]:46s} {'MISSING':>10s}")
            missing += 1
            continue
        scored += 1
        r, w, b, detail = score_page(h, rows, got.get("columns") or [], got.get("rows") or [])
        tot_r += r; tot_w += w; tot_b += b
        ok = (w == 0 and b == 0 and r + w + b > 0)
        perfect += ok
        n_got = len(got.get("rows") or [])
        tag = "PERFECT" if ok else f"{r}/{r + w + b}"
        # Row count is printed beside the score because over-capture is invisible in the
        # percentage: a page that swallows five footer rows can still score well.
        print(f"{key[:46]:46s} {tag:>10s} {f'{n_got}/{len(rows)}':>8s}")
        if args.verbose:
            for si, col, want, gotv in detail[:8]:
                print(f"      row{si} [{str(col)[:16]:16s}] want={str(want)[:34]!r} "
                      f"got={str(gotv)[:34]!r}")
    tot = tot_r + tot_w + tot_b
    print("-" * 70)
    if missing:
        print(f"!! {missing} truth page(s) had no result")
    print(f"\n=== right={tot_r} wrong={tot_w} blank={tot_b}  "
          f"strict={100 * tot_r / tot if tot else 0:.1f}%  perfect={perfect}/{scored} ===")
    print("--- reference: geometry pipeline 414/10/0 97.6% 12/16 | "
          "raw LightOnOCR markdown 406/38/1 91.2% 11/16 ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
