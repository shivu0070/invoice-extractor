"""Command line: one or more PDFs in, one JSON file per page out.

    python -m invoice_extractor data/samples/*.pdf --out out/
    python -m invoice_extractor invoice.pdf --keep-markdown
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..core.config import LLM_MODEL, OCR_MODEL, OLLAMA_URL
from ..core.pipeline import process_pdf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="invoice_extractor",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("pdfs", nargs="+", type=Path, help="PDF file(s) to read")
    ap.add_argument("--out", type=Path, default=Path("out"),
                    help="directory for the JSON output (default: ./out)")
    ap.add_argument("--host", default=OLLAMA_URL)
    ap.add_argument("--ocr-model", default=OCR_MODEL)
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--keep-markdown", action="store_true",
                    help="also write the vision model's raw markdown beside each JSON - the "
                         "first thing to look at when a page comes out wrong")
    args = ap.parse_args(argv)

    files = [p for p in args.pdfs if p.is_file()]
    missing = [p for p in args.pdfs if not p.is_file()]
    for p in missing:
        print(f"  skip (not a file): {p}", file=sys.stderr)
    if not files:
        print("no PDFs to process", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"ocr : {args.ocr_model}\nllm : {args.llm_model}\nhost: {args.host}\n")

    n_pages = n_err = 0
    t0 = time.time()
    for pdf in files:
        def report(res):
            nonlocal n_pages, n_err
            n_pages += 1
            n_err += bool(res.error)
            note = res.error or f"{len(res.rows)} row(s) x {len(res.columns)} col(s)"
            print(f"  {res.key[:52]:52s}  {note}", flush=True)

        def read(key, md):
            print(f"  read {key[:52]:52s}  {len(md):5d} chars", flush=True)

        for res in process_pdf(pdf, host=args.host, ocr_model=args.ocr_model,
                               llm_model=args.llm_model, on_page=report, on_ocr=read):
            safe = res.key.replace("#", "_")
            (args.out / f"{safe}.json").write_text(
                json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            if args.keep_markdown:
                (args.out / f"{safe}.md").write_text(res.markdown, encoding="utf-8")

    dt = time.time() - t0
    print(f"\n{n_pages} page(s) in {dt:.1f}s ({dt / max(n_pages, 1):.1f}s/page) -> {args.out}")
    if n_err:
        print(f"!! {n_err} page(s) failed")
    return 1 if n_err == n_pages else 0


if __name__ == "__main__":
    sys.exit(main())
