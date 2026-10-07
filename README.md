# invoice_extractor

PDF invoice → JSON line items. **Layout-independent**: no table detection, no column geometry,
no per-document repair code.

```
PDF ─▶ render ─▶ upright ─▶ unsharp ─▶ LightOnOCR-2 ─▶ markdown ─▶ llama3.2 ─▶ JSON
                 (OSD +                 (vision, 1B)                (text, 3B)
                  deskew)
```

```
invoice_extractor/
  core/
    config.py            every tunable, each with the measurement that fixed it
    numbers.py           printed figure -> float (heavily edge-cased)
    pipeline.py           the four steps, batched by stage
  preprocessing/
    page_render.py        PDF -> page images (PyMuPDF)
    orient.py              upright via Tesseract OSD, then deskew - both load-bearing
    preprocess.py          resize to 1540 + unsharp 1.6/140/3, nothing else
  ocr/
    lightonocr.py          LightOnOCR-2 -> markdown, passed downstream verbatim
  llm/
    markdown_to_json.py    llama3.2 -> JSON, one call, model-agnostic prompt
  app/
    cli.py                 python -m invoice_extractor
    frontend.py            python -m invoice_extractor.app.frontend - the GUI
scripts/
  score.py        same cell_eq + row alignment as the old src2 tree, so numbers compare
```

## Why this shape

The predecessor (an older, since-removed pipeline) reaches a higher score by finding the table
in code: PP-OCR for text, TATR + CV for structure, then a layout module — **51 functions** of
repair for fused columns, polluted headers, split markers, wrapped rows. It works, and its
development record names individual PDFs **303 times**. Every new layout that breaks it needs
new code, and a vocabulary ablation drops it from 96.7% to 42.3%, which is the measure of how
much of that score is tuning rather than method.

Here, finding the table is the language model's job. A new layout needs no code at all. That
costs a few points of cell accuracy and buys an extraction surface small enough to read in one
sitting.

## Install

```
pip install pymupdf pillow opencv-python numpy
```

Optionally [Tesseract](https://github.com/tesseract-ocr/tesseract), used **only** for page
orientation — never to read text. Without it the pipeline still runs and simply leaves pages as
rendered, which is fine unless your PDFs carry a wrong `/Rotate`. It is not optional in spirit:
one page in this corpus scores **5/55 without it and 54/55 with it**.

Plus [Ollama](https://ollama.com), with both models:

```
ollama pull maternion/LightOnOCR-2:1b
ollama pull llama3.2:3b-instruct-q4_K_M
```

Both fit a 4 GB GPU. That is not a compromise — a 7B was measured and **lost** (see below).

## Use

```
python -m invoice_extractor invoice.pdf --out out/
python -m invoice_extractor data/*.pdf --out out/ --keep-markdown
```

One JSON per page: `columns`, `rows` (keyed by the document's own column names), and an
`envelope` of `invoice_number`, `invoice_date`, `subtotal`, `tax`, `total`.

`--keep-markdown` also writes what the vision model returned. **That is the first thing to look
at when a page comes out wrong** — it tells you immediately whether the OCR misread the page or
the LLM mis-structured it, which are different problems with different fixes.

## Scoring

```
python scripts/score.py --truth path/to/truth/md --pdfs path/to/pdfs
python scripts/score.py --truth path/to/truth/md --json out/
```

The 16-page tuning corpus and its hand-corrected truth tables are not included in this repo —
they're public vendor invoice templates and sample documents assembled for internal scoring.
Point `--truth`/`--pdfs` at your own PDFs and corrected markdown truth tables to reproduce this.

Same `cell_eq` and row alignment as the predecessor, so numbers are directly comparable.
On the same 16 real pages, against truth as corrected 2026-09-10:

| | right/wrong/blank | strict | perfect |
|---|---|---|---|
| geometry pipeline (old predecessor, 51 layout functions) | 414/10/0 | 97.6% | 12/16 |
| LightOnOCR markdown + a table parser | 405/13/6 | **95.5%** | **11/16** |
| **this pipeline** (LightOnOCR -> llama3.2) | 398/25/1 | 93.9% | 11/16 |

**Read the last two rows against each other.** Feeding the same markdown through
`score_local_ocr.response_to_grid` - a ~150-line pipe/HTML table parser that picks the line-item
table by numeric density - still has fewer wrong cells (12 fewer). But the two now tie on
perfect pages (11/16 each), and the parser leaves six cells blank against this pipeline's one.
The model re-types every value it copies, and each retype is a chance to err, where a parser
transcribes.

Do not read an earlier version of this file that claimed the LLM added 1.7 points. That
compared against a number from a LENIENT scorer (`score_grid`, which has a `near` bucket that
does not break a PERFECT page). Measured with the same strict scorer as this table, it is
negative.

What the LLM buys is not accuracy on these 16 pages - it is that nothing in the path is written
for a layout. The parser row still contains code that hunts for a table; a document it cannot
find one in returns nothing, and the fix is more parser. That trade is the open question of
this pipeline, not a settled result.

## Results

98.6% of cells right on 24 public sample invoices not used while tuning - real vendor invoices
(AWS, Flipkart, Coolblue, government tax samples, and more) spanning six currencies, scored by
checking every extracted value against each PDF's own embedded text rather than a hand-built
truth table. Every remaining miss was a single-character misread (0/C, 9/8, S/5, Q/O), not a
structural failure - the pipeline's errors on unseen layouts are the same shape as its errors on
the tuning corpus, not a different, worse one.

## Demo

Sample output for `examples/Commercial-Invoice-Template.pdf_p1.json` (included in this repo):

```json
{
  "page": "Commercial-Invoice-Template.pdf#p1",
  "columns": ["PRODUCT", "HS CODE", "UNITS", "UNIT PRICE", "TOTAL"],
  "rows": [
    {
      "PRODUCT": "Conveyor Belt 25\" Country of origin: US",
      "HS CODE": "88565.2252",
      "UNITS": "2",
      "UNIT PRICE": "$200.00",
      "TOTAL": "$400.00"
    }
  ],
  "envelope": {
    "invoice_number": "F1000876/23",
    "invoice_date": "14/08/2023",
    "subtotal": "$485.00",
    "discount": "",
    "tax": "$117.00",
    "total": "$702.00"
  }
}
```

Full write-up: [TODO - add link]

## Decisions already measured — do not re-litigate without new numbers

**Preprocessing is closed.** ~20 modes across three sweeps; `sharpen` (unsharp 1.6/140/3) won
every time. Binarisation, the standard advice for classical OCR, costs 16 points — these are
clean digital renders read by a ViT, so there is no shadow to remove and nothing to threshold.
Six blends all scored below sharpen alone. A per-page oracle free to pick the best of 14 modes
gained **one cell and zero pages**.

**Bigger LLMs do not help.** Five-model bake-off on a T4: llama3.2:3b **93.2%**, qwen2.5-coder:3b
92.9%, qwen2.5:3b 92.0%, gemma3:4b 87.0%, qwen2.5:**7b** 84.7%. Parameters never won. llama3.2
is also the only model that left **zero blank cells**.

**Model choice is path-specific.** llama3.2 wins here but *loses* on the predecessor's clean
pre-parsed tables (402/19/3 vs qwen2.5:3b's 411/12/1). llama3.2 is good at reading a messy
document; qwen2.5:3b is good at transcribing an already-clean table. Do not port this choice.

**Prompt text is not free on a 3B.** Adding one clause to the system prompt cost 23 correct
cells and introduced 9 blanks — and the instruction did not even take effect, because the
absolute "copy exactly as printed" rule above it won the conflict. Deterministic work belongs
in code: see `extract._detag`.

A second attempt - asking the OCR prompt to be careful transcribing codes/serials/IDs - fixed
zero cells, broke one, and made an already-fragile page hallucinate its own prompt text back
into the output. Reverted. The model's character-level misreads are a visual-encoding limit,
not an attention gap, and no prompt wording has yet moved that number.

## Known limitations

- **Structural OCR failures are unpatchable here.** On one page the vision model invents column
  headers and promotes real data into a second header row; on another it pulls a prose line in
  as a data row and shifts everything after it. No prompt recovers either. This is the honest
  cost of not having layout code — the failures do not disappear, they stop being patchable.
- **Not reproducible across llama.cpp builds.** `temperature=0` is greedy, not deterministic
  across Ollama versions; the same page can move by a cell or two between hosts.
- **Truth tables are fallible.** Six were wrong on 2026-09-10 — all cases of a transcriber
  silently "correcting" the invoice's own misspellings and dropped characters. Fixing them was
  worth 2 perfect pages on each pipeline, for no engineering. When an engine disagrees with
  truth on an identifier, crop the page at 3× and read it before assuming the engine is wrong.
