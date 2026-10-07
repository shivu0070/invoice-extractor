"""Every tunable in the shipping pipeline, with the measurement that fixed it.

Nothing here is a guess. Each value below beat its alternatives on the same 16 real invoice
pages, scored cell-by-cell against `data/samples/md/`, and the number that decided it is
recorded beside it. Change one only with a new measurement, not an intuition.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- services

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").strip()

# --------------------------------------------------------------------------- models

#: Vision model: page image -> markdown table. 596M params + a 409M CLIP projector, Q8_0.
OCR_MODEL = os.environ.get("OCR_MODEL", "maternion/LightOnOCR-2:1b").strip()

#: Text model: markdown document -> JSON. Winner of a five-model bake-off on a T4, on THIS
#: path - whole page in, one call, model-agnostic prompt:
#:
#:      llama3.2:3b-instruct-q4_K_M   93.2%   <- best, and the only one with ZERO blanks
#:      qwen2.5-coder:3b-instruct     92.9%
#:      qwen2.5:3b-instruct           92.0%
#:      gemma3:4b-it-qat              87.0%
#:      qwen2.5:7b-instruct-q5_K_M    84.7%   <- more parameters never won
#:
#: The 7B losing matters: model size is not the bottleneck on this task, so a 4 GB GPU costs
#: no accuracy. ~2 GB at Q4, so it fits one.
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b-instruct-q4_K_M").strip()

# --------------------------------------------------------------------------- imaging

#: Render the PDF page at this long edge before preprocessing. Larger than the model's input
#: on purpose: downsampling from a generous render is sharper than rendering straight to size.
RENDER_LONG_EDGE = int(os.environ.get("RENDER_LONG_EDGE", "2048"))

#: LightOnOCR-2's documented native maximum longest edge. Anything larger is downscaled inside
#: the model anyway, so the extra pixels buy nothing and cost encode time.
OCR_LONG_EDGE = int(os.environ.get("OCR_LONG_EDGE", "1540"))

#: Unsharp mask, and nothing else. Roughly twenty preprocessing modes were measured across
#: three sweeps - binarisation, Otsu, CLAHE, gamma, Retinex background division, bilateral
#: denoise, table-rule darkening, ink darkening, super-resolution-style upscaling, and six
#: blends of those - and this setting won every time:
#:
#:      sharpen (1.6/140/3)   91.2%   <- ships
#:      docpipe (the full published bilateral->CLAHE->gamma->unsharp recipe)   89.9%
#:      none (untouched)      89.7%
#:      adaptive threshold    72.7%   <- classical OCR advice, catastrophic here
#:
#: A per-page oracle allowed to pick the best of all fourteen modes scored 406/37 against
#: sharpen-everywhere's 406/38: one cell, zero pages. Preprocessing is a closed avenue.
#: The reason is that these are clean digital renders, not photographs - there is no shadow to
#: remove, no illumination to normalise, no grain to denoise - and the model is a ViT trained
#: on colour documents, so anything that quantises or greys the page moves it off-distribution.
UNSHARP_RADIUS = 1.6
UNSHARP_PERCENT = 140
UNSHARP_THRESHOLD = 3

# --------------------------------------------------------------------------- generation

#: Hard cap on OCR output tokens. Defuses a repetition loop in which the model regenerated one
#: page over and over, emitting 90KB against a normal page's 2.2KB and taking 363s against a
#: median 8s. Average went 30.4 -> 10.2 s/page with byte-identical accuracy. The model family's
#: own training pipeline filters "loops and repetitions", so this is a known failure mode.
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "3000"))

OCR_TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "600"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))
