"""PDF -> one PIL image per page, the right way up.

The previous pipeline's renderer did four things: orientation (Tesseract OSD), deskew, denoise
and CLAHE. Two of those are gone and two are kept, and the split is measured, not aesthetic:

  DROPPED   denoise + CLAHE. Its own docstring recorded that the enhancement step HURT
            (92.0% -> 87.7%) and was retained only for run-to-run determinism. Three further
            sweeps here agreed - see `config.UNSHARP_*`.
  KEPT      orientation + deskew. I dropped these in the first build on the theory that a
            vision model reads a crooked page fine. It does not: ACER p2 fell to 5/55 from
            54/55, and July p3 to 11/14 from 14/14. See `orient` for the numbers.
"""
from __future__ import annotations

from pathlib import Path

import fitz                                            # PyMuPDF
from PIL import Image

from ..core.config import RENDER_LONG_EDGE
from .orient import upright


def page_images(pdf: Path, long_edge: int = RENDER_LONG_EDGE,
                use_osd: bool = True, use_deskew: bool = True
                ) -> list[tuple[str, Image.Image]]:
    """`[(page_key, image), ...]` for every page of `pdf`, rotated upright and deskewed.

    `page_key` is `<filename>#p<n>`, 1-based - the same key the scorer and the sample truth
    tables use, so an output file always lines up with the page it came from.

    `use_osd=False` skips orientation when Tesseract is unavailable or the input is known to be
    upright; `orient.upright` already degrades to a no-op in that case, so this is only a speed
    switch (OSD shells out once per page).
    """
    out: list[tuple[str, Image.Image]] = []
    doc = fitz.open(str(pdf))
    try:
        for i, page in enumerate(doc):
            # Scale by the LONG edge so portrait and landscape pages get the same detail
            # budget; a width-only zoom would starve a landscape page.
            rect = page.rect
            zoom = long_edge / max(rect.width, rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append((f"{pdf.name}#p{i + 1}",
                        upright(img, use_osd=use_osd, use_deskew=use_deskew)))
    finally:
        doc.close()
    return out
