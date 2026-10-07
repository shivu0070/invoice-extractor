"""Resize to the model's native input, then one unsharp mask. That is the entire step.

The short version of three sweeps and about twenty modes: **keep the greyscale, steepen the
edges, change nothing else.** Every mode that discarded information lost points in proportion
to how much it discarded, and every mode that added a second correction on top of the unsharp
mask double-counted the first and lost too. `config.UNSHARP_*` carries the full table.
"""
from __future__ import annotations

from PIL import Image, ImageFilter

from ..core.config import OCR_LONG_EDGE, UNSHARP_PERCENT, UNSHARP_RADIUS, UNSHARP_THRESHOLD


def prepare(img: Image.Image, long_edge: int = OCR_LONG_EDGE) -> Image.Image:
    """The exact pixels the OCR model sees.

    Resize happens BEFORE the sharpen, not after: sharpening and then downsampling throws away
    most of the edge contrast it just created.
    """
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) != long_edge:
        s = long_edge / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=UNSHARP_RADIUS,
                                              percent=UNSHARP_PERCENT,
                                              threshold=UNSHARP_THRESHOLD))
