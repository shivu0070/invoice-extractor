"""Preprocessing stage - PDF page to the exact pixels the vision model sees.

    page_render.page_images()   PDF      -> one upright image per page
    orient.upright()            image    -> rotated to reading orientation, text rows levelled
    preprocess.prepare()        image    -> resized to 1540px long edge, one unsharp mask

Three steps, and each one is here because removing it was measured and cost points. Orientation
is worth 49 cells on a single page; deskew 3 more; the unsharp mask about 1.5 points overall.
Everything else that was tried - binarisation, CLAHE, denoise, background division, cropping -
lost, and `core.config.UNSHARP_*` carries that table.
"""
from .orient import upright
from .page_render import page_images
from .preprocess import prepare

__all__ = ["page_images", "prepare", "upright"]
