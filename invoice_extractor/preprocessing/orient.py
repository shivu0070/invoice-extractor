"""Get the page the right way up and its text rows horizontal. Nothing else.

This module exists because I removed it and measured the cost. The first build of this pipeline
rendered pages straight from the PDF on the theory that "a vision model reads a rotated or
skewed page fine". It does not:

    ACER INVOICE 184.pdf#p2     5/55   without orientation     54/55  with it
    July_2026.pdf#p3          11/14   without deskew          14/14  with it

`ACER INVOICE 184.pdf` carries `/Rotate 270` in the PDF and *still* needs a further 90 degrees
after that is applied - the stored rotation is simply wrong about the content. July p3 sits at
about 1.25 degrees, which is enough that its text rows stop reading as rows.

Two rejected alternatives, both measured:

* **Ink-projection orientation.** Sum ink per row and compare against the 90-degree rotation;
  text lines should band strongly across the reading direction. It gets ACER p2 right but
  wants to rotate `invoice_1_upscaled` too, which is already correct and scores PERFECT - a
  false positive at 3.1x signal strength, so no threshold separates the two cases.
* **Ask the OCR model twice and keep the better answer.** Robust, but doubles the cost of the
  most expensive stage on every page to fix one page in sixteen.

Tesseract's OSD is what actually distinguishes them, which is why it is a dependency here and
nowhere else in the package. It is used ONLY to decide an angle - never to read text - and when
it is missing or unsure the page is left exactly as rendered.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: Tesseract's own confidence scale. Below this it is coin-flipping, and guessing orientation
#: wrong costs far more than leaving a page alone.
OSD_MIN_CONFIDENCE = 1.5

#: OSD is run at full render resolution with an explicit --dpi: downscaling makes it bail with
#: "Too few characters" on a sparse page like ACER p2, one small table on an otherwise empty A4.
OSD_DPI = 300

SKEW_RANGE_DEG = 3.0
SKEW_STEP_DEG = 0.25
SKEW_DOWNSCALE = 4


def _tesseract() -> str | None:
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(c).exists():
            return c
    return None


def osd_rotation(gray: np.ndarray) -> tuple[int, float]:
    """`(clockwise degrees still needed, confidence)`; `(0, -1.0)` when OSD abstains.

    Shelled out rather than via pytesseract, which imports pandas at module level and drags a
    heavy, ABI-fragile dependency into a package that otherwise needs none.
    """
    exe = _tesseract()
    if not exe:
        return 0, -1.0
    tmp = Path(os.environ.get("TEMP", ".")) / f"_osd_{os.getpid()}.png"
    try:
        cv2.imwrite(str(tmp), gray)
        out = subprocess.run([exe, str(tmp), "stdout", "--psm", "0", "--dpi", str(OSD_DPI)],
                             capture_output=True, text=True, timeout=180)
        if out.returncode != 0 and not out.stdout.strip():
            return 0, -1.0
        rot, conf = 0, 0.0
        for line in out.stdout.splitlines():
            if line.startswith("Rotate:"):
                rot = int(line.split(":")[1].strip())
            elif line.startswith("Orientation confidence:"):
                conf = float(line.split(":")[1].strip())
        return rot % 360, conf
    except Exception:                       # noqa: BLE001 - absent tooling is not an error
        return 0, -1.0
    finally:
        tmp.unlink(missing_ok=True)


def _rotate_cw(a: np.ndarray, deg: int) -> np.ndarray:
    deg %= 360
    if deg == 90:
        return cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(a, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return a


def _projection_score(binary: np.ndarray) -> float:
    """Sum of squared row-to-row differences of the horizontal ink profile.

    Maximal when text rows are horizontal: the profile becomes a square wave of dense rows and
    empty gaps, so consecutive differences are large.
    """
    d = np.diff(binary.sum(axis=1, dtype=np.float64))
    return float((d * d).sum())


def estimate_skew(gray: np.ndarray) -> float:
    """The angle within +/- SKEW_RANGE_DEG that maximises the projection score."""
    small = cv2.resize(gray, None, fx=1.0 / SKEW_DOWNSCALE, fy=1.0 / SKEW_DOWNSCALE,
                       interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    centre = (w / 2.0, h / 2.0)
    best_a, best_s = 0.0, -1.0
    a = -SKEW_RANGE_DEG
    while a <= SKEW_RANGE_DEG + 1e-9:
        m = cv2.getRotationMatrix2D(centre, a, 1.0)
        s = _projection_score(cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        if s > best_s:
            best_s, best_a = s, a
        a += SKEW_STEP_DEG
    return round(best_a, 3)


def upright(img: Image.Image, use_osd: bool = True, use_deskew: bool = True) -> Image.Image:
    """Rotate to reading orientation, then level the text rows. Returns RGB."""
    a = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    if use_osd:
        deg, conf = osd_rotation(gray)
        # Abstain rather than guess: a low-confidence reading leaves the page untouched.
        if deg and conf >= OSD_MIN_CONFIDENCE:
            a = _rotate_cw(a, deg)
            gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    if use_deskew:
        angle = estimate_skew(gray)
        if abs(angle) >= 1e-6:
            h, w = a.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
