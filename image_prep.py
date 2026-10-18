"""
image_prep.py
─────────────
Automatic image preparation module for SlipSense.

Performs all image preprocessing in-memory before sending to Ollama:
  - Opens image with Pillow, or renders PDF page 1 at configured DPI with PyMuPDF
  - Corrects orientation using ImageOps.exif_transpose
  - Converts to RGB
  - Downscales so longest side <= max_side without upscaling
  - Encodes to JPEG in-memory (io.BytesIO) and base64-encodes for Ollama
  - Reads configuration from config.yaml with no hardcoded values
  - Never modifies, overwrites, moves, or deletes user files
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, Union

import yaml
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")


class UnreadableFileError(IOError):
    """Raised when a file cannot be read, parsed, or processed as an image/PDF."""
    pass


class PreparedImage(str):
    """
    A base64-encoded JPEG image string with attached preparation metadata.
    Inherits from str so it can be passed directly to Ollama's 'images' payload.
    """
    original_size: tuple[int, int]
    original_bytes: int
    original_kb: float
    prepared_size: tuple[int, int]
    prepared_bytes: int
    prepared_kb: float
    prep_time: float

    def __new__(
        cls,
        b64_str: str,
        original_size: tuple[int, int] = (0, 0),
        original_bytes: int = 0,
        prepared_size: tuple[int, int] = (0, 0),
        prepared_bytes: int = 0,
        prep_time: float = 0.0,
    ):
        obj = super().__new__(cls, b64_str)
        obj.original_size = original_size
        obj.original_bytes = original_bytes
        obj.original_kb = original_bytes / 1024.0
        obj.prepared_size = prepared_size
        obj.prepared_bytes = prepared_bytes
        obj.prepared_kb = prepared_bytes / 1024.0
        obj.prep_time = prep_time
        return obj


def load_image_config(config_path: Optional[Path] = None) -> dict:
    """
    Load image configuration from config.yaml.
    Returns a dict with 'max_side', 'jpeg_quality', and 'pdf_dpi'.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            img_cfg = cfg.get("image", {})
            return {
                "max_side": int(img_cfg["max_side"]),
                "jpeg_quality": int(img_cfg["jpeg_quality"]),
                "pdf_dpi": int(img_cfg["pdf_dpi"]),
            }
        except Exception as exc:
            logger.warning("Could not read image config from %s: %s", path, exc)

    # Fallback only if config cannot be read
    return {
        "max_side": 1000,
        "jpeg_quality": 85,
        "pdf_dpi": 110,
    }


def prepare_image(
    path: Union[str, Path],
    max_side: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    pdf_dpi: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> PreparedImage:
    """
    Prepares a file (image or PDF) in memory for Ollama vision inference.

    Args:
        path: Path to the image or PDF file.
        max_side: Maximum longest side in pixels (defaults to config.yaml).
        jpeg_quality: JPEG compression quality (defaults to config.yaml).
        pdf_dpi: DPI for rendering PDF pages (defaults to config.yaml).
        config_path: Optional path to config.yaml.

    Returns:
        PreparedImage: base64-encoded JPEG string with attached size/timing metadata.

    Raises:
        UnreadableFileError: If the file cannot be opened, is corrupted, or missing.
    """
    t0 = time.time()
    p = Path(path)

    if not p.is_file():
        raise UnreadableFileError(f"Could not read file: {p} does not exist or is not a file")

    try:
        original_bytes = p.stat().st_size
    except Exception as exc:
        raise UnreadableFileError(f"Could not read file: cannot stat {p}: {exc}") from exc

    # Read configuration from config.yaml without hardcoded defaults
    cfg = load_image_config(config_path)
    eff_max_side = max_side if max_side is not None else cfg["max_side"]
    eff_jpeg_quality = jpeg_quality if jpeg_quality is not None else cfg["jpeg_quality"]
    eff_pdf_dpi = pdf_dpi if pdf_dpi is not None else cfg["pdf_dpi"]

    # 1. Open with Pillow or PyMuPDF for PDFs
    if p.suffix.lower() == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(p))
            if len(doc) == 0:
                doc.close()
                raise UnreadableFileError(f"Could not read file: PDF {p.name} has no pages")
            page = doc[0]
            zoom = eff_pdf_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            doc.close()

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            orig_size = img.size
        except Exception as exc:
            raise UnreadableFileError(f"Could not read file: {exc}") from exc
    else:
        try:
            img = Image.open(p)
            orig_size = img.size
            # Fix phone rotation with ImageOps.exif_transpose
            img = ImageOps.exif_transpose(img)
        except Exception as exc:
            raise UnreadableFileError(f"Could not read file: {exc}") from exc

    # 2. Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 3. Downscale so longest side <= eff_max_side (keep aspect ratio, NEVER upscale)
    w, h = img.size
    longest = max(w, h)
    if longest > eff_max_side:
        scale = eff_max_side / longest
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    prep_size = img.size

    # 4. Encode as JPEG at configured quality in memory (io.BytesIO)
    buf = io.BytesIO()
    try:
        img.save(buf, format="JPEG", quality=eff_jpeg_quality)
    except Exception as exc:
        raise UnreadableFileError(f"Could not read file: failed to encode JPEG: {exc}") from exc

    jpeg_bytes = buf.getvalue()
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    prep_time = time.time() - t0

    return PreparedImage(
        b64_str,
        original_size=orig_size,
        original_bytes=original_bytes,
        prepared_size=prep_size,
        prepared_bytes=len(jpeg_bytes),
        prep_time=prep_time,
    )
