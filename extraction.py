"""
extraction.py
─────────────
Receipt/invoice field extraction pipeline.

Responsibilities:
  - Define the Pydantic model for a receipt (mirrors the JSON schema)
  - Build the system prompt (categories + few-shot examples injected)
  - Define the JSON schema sent to Ollama's "format" field
  - Coordinate: load image → call Ollama → parse → retry on failure → return result

Schema fields (v2):  vendor, date, currency, subtotal, tax, total, category
Removed:  document_type, line_items, payment_method, invoice_number, reasoning_note
"""

import hashlib
import io
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator

from ollama_client import extract_from_image

logger = logging.getLogger(__name__)


# ── Pydantic model ─────────────────────────────────────────────────────────────

class ReceiptData(BaseModel):
    """Mirrors the JSON schema sent to Ollama. All fields nullable."""

    vendor: Optional[str] = None
    date: Optional[str] = None       # ISO YYYY-MM-DD or null
    currency: Optional[str] = None   # e.g. INR, USD
    subtotal: Optional[float] = None # amount BEFORE tax
    tax: Optional[float] = None      # SUM of all tax lines (GST/VAT/CGST/SGST/IGST/cess)
    total: Optional[float] = None    # final payable AFTER tax
    category: Optional[str] = None   # must be one of categories.yaml

    @field_validator("total", "subtotal", "tax", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Accept string numbers by stripping currency symbols and commas."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace("₹", "").replace("Rs.", "").replace("Rs", "")
            cleaned = cleaned.replace("$", "").replace("£", "").replace("€", "")
            cleaned = cleaned.replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None


# ── JSON schema (sent in Ollama "format" field) ────────────────────────────────

RECEIPT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": ["string", "null"],
            "description": "Merchant or seller name exactly as printed",
        },
        "date": {
            "type": ["string", "null"],
            "description": (
                "Transaction date in YYYY-MM-DD format. "
                "Return null if no date is printed. "
                "Indian dates are day-first (DD/MM/YYYY); convert to YYYY-MM-DD."
            ),
        },
        "currency": {
            "type": ["string", "null"],
            "description": (
                "ISO 4217 currency code, e.g. INR, USD, EUR. "
                "₹ and Rs. mean INR."
            ),
        },
        "subtotal": {
            "type": ["number", "null"],
            "description": (
                "Amount BEFORE tax. Do NOT include any tax in this field. "
                "Return null if not clearly printed."
            ),
        },
        "tax": {
            "type": ["number", "null"],
            "description": (
                "SUM of ALL tax lines: GST, CGST, SGST, IGST, VAT, sales tax, cess. "
                "Do NOT include service charge, tips, or delivery fees. "
                "Return null if no tax line is printed."
            ),
        },
        "total": {
            "type": ["number", "null"],
            "description": (
                "FINAL amount payable INCLUDING all taxes. "
                "This is the bottom-line number the customer pays. "
                "Return null only if completely unreadable."
            ),
        },
        "category": {
            "type": ["string", "null"],
            "description": "Expense category chosen from the provided list. Return null if unsure.",
        },
    },
    "required": ["vendor", "date", "currency", "subtotal", "tax", "total", "category"],
    "additionalProperties": False,
}


# ── Prompt builders ────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def build_system_prompt(
    categories_path: Path = Path("categories.yaml"),
    examples_path: Path = Path("examples.yaml"),
) -> str:
    """
    Construct the system prompt injected into every Ollama call.
    Includes category definitions and few-shot vendor→category examples.
    """
    cats_data = _load_yaml(categories_path)
    examples_data = _load_yaml(examples_path)

    categories = cats_data.get("categories", {})
    examples = examples_data.get("examples", []) or []

    # Build category block
    cat_lines = ["Available categories (pick EXACTLY one of these):"]
    for name, desc in categories.items():
        if desc:
            cat_lines.append(f"  • {name}: {desc}")
        else:
            cat_lines.append(f"  • {name}")
    categories_block = "\n".join(cat_lines)

    # Build few-shot examples block
    if examples:
        ex_lines = ["\nVendor → Category examples (from user corrections):"]
        for ex in examples:
            ex_lines.append(f"  • \"{ex.get('vendor', '')}\" → \"{ex.get('category', '')}\"")
        examples_block = "\n".join(ex_lines)
    else:
        examples_block = ""

    return f"""You are a precise receipt and invoice data extraction assistant.
Your job is to read the document image and extract structured data as JSON.

STRICT RULES:
1. Read ONLY what is printed on the document. Never guess or invent values.
2. If a field is not clearly readable, return null for that field.
3. "total" is the FINAL amount actually paid or due (after all taxes, fees, and discounts).
   It is NOT the subtotal, NOT a single line item. It is the bottom-line payable amount.
4. "subtotal" is the amount BEFORE tax. Do NOT include tax in subtotal.
5. "tax" = SUM of GST + CGST + SGST + IGST + VAT + cess only.
   Do NOT include service charge, tips, or delivery fee in tax.
6. Amounts may use Indian number formatting (e.g. 1,23,456.00) — parse them correctly.
   Currency symbols ₹ and "Rs." both mean Indian Rupees → currency = "INR".
7. Dates: DD/MM/YYYY is Indian day-first convention. Convert to YYYY-MM-DD.
   If no date is printed, return null.
8. For category: choose EXACTLY one category from the list below. Do not invent new ones.
9. Return ONLY valid JSON. No markdown, no explanation, no extra text.

{categories_block}
{examples_block}

Return ONLY valid JSON matching the schema. No extra text, no markdown fences."""


# ── Image loading and pre-processing ──────────────────────────────────────────

from image_prep import (
    PreparedImage,
    UnreadableFileError,
    load_image_config,
    prepare_image,
)

MAX_SIDE = 1000  # Default fallback if config not found


def load_image_as_pil(image_path: Path, max_side: int = MAX_SIDE) -> Optional[Image.Image]:
    """
    Load an image file and downscale it if the longest side exceeds max_side.
    Returns a PIL Image (RGB) or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.debug("Resized %s from %dx%d to %dx%d", image_path.name, w, h, *img.size)
        return img
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load image %s: %s", image_path, exc)
        return None


def save_image_as_jpeg(img: Image.Image, quality: int = 85) -> Path:
    """
    Save a PIL Image as a temporary JPEG file and return the path.
    Caller is responsible for deleting the temp file.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tmp_path = Path(tf.name)
    img.save(str(tmp_path), format="JPEG", quality=quality, optimize=True)
    return tmp_path


def render_pdf_pages(
    pdf_path: Path, dpi: int = 110, all_pages: bool = False
) -> list[tuple[int, Path]]:
    """
    Render PDF page(s) to temporary JPEG files using PyMuPDF.
    Returns list of (page_number, temp_jpeg_path).
    Caller is responsible for deleting the temp files after use.
    """
    import fitz  # PyMuPDF  # noqa: PLC0415

    results = []
    try:
        doc = fitz.open(str(pdf_path))
        pages_to_render = range(len(doc)) if all_pages else range(min(1, len(doc)))
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in pages_to_render:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            tmp_path = save_image_as_jpeg(img)
            results.append((page_num, tmp_path))

        doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF render failed for %s: %s", pdf_path, exc)

    return results


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (used for cache keying)."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Parse and retry logic ─────────────────────────────────────────────────────

def _parse_receipt_json(raw: str) -> tuple[Optional[ReceiptData], Optional[str]]:
    """
    Try to parse a raw JSON string into a ReceiptData.
    Returns (receipt_data, error_message).
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(inner).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    known_keys = set(ReceiptData.model_fields.keys())
    obj = {k: v for k, v in obj.items() if k in known_keys}

    try:
        receipt = ReceiptData(**obj)
        return receipt, None
    except ValidationError as exc:
        errors = exc.errors()
        error_msg = "; ".join(f"{e['loc']}: {e['msg']}" for e in errors[:3])
        try:
            receipt = ReceiptData.model_construct(**obj)
            return receipt, f"Validation warnings: {error_msg}"
        except Exception:  # noqa: BLE001
            return None, f"Validation failed: {error_msg}"


def extract_receipt(
    image_path: Path | str,
    ollama_url: str,
    model_name: str,
    system_prompt: str,
    temperature: float = 0,
    max_side: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    pdf_dpi: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Full extraction pipeline for a receipt file (image or PDF).

    Guarantees:
      - Always prepares image in memory via prepare_image() before Ollama
      - Never sends raw original image to Ollama
      - Logs original vs prepared sizes and preparation vs Ollama seconds
      - Handles corrupt/unreadable files safely without crashing

    Returns a dict with:
      {
        "data": ReceiptData | None,
        "raw": str | None,
        "error": str | None,
        "retried": bool,
        "retry_error": str | None,
        "unreadable": bool,
        "original_size": tuple[int, int],
        "original_kb": float,
        "prepared_size": tuple[int, int],
        "prepared_kb": float,
        "prep_seconds": float,
        "ollama_seconds": float,
      }
    """
    p = Path(image_path)
    result: dict[str, Any] = {
        "data": None,
        "raw": None,
        "error": None,
        "retried": False,
        "retry_error": None,
        "unreadable": False,
        "original_size": None,
        "original_kb": None,
        "prepared_size": None,
        "prepared_kb": None,
        "prep_seconds": 0.0,
        "ollama_seconds": 0.0,
    }

    # ── Step 1: Automatic Image Preparation in memory ──────────────────────────
    t_prep_start = time.time()
    try:
        prep_img = prepare_image(
            p,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            pdf_dpi=pdf_dpi,
            config_path=config_path,
        )
        prep_sec = time.time() - t_prep_start
    except Exception as exc:
        logger.error("Could not read file %s: %s", p.name, exc)
        result["error"] = "Could not read file"
        result["unreadable"] = True
        result["prep_seconds"] = round(time.time() - t_prep_start, 3)
        return result

    result["original_size"] = prep_img.original_size
    result["original_kb"] = round(prep_img.original_kb, 1)
    result["prepared_size"] = prep_img.prepared_size
    result["prepared_kb"] = round(prep_img.prepared_kb, 1)
    result["prep_seconds"] = round(prep_sec, 3)

    # ── Step 2: Call Ollama with prepared image (never raw original) ───────────
    user_msg = "Extract all fields from this receipt or invoice image."

    t_ollama_start = time.time()
    raw, err = extract_from_image(
        ollama_url,
        model_name,
        prep_img,
        system_prompt,
        user_msg,
        RECEIPT_JSON_SCHEMA,
        temperature,
    )
    ollama_sec = time.time() - t_ollama_start
    result["ollama_seconds"] = round(ollama_sec, 3)

    # Log size & timing measurements (§4)
    logger.info(
        "File: %s | Original: %dx%d (%.1f KB) | Prepared: %dx%d (%.1f KB) | Prep: %.3fs | Ollama: %.3fs",
        p.name,
        prep_img.original_size[0],
        prep_img.original_size[1],
        prep_img.original_kb,
        prep_img.prepared_size[0],
        prep_img.prepared_size[1],
        prep_img.prepared_kb,
        prep_sec,
        ollama_sec,
    )

    if err:
        result["error"] = err
        return result

    result["raw"] = raw
    receipt, parse_err = _parse_receipt_json(raw)

    if receipt and not parse_err:
        result["data"] = receipt
        return result

    # ── Step 3: Retry once with corrective message if JSON parse failed ───────
    result["retried"] = True
    result["retry_error"] = parse_err or "Could not parse response"
    logger.warning("Retrying %s. Reason: %s", p.name, result["retry_error"])

    corrective_msg = (
        f"Your previous response had an issue: {parse_err}. "
        "Please respond with ONLY valid JSON matching the schema exactly. "
        "No markdown, no extra text. All numeric fields must be plain numbers (no currency symbols). "
        "If a field is unknown, use null. "
        "Fields: vendor, date, currency, subtotal, tax, total, category."
    )

    t_retry_start = time.time()
    raw2, err2 = extract_from_image(
        ollama_url,
        model_name,
        prep_img,
        system_prompt,
        corrective_msg,
        RECEIPT_JSON_SCHEMA,
        temperature,
    )
    retry_sec = time.time() - t_retry_start
    result["ollama_seconds"] = round(result["ollama_seconds"] + retry_sec, 3)

    if err2:
        result["error"] = f"Retry also failed: {err2}"
        if receipt:
            result["data"] = receipt
        return result

    result["raw"] = raw2
    receipt2, parse_err2 = _parse_receipt_json(raw2)

    if receipt2:
        result["data"] = receipt2
    elif receipt:
        result["data"] = receipt
        result["error"] = f"Retry parse failed ({parse_err2}); using partial first-attempt data"
    else:
        result["error"] = f"Both attempts failed. Last error: {parse_err2}"

    return result

