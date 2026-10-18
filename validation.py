"""
validation.py
─────────────
Rule-based validation of extracted receipt data.

Post-processing rules (§2) and status/reason logic (§3) — my code, not the model.
All rules are explicit and deterministic. Model confidence is never used.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Tolerance for arithmetic checks (1%)
ARITHMETIC_TOLERANCE = 0.01


# ── Date helpers ───────────────────────────────────────────────────────────────

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Try to parse a date string into a Python date object.
    Accepts YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY, natural language.
    Returns None if parsing fails.
    """
    if not date_str:
        return None

    s = date_str.strip()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    for fmt in ("%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    return None


# ── Arithmetic helpers ─────────────────────────────────────────────────────────

def _within_tolerance(a: float, b: float, tol: float = ARITHMETIC_TOLERANCE) -> bool:
    """Return True if |a - b| / max(|b|, 1) <= tol."""
    if b == 0 and a == 0:
        return True
    denominator = max(abs(b), 1.0)
    return abs(a - b) / denominator <= tol


# ── Duplicate detection ────────────────────────────────────────────────────────

def find_duplicates(rows: list[dict]) -> set[int]:
    """
    Return the set of row indices that are likely duplicates.
    A duplicate is two rows sharing the same vendor + date + total.
    """
    seen: dict[tuple, int] = {}
    dupes: set[int] = set()
    for i, row in enumerate(rows):
        vendor = (row.get("vendor") or "").strip().lower()
        date_val = row.get("date") or ""
        total = row.get("total")
        if vendor and date_val and total is not None:
            key = (vendor, date_val, round(float(total), 2))
            if key in seen:
                dupes.add(i)
                dupes.add(seen[key])
            else:
                seen[key] = i
    return dupes


# ── Main validation function ───────────────────────────────────────────────────

def validate_row(
    data: dict,
    allowed_categories: list[str],
    is_duplicate: bool = False,
    retried: bool = False,
) -> dict:
    """
    Validate a single extracted row and apply post-processing rules (§2).

    Returns the row dict with added keys:
      - "status":  "OK" | "Needs review"
      - "reasons": list[str]  — plain-English, will be joined with '; '

    Needs review triggers (§3):
      - vendor missing
      - total missing / zero / negative
      - subtotal + tax differs from total by > 1% (mention service charge/rounding)
      - category not in list (set Uncategorized)
      - date in the future
      - possible duplicate
      - retry was needed

    Date null alone does NOT make the row "Needs review" (§2).
    """
    row = dict(data)   # shallow copy — never mutate caller's dict
    reasons: list[str] = []

    # ── 1. Date ───────────────────────────────────────────────────────────────
    date_str = row.get("date")
    parsed_date = parse_date(date_str)

    if not date_str or parsed_date is None:
        # Date null → display "NA", add note, but NOT a "Needs review" trigger (§2/§3)
        row["date"] = None   # keep as None; table will show "NA"
        reasons_date_note = "No date found"
        # Append to a separate non-flagging notes list we'll merge later
        row["_date_note"] = reasons_date_note
    else:
        today = date.today()
        if parsed_date > today:
            reasons.append(f"Date {parsed_date.isoformat()} is in the future")
        row["date"] = parsed_date.isoformat()

    # ── 2. Total ──────────────────────────────────────────────────────────────
    total = row.get("total")
    total_f: Optional[float] = None
    if total is None:
        reasons.append("Total amount is missing")
    else:
        try:
            total_f = float(total)
            if total_f <= 0:
                reasons.append(f"Total is zero or negative ({total_f})")
            else:
                row["total"] = total_f
        except (ValueError, TypeError):
            reasons.append(f"Total is not numeric ({total!r})")
            total_f = None

    # ── 3. Vendor ─────────────────────────────────────────────────────────────
    vendor = (row.get("vendor") or "").strip()
    if not vendor:
        reasons.append("Vendor name is missing")

    # ── 4. Post-processing: derive missing tax / subtotal (§2) ─────────────────
    subtotal = row.get("subtotal")
    tax = row.get("tax")

    subtotal_f: Optional[float] = None
    tax_f: Optional[float] = None

    try:
        subtotal_f = float(subtotal) if subtotal is not None else None
    except (ValueError, TypeError):
        subtotal_f = None

    try:
        tax_f = float(tax) if tax is not None else None
    except (ValueError, TypeError):
        tax_f = None

    # Rule §2-a: tax null but subtotal and total present → calculate tax
    if tax_f is None and subtotal_f is not None and total_f is not None:
        tax_f = round(total_f - subtotal_f, 4)
        row["tax"] = tax_f
        row.setdefault("_pp_notes", []).append("Tax calculated from total minus subtotal")

    # Rule §2-b: subtotal null but total and tax present → calculate subtotal
    elif subtotal_f is None and tax_f is not None and total_f is not None:
        subtotal_f = round(total_f - tax_f, 4)
        row["subtotal"] = subtotal_f
        row.setdefault("_pp_notes", []).append("Subtotal calculated from total minus tax")

    # Rule §2-c: no tax printed and subtotal == total → tax = 0
    if tax is None and subtotal_f is not None and total_f is not None:
        if abs(subtotal_f - total_f) < 0.01:
            tax_f = 0.0
            row["tax"] = 0.0

    # Update row numeric fields
    if subtotal_f is not None:
        row["subtotal"] = subtotal_f
    if tax_f is not None:
        row["tax"] = tax_f

    # ── 5. Arithmetic check: subtotal + tax ≈ total (§3) ─────────────────────
    if total_f is not None and subtotal_f is not None and tax_f is not None:
        computed = subtotal_f + tax_f
        if not _within_tolerance(computed, total_f):
            diff_pct = abs(computed - total_f) / max(abs(total_f), 1.0) * 100
            reasons.append(
                f"Subtotal ({subtotal_f:.2f}) + tax ({tax_f:.2f}) = {computed:.2f} "
                f"but total is {total_f:.2f} ({diff_pct:.1f}% difference — "
                "possible service charge or rounding)"
            )

    # ── 6. Category ───────────────────────────────────────────────────────────
    category = (row.get("category") or "").strip()
    allowed_lower = {c.lower(): c for c in allowed_categories}
    if not category or category.lower() not in allowed_lower:
        if category and category.lower() not in allowed_lower:
            reasons.append(f"Category '{category}' not in the allowed list")
        row["category"] = "Uncategorized"
    else:
        row["category"] = allowed_lower[category.lower()]

    # ── 7. Duplicate ─────────────────────────────────────────────────────────
    if is_duplicate:
        reasons.append(
            "Possible duplicate: same vendor, date, and total as another file"
        )

    # ── 8. Retry was needed ───────────────────────────────────────────────────
    if retried:
        reasons.append("A retry was needed; extraction may be less reliable")

    # ── Merge post-processing notes into reasons (non-flagging, appended last) ─
    pp_notes = row.pop("_pp_notes", [])
    date_note = row.pop("_date_note", None)
    all_reasons = list(reasons)  # flagging reasons
    if date_note:
        all_reasons.append(date_note)
    all_reasons.extend(pp_notes)

    # ── Final status (§3): only the "real" flagging reasons count ─────────────
    row["status"] = "Needs review" if reasons else "OK"
    row["reasons"] = all_reasons   # full list for display

    return row
