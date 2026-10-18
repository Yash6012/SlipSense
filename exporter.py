"""
exporter.py
───────────
Handles all output operations for SlipSense:

  - expenses.csv:         9 columns in the specified order
  - category_summary.csv: Category | Currency | Receipts | Sub Total | Tax | Total
  - Copies (never moves) originals to output/<Category>/YYYY-MM-DD_Vendor.ext
    with numeric suffixes on collision and sanitized characters.

IMPORTANT: Never modifies or deletes original files.
"""

import csv
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# §6: exactly these 9 columns in this order
CSV_COLUMNS = [
    "Status",
    "Vendor",
    "Sub Total",
    "Tax",
    "Total",
    "Currency",
    "Category",
    "Date",
    "Reason",
]

# Characters illegal in common filesystems (Windows is strictest)
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')


# ── Filename helpers ───────────────────────────────────────────────────────────

def sanitize_segment(text: str, max_len: int = 40) -> str:
    """
    Sanitize a string for use as a filename segment:
      - Replace illegal chars with '_'
      - Collapse whitespace/underscores
      - Truncate to max_len
    """
    if not text:
        return "Unknown"
    cleaned = _ILLEGAL_CHARS.sub("_", text)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len] if cleaned else "Unknown"


def build_new_filename(row: dict, original_path: Path, existing: set[str]) -> str:
    """
    Build:  YYYY-MM-DD_Vendor.ext  (no price in filename per §6)
    Use "NA" for missing date; numeric suffix on collision.
    """
    date_val = row.get("date")
    date_part = sanitize_segment(date_val if date_val else "NA")
    vendor_part = sanitize_segment(row.get("vendor") or "Unknown-Vendor")
    ext = original_path.suffix.lower()

    base = f"{date_part}_{vendor_part}"
    candidate = f"{base}{ext}"

    counter = 2
    while candidate in existing:
        candidate = f"{base}_{counter}{ext}"
        counter += 1

    existing.add(candidate)
    return candidate


# ── Copy files into category subfolders ───────────────────────────────────────

def copy_categorized_files(rows: list[dict], output_folder: Path) -> list[dict]:
    """
    Copy original files to output_folder/<Category>/YYYY-MM-DD_Vendor.ext
    Adds/updates "new_filename" and "new_filepath" keys in each row.
    Never moves or deletes originals.
    """
    # Track used names per category folder to handle collisions
    used_names: dict[str, set[str]] = {}

    for row in rows:
        original_path = Path(row.get("original_filepath", ""))
        if not original_path.exists():
            logger.warning("Original file not found: %s", original_path)
            row["new_filename"] = ""
            continue

        category = sanitize_segment(row.get("category") or "Uncategorized")
        cat_dir = output_folder / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        if category not in used_names:
            used_names[category] = set()

        new_name = build_new_filename(row, original_path, used_names[category])
        dest = cat_dir / new_name

        try:
            shutil.copy2(str(original_path), str(dest))
            row["new_filename"] = new_name
            row["new_filepath"] = str(dest)
            logger.info("Copied %s → %s/%s", original_path.name, category, new_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to copy %s: %s", original_path, exc)
            row["new_filename"] = ""

    return rows


# ── Row → display dict helper ──────────────────────────────────────────────────

def _reasons_str(r: Any) -> str:
    if isinstance(r, list):
        return "; ".join(r)
    return str(r) if r else ""


def row_to_display(row: dict) -> dict:
    """Convert an internal result row to the 9-column display/export format."""
    date_val = row.get("date")
    return {
        "Status":     row.get("status", "OK"),
        "Vendor":     row.get("vendor") or "",
        "Sub Total":  row.get("subtotal"),
        "Tax":        row.get("tax"),
        "Total":      row.get("total"),
        "Currency":   row.get("currency") or "",
        "Category":   row.get("category") or "Uncategorized",
        "Date":       date_val if date_val else "NA",
        "Reason":     _reasons_str(row.get("reasons", [])),
    }


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(rows: list[dict], output_folder: Path) -> Path:
    """Write expenses.csv with the 9 standard columns."""
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / "expenses.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_display(row))

    logger.info("Exported %d rows to %s", len(rows), csv_path)
    return csv_path


# ── Category summary ──────────────────────────────────────────────────────────

def compute_category_summary(rows: list[dict]) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      Category | Currency | Receipts | Sub Total | Tax | Total

    One row per (category, currency) combination.
    Grand-total rows (one per currency) are appended at the bottom.
    Different currencies are never summed together (§5).
    """
    records = []
    for row in rows:
        cat = row.get("category") or "Uncategorized"
        curr = (row.get("currency") or "").strip() or "?"
        total = row.get("total")
        subtotal = row.get("subtotal")
        tax = row.get("tax")
        try:
            records.append({
                "Category": cat,
                "Currency": curr,
                "Sub Total": float(subtotal) if subtotal is not None else 0.0,
                "Tax":       float(tax) if tax is not None else 0.0,
                "Total":     float(total) if total is not None else 0.0,
            })
        except (TypeError, ValueError):
            pass

    if not records:
        return pd.DataFrame(columns=["Category", "Currency", "Receipts", "Sub Total", "Tax", "Total"])

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["Category", "Currency"], as_index=False)
        .agg(
            Receipts=("Total", "count"),
            **{"Sub Total": ("Sub Total", "sum"),
               "Tax":       ("Tax",       "sum"),
               "Total":     ("Total",     "sum")},
        )
        .sort_values(["Currency", "Category"])
        .reset_index(drop=True)
    )

    # Grand total per currency
    grand_rows = []
    for curr, grp in summary.groupby("Currency"):
        grand_rows.append({
            "Category":  "── TOTAL ──",
            "Currency":  curr,
            "Receipts":  int(grp["Receipts"].sum()),
            "Sub Total": grp["Sub Total"].sum(),
            "Tax":       grp["Tax"].sum(),
            "Total":     grp["Total"].sum(),
        })

    if grand_rows:
        grand_df = pd.DataFrame(grand_rows)
        summary = pd.concat([summary, grand_df], ignore_index=True)

    # Round numeric columns
    for col in ["Sub Total", "Tax", "Total"]:
        summary[col] = summary[col].round(2)

    return summary[["Category", "Currency", "Receipts", "Sub Total", "Tax", "Total"]]


def compute_category_totals(rows: list[dict]) -> pd.DataFrame:
    """Group rows by category and sum 'Total Spent', sorted descending."""
    counts: dict[str, float] = {}
    for r in rows:
        val = r.get("total")
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        cat = r.get("category") or "Uncategorized"
        counts[cat] = counts.get(cat, 0.0) + val
    if not counts:
        return pd.DataFrame(columns=["Category", "Total Spent"])
    df = pd.DataFrame([{"Category": k, "Total Spent": v} for k, v in counts.items()])
    return df.sort_values(by="Total Spent", ascending=False).reset_index(drop=True)


def export_category_summary(rows: list[dict], output_folder: Path) -> tuple[pd.DataFrame, Path]:
    """Write category_summary.csv and return (dataframe, path)."""
    output_folder.mkdir(parents=True, exist_ok=True)
    df = compute_category_summary(rows)
    csv_path = output_folder / "category_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Category summary saved to %s", csv_path)
    return df, csv_path


# ── Full export pipeline ──────────────────────────────────────────────────────

def run_export(rows: list[dict], output_folder: Path) -> tuple[Path, pd.DataFrame, Path]:
    """
    Copy files into category subfolders, write expenses.csv and category_summary.csv.
    Returns (expenses_csv_path, summary_df, summary_csv_path).
    """
    copy_categorized_files(rows, output_folder)
    csv_path = export_csv(rows, output_folder)
    summary_df, summary_csv = export_category_summary(rows, output_folder)
    return csv_path, summary_df, summary_csv
