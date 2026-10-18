"""
evaluate.py
───────────
Accuracy evaluation script for SlipSense.

Usage:
    python evaluate.py --receipts ./synthetic_receipts --ground-truth ./synthetic_receipts/ground_truth.csv

This script:
  1. Reads ground_truth.csv (columns: filename, vendor, date, subtotal, tax, total, currency, category)
  2. Runs the full extraction pipeline on each file
  3. Compares extracted values against ground truth
  4. Prints field-level accuracy for each field
  5. Reports average seconds per file and list of failures (expected vs actual)
  6. Saves eval_report.md

IMPORTANT: Reports only real measured numbers. Never fabricates or estimates.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.WARNING)


# ── Match helpers ──────────────────────────────────────────────────────────────

def _vendor_match(expected: str, actual: Optional[str], threshold: float = 0.8) -> bool:
    """Jaccard similarity on character bigrams, case-insensitive."""
    if actual is None:
        return False

    def ngrams(s: str, n: int = 2) -> set:
        s = s.lower().strip()
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    a_set = ngrams(expected)
    b_set = ngrams(actual)
    if not a_set and not b_set:
        return True
    if not a_set or not b_set:
        return False
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return (intersection / union) >= threshold


def _date_match(expected: str, actual: Optional[str]) -> bool:
    """Exact ISO YYYY-MM-DD match after stripping whitespace."""
    return (actual or "").strip() == expected.strip()


def _numeric_match(expected: float, actual: Optional[float], tol: float = 0.01) -> bool:
    """Absolute tolerance of 0.01."""
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (ValueError, TypeError):
        return False


def _category_match(expected: str, actual: Optional[str]) -> bool:
    """Case-insensitive exact match."""
    return (actual or "").strip().lower() == expected.strip().lower()


def _currency_match(expected: str, actual: Optional[str]) -> bool:
    """Case-insensitive exact match."""
    return (actual or "").strip().upper() == expected.strip().upper()


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(receipts_dir: Path, ground_truth_csv: Path, config_path: Path) -> None:
    # ── Load config ────────────────────────────────────────────────────────────
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ollama_url = cfg["ollama_url"]
    model_name = cfg["model_name"]
    temperature = cfg.get("temperature", 0)

    from extraction import (
        MAX_SIDE,
        build_system_prompt,
        extract_receipt,
        load_image_as_pil,
        render_pdf_pages,
        save_image_as_jpeg,
    )
    from validation import validate_row
    from ollama_client import check_ollama, is_model_available

    # ── Check Ollama ───────────────────────────────────────────────────────────
    status = check_ollama(ollama_url)
    if not status["running"]:
        print(f"❌ Ollama is not running: {status['error']}")
        return

    found, _ = is_model_available(ollama_url, model_name)
    if not found:
        print(f"❌ Model '{model_name}' not found. Run: ollama pull {model_name}")
        return

    print(f"✅ Ollama running | Model: {model_name}")
    print(f"📂 Receipts dir: {receipts_dir}")
    print(f"📋 Ground truth: {ground_truth_csv}\n")

    # ── Load ground truth ──────────────────────────────────────────────────────
    gt_rows: list[dict] = []
    with open(ground_truth_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_rows.append(row)

    if not gt_rows:
        print("❌ Ground truth CSV is empty.")
        return

    categories_path = Path("categories.yaml")
    examples_path = Path("examples.yaml")
    cats_data = yaml.safe_load(categories_path.read_text()) if categories_path.exists() else {}
    allowed_cats = list((cats_data.get("categories", {})).keys())
    system_prompt = build_system_prompt(categories_path, examples_path)

    # ── Field counters (§7) ────────────────────────────────────────────────────
    counters = {
        "vendor": 0,
        "date": 0,
        "subtotal": 0,
        "tax": 0,
        "total": 0,
        "currency": 0,
        "category": 0,
    }
    n = 0
    flagged_count = 0
    elapsed_times: list[float] = []
    failures: list[dict] = []

    for gt in gt_rows:
        fname = gt["filename"]
        fpath = receipts_dir / fname

        if not fpath.exists():
            print(f"⚠️  File not found: {fpath}")
            continue

        n += 1
        t0 = time.time()
        result = extract_receipt(
            fpath,
            ollama_url=ollama_url,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            config_path=config_path,
        )

        elapsed = time.time() - t0
        elapsed_times.append(elapsed)

        extracted = None
        if result["data"]:
            extracted = result["data"].model_dump()
            extracted["_retried"] = result["retried"]

        if extracted is None:
            err_msg = result.get("error") or "extraction returned nothing"
            print(f"  ❌ {fname}: {err_msg}")
            failures.append({
                "filename": fname,
                "field": "all",
                "expected": "–",
                "actual": "extraction failed",
            })
            flagged_count += 1
            continue

        # Validate to apply post-processing (derive missing fields)
        v_row = validate_row(extracted, allowed_cats, retried=extracted.get("_retried", False))
        if v_row.get("status") == "Needs review":
            flagged_count += 1

        # ── Field accuracy ─────────────────────────────────────────────────────
        def _gt_float(key: str) -> Optional[float]:
            val = gt.get(key, "").strip()
            if val == "" or val.lower() == "null":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        exp_vendor   = gt.get("vendor", "")
        exp_date     = gt.get("date", "")
        exp_subtotal = _gt_float("subtotal")
        exp_tax      = _gt_float("tax")
        exp_total    = _gt_float("total")
        exp_currency = gt.get("currency", "")
        exp_cat      = gt.get("category", "")

        act_vendor   = v_row.get("vendor")
        act_date     = v_row.get("date")   # normalised ISO
        act_subtotal = v_row.get("subtotal")
        act_tax      = v_row.get("tax")
        act_total    = v_row.get("total")
        act_currency = v_row.get("currency")
        act_cat      = v_row.get("category")

        results_map = {
            "vendor":   _vendor_match(exp_vendor, act_vendor),
            "date":     _date_match(exp_date, act_date),
            "subtotal": _numeric_match(exp_subtotal, act_subtotal) if exp_subtotal is not None else True,
            "tax":      _numeric_match(exp_tax, act_tax) if exp_tax is not None else True,
            "total":    _numeric_match(exp_total, act_total) if exp_total is not None else True,
            "currency": _currency_match(exp_currency, act_currency) if exp_currency else True,
            "category": _category_match(exp_cat, act_cat),
        }

        for field, ok in results_map.items():
            counters[field] += int(ok)

        # Record failures
        row_failures = []
        exp_map = {
            "vendor": exp_vendor, "date": exp_date, "subtotal": exp_subtotal,
            "tax": exp_tax, "total": exp_total, "currency": exp_currency, "category": exp_cat,
        }
        act_map = {
            "vendor": act_vendor, "date": act_date, "subtotal": act_subtotal,
            "tax": act_tax, "total": act_total, "currency": act_currency, "category": act_cat,
        }
        for field, ok in results_map.items():
            if not ok:
                row_failures.append({
                    "filename": fname,
                    "field": field,
                    "expected": exp_map[field],
                    "actual": act_map[field],
                })

        failures.extend(row_failures)

        status_icon = "✅" if not row_failures else "⚠️"
        field_icons = " ".join(
            f"{f}={'✓' if ok else '✗'}" for f, ok in results_map.items()
        )
        print(f"  {status_icon} {fname} ({elapsed:.1f}s) | {field_icons}")

    if n == 0:
        print("No files were processed.")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    avg_sec = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    print("\n" + "═" * 70)
    print("EVALUATION RESULTS")
    print("═" * 70)
    print(f"Files processed   : {n}")
    print(f"Rows flagged      : {flagged_count} ({100*flagged_count/n:.0f}%)")
    print(f"Avg sec/file      : {avg_sec:.1f}s")
    print()
    for field in ["vendor", "date", "subtotal", "tax", "total", "currency", "category"]:
        hits = counters[field]
        print(f"{field:12s} accuracy : {hits}/{n}  ({100*hits/n:.1f}%)")
    print("═" * 70)

    if failures:
        print("\nFAILURES:")
        for fail in failures:
            print(
                f"  {fail['filename']} | {fail['field']:10s} | "
                f"expected={str(fail['expected'])!r:30s} | actual={str(fail['actual'])!r}"
            )

    # ── Write eval_report.md ───────────────────────────────────────────────────
    report_lines = [
        "# SlipSense Evaluation Report",
        "",
        f"**Model:** `{model_name}`  ",
        f"**Receipts directory:** `{receipts_dir}`  ",
        f"**Ground truth:** `{ground_truth_csv}`  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Files processed | {n} |",
        f"| Rows flagged for review | {flagged_count} ({100*flagged_count/n:.0f}%) |",
        f"| Avg seconds per file | {avg_sec:.1f}s |",
        "",
        "## Field-Level Accuracy",
        "",
        "| Field | Correct | Total | Accuracy | Match Method |",
        "|-------|---------|-------|----------|--------------|",
        f"| vendor   | {counters['vendor']}   | {n} | {100*counters['vendor']/n:.1f}%   | Jaccard bigrams ≥80% |",
        f"| date     | {counters['date']}     | {n} | {100*counters['date']/n:.1f}%     | Exact ISO YYYY-MM-DD |",
        f"| subtotal | {counters['subtotal']} | {n} | {100*counters['subtotal']/n:.1f}% | Absolute ±0.01 |",
        f"| tax      | {counters['tax']}      | {n} | {100*counters['tax']/n:.1f}%      | Absolute ±0.01 |",
        f"| total    | {counters['total']}    | {n} | {100*counters['total']/n:.1f}%    | Absolute ±0.01 |",
        f"| currency | {counters['currency']} | {n} | {100*counters['currency']/n:.1f}% | Case-insensitive exact |",
        f"| category | {counters['category']} | {n} | {100*counters['category']/n:.1f}% | Case-insensitive exact |",
        "",
        "## Failures",
        "",
    ]

    if failures:
        report_lines.append("| File | Field | Expected | Actual |")
        report_lines.append("|------|-------|----------|--------|")
        for fail in failures:
            report_lines.append(
                f"| {fail['filename']} | {fail['field']} | "
                f"`{fail['expected']}` | `{fail['actual']}` |"
            )
    else:
        report_lines.append("_No failures! All fields matched._")

    report_lines += [
        "",
        "## Notes",
        "",
        "- vendor: Jaccard similarity on character bigrams (threshold 80%).",
        "- date: Exact ISO YYYY-MM-DD string comparison after normalisation.",
        "- subtotal/tax/total: Absolute tolerance ±0.01.",
        "- currency: Case-insensitive exact match.",
        "- category: Case-insensitive exact string comparison.",
        "- These numbers reflect real pipeline output. No values are fabricated.",
    ]

    report_path = Path("eval_report.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n📄 Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SlipSense extraction accuracy.")
    parser.add_argument(
        "--receipts",
        type=Path,
        default=Path("synthetic_receipts"),
        help="Directory of receipt images",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("synthetic_receipts/ground_truth.csv"),
        help="Path to ground_truth.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    run_evaluation(args.receipts, args.ground_truth, args.config)
