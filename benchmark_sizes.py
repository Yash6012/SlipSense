"""
benchmark_sizes.py
──────────────────
Benchmark extraction speed and accuracy across different image sizes (§6).

Runs the real Ollama pipeline on the same receipt files at:
  - max_side = 1400
  - max_side = 1000
  - max_side = 800

Measures:
  - average seconds per file (real measured elapsed time)
  - files flagged "Needs review"
  - total-field accuracy (if ground_truth.csv exists)

Saves the table and full report to benchmark_report.md.
Reports real measured results only. Never estimates or invents numbers.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# Support running directly from project root or inside a subfolder
_DIR = Path(__file__).resolve().parent
if (_DIR / "extraction.py").exists():
    sys.path.insert(0, str(_DIR))
elif (_DIR.parent / "extraction.py").exists():
    sys.path.insert(0, str(_DIR.parent))

from extraction import build_system_prompt, extract_receipt
from ollama_client import check_ollama, is_model_available
from validation import validate_row


# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")


def _numeric_match(expected: Optional[float], actual: Optional[float], tol: float = 0.01) -> bool:
    """Check if numbers match within 0.01 absolute tolerance."""
    if expected is None or actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (ValueError, TypeError):
        return False


def run_benchmark(
    receipts_dir: Path,
    ground_truth_path: Optional[Path],
    config_path: Path,
    sizes: list[int],
    limit: Optional[int] = None,
) -> dict:
    """Run extraction pipeline across multiple image sizes and measure results."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ollama_url = cfg.get("ollama_url", "http://localhost:11434")
    model_name = cfg.get("model_name", "qwen2.5vl:3b")
    temperature = cfg.get("temperature", 0)

    # Health check
    status = check_ollama(ollama_url)
    if not status["running"]:
        print(f"❌ Ollama is not running at {ollama_url}: {status['error']}")
        sys.exit(1)

    found, _ = is_model_available(ollama_url, model_name)
    if not found:
        print(f"❌ Model '{model_name}' not found in Ollama.")
        sys.exit(1)

    print(f"✅ Ollama running | Model: {model_name}")
    print(f"📁 Receipts: {receipts_dir}")

    # Load categories for validation
    cats_path = Path("categories.yaml")
    cats_data = yaml.safe_load(cats_path.read_text(encoding="utf-8")) if cats_path.exists() else {}
    allowed_cats = list((cats_data.get("categories", {})).keys())

    # Build system prompt
    system_prompt = build_system_prompt(cats_path, Path("examples.yaml"))

    # Load ground truth if available
    ground_truth: dict[str, dict] = {}
    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row["filename"]] = row
        print(f"📋 Loaded ground truth for {len(ground_truth)} files")
    else:
        print("ℹ️ No ground truth CSV provided or found")

    # Gather receipt files
    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
    receipt_files = sorted(
        [p for p in receipts_dir.iterdir() if p.suffix.lower() in supported_exts]
    )

    if not receipt_files:
        print(f"❌ No supported image or PDF files found in {receipts_dir}")
        sys.exit(1)

    if limit and limit > 0:
        receipt_files = receipt_files[:limit]

    print(f"📊 Running benchmark on {len(receipt_files)} files across sizes: {sizes}\n")

    results_by_size: dict[int, dict] = {}

    for size in sizes:
        print(f"── Testing max_side = {size} ─────────────────────────────")
        total_seconds = 0.0
        flagged_count = 0
        total_field_matches = 0
        gt_total_count = 0
        file_results = []

        for idx, fpath in enumerate(receipt_files):
            fname = fpath.name
            t0 = time.time()

            result = extract_receipt(
                fpath,
                ollama_url=ollama_url,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_side=size,
                config_path=config_path,
            )

            elapsed = time.time() - t0
            total_seconds += elapsed

            is_flagged = False
            extracted_dict = None

            if result.get("unreadable") or result.get("error") == "Could not read file":
                is_flagged = True
                flagged_count += 1
            elif result["data"] is None:
                is_flagged = True
                flagged_count += 1
            else:
                extracted_dict = result["data"].model_dump()
                v_row = validate_row(
                    extracted_dict,
                    allowed_categories=allowed_cats,
                    retried=result.get("retried", False),
                )
                if v_row.get("status") == "Needs review":
                    is_flagged = True
                    flagged_count += 1

            # Accuracy check against ground truth for the 'total' field
            matched_total = False
            gt_total_val = None
            extracted_total_val = None

            if fname in ground_truth:
                gt_row = ground_truth[fname]
                gt_total_raw = gt_row.get("total")
                if gt_total_raw not in (None, "", "null"):
                    try:
                        gt_total_val = float(gt_total_raw)
                        gt_total_count += 1
                        if extracted_dict and extracted_dict.get("total") is not None:
                            extracted_total_val = float(extracted_dict["total"])
                            if _numeric_match(gt_total_val, extracted_total_val):
                                matched_total = True
                                total_field_matches += 1
                    except (ValueError, TypeError):
                        pass

            file_results.append({
                "filename": fname,
                "elapsed": elapsed,
                "flagged": is_flagged,
                "gt_total": gt_total_val,
                "extracted_total": extracted_total_val,
                "matched_total": matched_total,
                "prep_sec": result.get("prep_seconds", 0.0),
                "ollama_sec": result.get("ollama_seconds", 0.0),
            })

            status_mark = "⚠️ Flagged" if is_flagged else "✅ OK"
            match_mark = ""
            if gt_total_val is not None:
                match_mark = " | Total match: YES" if matched_total else f" | Total match: NO (exp {gt_total_val}, got {extracted_total_val})"

            print(f"  [{idx+1}/{len(receipt_files)}] {fname}: {elapsed:.2f}s | {status_mark}{match_mark}")

        n_files = len(receipt_files)
        avg_sec = total_seconds / n_files if n_files > 0 else 0.0
        acc_pct = (total_field_matches / gt_total_count * 100.0) if gt_total_count > 0 else None

        results_by_size[size] = {
            "max_side": size,
            "n_files": n_files,
            "total_seconds": total_seconds,
            "avg_seconds": avg_sec,
            "flagged_count": flagged_count,
            "gt_total_count": gt_total_count,
            "total_field_matches": total_field_matches,
            "total_accuracy_pct": acc_pct,
            "file_results": file_results,
        }
        print()

    return results_by_size


def generate_report(results: dict, ground_truth_exists: bool) -> str:
    """Generate Markdown report and table."""
    lines = []
    lines.append("# Image Resolution Benchmark Report")
    lines.append("")
    lines.append("Measured performance and extraction accuracy across image resolutions (`max_side`):")
    lines.append("")

    if ground_truth_exists:
        lines.append("| max_side | average seconds per file | files flagged \"Needs review\" | total-field accuracy |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for size, data in results.items():
            acc_str = f"{data['total_accuracy_pct']:.1f}% ({data['total_field_matches']}/{data['gt_total_count']})" if data['total_accuracy_pct'] is not None else "N/A"
            lines.append(f"| {size} | {data['avg_seconds']:.2f}s | {data['flagged_count']} / {data['n_files']} | {acc_str} |")
    else:
        lines.append("| max_side | average seconds per file | files flagged \"Needs review\" |")
        lines.append("|:---:|:---:|:---:|")
        for size, data in results.items():
            lines.append(f"| {size} | {data['avg_seconds']:.2f}s | {data['flagged_count']} / {data['n_files']} |")

    lines.append("")
    lines.append("## Analysis & Recommendation")
    lines.append("")

    # Determine recommended size based strictly on measured results
    best_size = None
    best_score = (-1.0, 999999.0)  # (accuracy, avg_seconds)

    for size, data in results.items():
        acc = data["total_accuracy_pct"] if data["total_accuracy_pct"] is not None else (100.0 - (data["flagged_count"] / data["n_files"] * 100.0))
        sec = data["avg_seconds"]
        # Prioritize accuracy, then speed
        if acc > best_score[0] or (acc == best_score[0] and sec < best_score[1]):
            best_score = (acc, sec)
            best_size = size

    rec_data = results.get(best_size, {})
    lines.append(f"**Recommended `max_side`: {best_size}**")
    lines.append("")
    lines.append(f"- **Accuracy**: Achieved {rec_data.get('total_accuracy_pct', 0):.1f}% total-field accuracy with {rec_data.get('flagged_count', 0)} files flagged.")
    lines.append(f"- **Speed**: Averaged {rec_data.get('avg_seconds', 0):.2f} seconds per file.")
    lines.append("- **Rationale**: Balances visual clarity required by Ollama's vision model with minimal latency and memory footprint. Downscaling prevents out-of-memory errors on large images while preserving fine receipt details.")

    lines.append("")
    lines.append("## Detailed Measurements by File")
    lines.append("")
    for size, data in results.items():
        lines.append(f"### max_side = {size}")
        lines.append("| File | Total Elapsed | Prep Time | Ollama Time | Status | Expected Total | Extracted Total |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in data["file_results"]:
            st = "Needs review" if r["flagged"] else "OK"
            exp_t = f"{r['gt_total']:.2f}" if r["gt_total"] is not None else "-"
            ext_t = f"{r['extracted_total']:.2f}" if r["extracted_total"] is not None else "-"
            lines.append(f"| {r['filename']} | {r['elapsed']:.2f}s | {r['prep_sec']:.3f}s | {r['ollama_sec']:.3f}s | {st} | {exp_t} | {ext_t} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark extraction across image resolutions.")
    parser.add_argument("--receipts", type=Path, default=Path("synthetic_receipts"), help="Directory containing receipts")
    parser.add_argument("--ground-truth", type=Path, default=Path("synthetic_receipts/ground_truth.csv"), help="Ground truth CSV")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 1000, 800], help="Sizes to benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of receipts to benchmark")
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.md"), help="Report output file")
    args = parser.parse_args()

    gt_path = args.ground_truth if args.ground_truth.exists() else None
    results = run_benchmark(
        receipts_dir=args.receipts,
        ground_truth_path=gt_path,
        config_path=args.config,
        sizes=args.sizes,
        limit=args.limit,
    )

    report = generate_report(results, ground_truth_exists=(gt_path is not None))

    # Print summary table to console
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    if gt_path:
        print(f"{'max_side':<10} | {'avg seconds/file':<18} | {'files flagged':<15} | {'total-field accuracy':<20}")
        print("-" * 70)
        for size, d in results.items():
            acc_str = f"{d['total_accuracy_pct']:.1f}% ({d['total_field_matches']}/{d['gt_total_count']})"
            print(f"{size:<10} | {d['avg_seconds']:.2f}s{'':<12} | {d['flagged_count']}/{d['n_files']:<13} | {acc_str:<20}")
    else:
        print(f"{'max_side':<10} | {'avg seconds/file':<18} | {'files flagged':<15}")
        print("-" * 70)
        for size, d in results.items():
            print(f"{size:<10} | {d['avg_seconds']:.2f}s{'':<12} | {d['flagged_count']}/{d['n_files']:<13}")
    print("=" * 70)

    # Save to benchmark_report.md
    args.output.write_text(report, encoding="utf-8")
    print(f"\n📄 Saved full benchmark report to {args.output}")


if __name__ == "__main__":
    main()
