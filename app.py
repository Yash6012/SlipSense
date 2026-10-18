"""
app.py
──────
Shoebox – Receipt & Invoice Sorter for Freelancers
Streamlit single-page UI.

Run:  streamlit run app.py
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from exporter import (
    compute_category_summary,
    row_to_display,
    run_export,
    _reasons_str,
)
from extraction import (
    build_system_prompt,
    extract_receipt,
    file_hash,
    load_image_as_pil,
    render_pdf_pages,
    save_image_as_jpeg,
    MAX_SIDE,
)
from ollama_client import check_ollama, is_model_available
from validation import find_duplicates, validate_row

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("slipsense.app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SlipSense – Receipt Sorter",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark card-style sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #cbd5e1 !important;
}

/* Accent colour for buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Status badges */
.badge-ok {
    display: inline-block;
    background: #22c55e22;
    color: #16a34a;
    border: 1px solid #16a34a55;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-review {
    display: inline-block;
    background: #f59e0b22;
    color: #d97706;
    border: 1px solid #d9770655;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Summary card */
.summary-card {
    background: linear-gradient(135deg, #6366f110 0%, #8b5cf610 100%);
    border: 1px solid #6366f133;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0 1rem;
}

/* File status list */
.file-ok   { color: #22c55e; font-weight: 500; }
.file-warn { color: #f59e0b; font-weight: 500; }
.file-err  { color: #ef4444; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Config loading ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config.yaml")
CATEGORIES_PATH = Path("categories.yaml")
EXAMPLES_PATH = Path("examples.yaml")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


@st.cache_data(show_spinner=False)
def load_categories() -> dict[str, str]:
    data = yaml.safe_load(CATEGORIES_PATH.read_text(encoding="utf-8")) or {}
    return data.get("categories", {})


def load_examples() -> list[dict]:
    data = yaml.safe_load(EXAMPLES_PATH.read_text(encoding="utf-8")) or {}
    return data.get("examples", []) or []


def save_example(vendor: str, category: str) -> None:
    """Append a new vendor→category example to examples.yaml."""
    examples = load_examples()
    for ex in examples:
        if ex.get("vendor", "").lower() == vendor.lower():
            ex["category"] = category
            break
    else:
        examples.append({"vendor": vendor, "category": category})
    data = {"examples": examples}
    EXAMPLES_PATH.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )
    load_config.clear()


# ── Disk cache for results ─────────────────────────────────────────────────────

def _results_cache_path(output_folder: Path) -> Path:
    # Check .slipsense_cache first, fallback to legacy .shoebox_cache if present
    primary = output_folder / ".slipsense_cache"
    legacy = output_folder / ".shoebox_cache"
    if not (primary / "results.json").exists() and (legacy / "results.json").exists():
        return legacy / "results.json"
    primary.mkdir(parents=True, exist_ok=True)
    return primary / "results.json"


def load_results_cache(output_folder: Path) -> dict[str, dict]:
    path = _results_cache_path(output_folder)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_results_cache(output_folder: Path, cache: dict[str, dict]) -> None:
    path = _results_cache_path(output_folder)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "results": [],
        "file_statuses": {},
        "processing_done": False,
        "selected_row_idx": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────

cfg = load_config()

with st.sidebar:
    st.markdown("## 🗂️ SlipSense")
    st.markdown("*Receipt & invoice sorter — 100% local*")
    default_input = cfg.get("input_folder")
    if not default_input:
        default_input = "synthetic_receipts" if Path("synthetic_receipts").exists() else str(Path.home() / "receipts")

    input_folder = st.text_input(
        "📁 Input folder",
        value=default_input,
        key="input_folder",
        help="Folder containing receipt images and PDFs",
    )

    output_folder = st.text_input(
        "📤 Output folder",
        value=cfg.get("output_folder", "./output"),
        key="output_folder",
        help="Where CSV, renamed files, and cache are saved",
    )

    model_name = st.text_input(
        "🤖 Model",
        value=cfg.get("model_name", "qwen2.5vl:3b"),
        key="model_name",
        help="Ollama model name. Change config.yaml to make it permanent.",
    )

    ollama_url = cfg.get("ollama_url", "http://localhost:11434")

    # Image preparation configuration status
    img_cfg = cfg.get("image", {})
    st.markdown("**Image preparation (§8)**")
    st.caption(f"• Max side: **{img_cfg.get('max_side', 1000)}px** (recommended)")
    st.caption(f"• JPEG quality: **{img_cfg.get('jpeg_quality', 85)}**")
    st.caption(f"• PDF render: **{img_cfg.get('pdf_dpi', 110)} DPI**")

    st.divider()
    st.markdown("**Ollama status**")

    if st.button("🔄 Check Ollama", key="check_ollama"):
        st.session_state["_ollama_checked"] = True

    if st.session_state.get("_ollama_checked", True):
        with st.spinner("Checking…"):
            status = check_ollama(ollama_url)

        if not status["running"]:
            st.error(f"❌ Ollama isn't running.\n\n{status['error']}\n\nStart it, then reload.")
        else:
            found, all_models = is_model_available(ollama_url, model_name)
            if found:
                st.success(f"✅ Ollama running\n✅ Model **{model_name}** ready")
            else:
                st.warning(
                    f"✅ Ollama running\n"
                    f"⚠️ Model **{model_name}** not found.\n\n"
                    f"Run:\n```\nollama pull {model_name}\n```"
                )
                if all_models:
                    st.caption("Available: " + ", ".join(all_models[:5]))

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🗂️ SlipSense — Receipt Sorter")
st.markdown("*All processing happens on your machine. No data leaves localhost.*")
st.divider()


# ── Collect files ──────────────────────────────────────────────────────────────

input_path = Path(input_folder)
output_path = Path(output_folder)

files: list[Path] = []
if input_path.exists() and input_path.is_dir():
    files = sorted(
        [f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    )

if files:
    st.markdown(f"**{len(files)} files** found in `{input_folder}`")
else:
    st.info(
        f"No supported files found in `{input_folder}`. "
        "Add receipt images (.jpg, .png, .webp) or PDFs there."
    )

# ── Process folder button ──────────────────────────────────────────────────────

col_btn, col_clear = st.columns([2, 1])
with col_btn:
    process_clicked = st.button("⚡ Process folder", key="process_btn", disabled=not files)
with col_clear:
    if st.button("🗑️ Clear results", key="clear_btn"):
        st.session_state["results"] = []
        st.session_state["file_statuses"] = {}
        st.session_state["processing_done"] = False
        st.rerun()

if process_clicked and files:
    status = check_ollama(ollama_url)
    if not status["running"]:
        st.error("❌ Ollama isn't running. Start it with `ollama serve`, then reload.")
        st.stop()

    found, _ = is_model_available(ollama_url, model_name)
    if not found:
        st.error(
            f"❌ Model **{model_name}** is not available in Ollama.\n\n"
            f"Run:  `ollama pull {model_name}`"
        )
        st.stop()

    categories_dict = load_categories()
    allowed_cats = list(categories_dict.keys())
    system_prompt = build_system_prompt(CATEGORIES_PATH, EXAMPLES_PATH)

    output_path.mkdir(parents=True, exist_ok=True)
    disk_cache = load_results_cache(output_path)

    progress_bar = st.progress(0, text="Starting…")
    status_area = st.empty()
    log_lines: list[str] = []

    new_results: list[dict] = []
    total_files = len(files)

    existing_by_hash: dict[str, dict] = {
        r["_hash"]: r for r in st.session_state["results"] if "_hash" in r
    }

    for i, fpath in enumerate(files):
        pct = i / total_files
        fname = fpath.name
        progress_bar.progress(pct, text=f"Processing {fname} ({i+1}/{total_files})…")

        try:
            fhash = file_hash(fpath)
        except Exception as exc:
            log_lines.append(f"❌ **{fname}**: could not hash — {exc}")
            status_area.markdown("\n".join(log_lines[-10:]))
            continue

        if fhash in existing_by_hash:
            new_results.append(existing_by_hash[fhash])
            log_lines.append(f"⏭️ **{fname}**: skipped (already processed)")
            status_area.markdown("\n".join(log_lines[-10:]))
            continue

        if fhash in disk_cache:
            cached_row = disk_cache[fhash]
            new_results.append(cached_row)
            log_lines.append(f"💾 **{fname}**: loaded from cache")
            status_area.markdown("\n".join(log_lines[-10:]))
            continue

        # ── Actual processing ─────────────────────────────────────────────────
        t_start = time.time()

        try:
            result = extract_receipt(
                fpath,
                ollama_url=ollama_url,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=cfg.get("temperature", 0),
                config_path=CONFIG_PATH,
            )

            if result.get("unreadable") or result.get("error") == "Could not read file":
                error_row = {
                    "original_filename": fname,
                    "original_filepath": str(fpath),
                    "_hash": fhash,
                    "_retried": False,
                    "_extract_error": "Could not read file",
                    "status": "Needs review",
                    "reasons": ["Could not read file"],
                    "vendor": None,
                    "date": None,
                    "total": None,
                    "subtotal": None,
                    "tax": None,
                    "currency": None,
                    "category": "Uncategorized",
                    "elapsed_sec": round(time.time() - t_start, 1),
                }
                disk_cache[fhash] = error_row
                new_results.append(error_row)
                log_lines.append(f"⚠️ **{fname}**: Could not read file (flagged Needs review)")
                status_area.markdown("\n".join(log_lines[-10:]))
                continue

            if result["data"] is None:
                raise ValueError(result.get("error", "Extraction returned no data"))

            data_dict = result["data"].model_dump()
            data_dict["original_filename"] = fname
            data_dict["original_filepath"] = str(fpath)
            data_dict["_hash"] = fhash
            data_dict["_retried"] = result["retried"]
            data_dict["_extract_error"] = result.get("error")
            data_dict["elapsed_sec"] = round(time.time() - t_start, 1)
            combined_row = data_dict

        except Exception as exc:
            reason = "Could not read file" if "Could not read file" in str(exc) else f"Extraction failed: {exc}"
            log_lines.append(f"❌ **{fname}**: {reason}")
            status_area.markdown("\n".join(log_lines[-10:]))
            error_row = {
                "original_filename": fname,
                "original_filepath": str(fpath),
                "_hash": fhash,
                "_retried": False,
                "_extract_error": str(exc),
                "status": "Needs review",
                "reasons": [reason],
                "vendor": None,
                "date": None,
                "total": None,
                "subtotal": None,
                "tax": None,
                "currency": None,
                "category": "Uncategorized",
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            disk_cache[fhash] = error_row
            new_results.append(error_row)
            continue

        elapsed = round(time.time() - t_start, 1)
        log_lines.append(f"✅ **{fname}**: done in {elapsed}s")
        status_area.markdown("\n".join(log_lines[-10:]))
        new_results.append(combined_row)
        disk_cache[fhash] = combined_row

    # ── Duplicate detection ────────────────────────────────────────────────────
    dupe_indices = find_duplicates(new_results)

    # ── Validate all rows ──────────────────────────────────────────────────────
    categories_dict = load_categories()
    allowed_cats = list(categories_dict.keys())
    validated: list[dict] = []
    for idx, row in enumerate(new_results):
        if row.get("status") in ("OK", "Needs review") and "reasons" in row:
            validated.append(row)
            continue
        v_row = validate_row(
            row,
            allowed_categories=allowed_cats,
            is_duplicate=(idx in dupe_indices),
            retried=row.get("_retried", False),
        )
        validated.append(v_row)

    # §4: sort — Needs review first, then by category, then by date
    def _sort_key(r):
        status_ord = 0 if r.get("status") == "Needs review" else 1
        cat = (r.get("category") or "").lower()
        date_val = r.get("date") or "9999"
        return (status_ord, cat, date_val)

    validated.sort(key=_sort_key)

    st.session_state["results"] = validated
    save_results_cache(output_path, disk_cache)
    st.session_state["processing_done"] = True

    progress_bar.progress(1.0, text="Done!")
    st.success(f"✅ Processed {len(files)} files.")
    st.rerun()

# ── Results table ──────────────────────────────────────────────────────────────

results = st.session_state.get("results", [])

if results:
    needs_review = sum(1 for r in results if r.get("status") == "Needs review")
    ok_count = len(results) - needs_review

    st.markdown(
        f"""<div class="summary-card">
        📊 <strong>{len(results)} files processed</strong> —
        <span style="color:#22c55e">✅ {ok_count} OK</span> &nbsp;|&nbsp;
        <span style="color:#f59e0b">⚠️ {needs_review} need review</span>
        </div>""",
        unsafe_allow_html=True,
    )

    categories_dict = load_categories()
    allowed_cats = list(categories_dict.keys()) + ["Uncategorized"]

    # ── Category filter (§4) ───────────────────────────────────────────────────
    all_cats_in_results = sorted(
        {r.get("category") or "Uncategorized" for r in results}
    )
    filter_options = ["All categories"] + all_cats_in_results
    selected_cat_filter = st.selectbox(
        "🔽 Filter by category",
        options=filter_options,
        key="cat_filter",
    )

    filtered_results = results
    if selected_cat_filter != "All categories":
        filtered_results = [
            r for r in results
            if (r.get("category") or "Uncategorized") == selected_cat_filter
        ]

    # Build display DataFrame with the 9 required columns (§4)
    # Status | Vendor | Sub Total | Tax | Total | Currency | Category | Date | Reason
    display_data = []
    for row in filtered_results:
        date_val = row.get("date")
        display_data.append({
            "Status":     row.get("status", "OK"),
            "Vendor":     row.get("vendor") or "",
            "Sub Total":  row.get("subtotal"),
            "Tax":        row.get("tax"),
            "Total":      row.get("total"),
            "Currency":   row.get("currency") or "",
            "Category":   row.get("category") or "Uncategorized",
            "Date":       date_val if date_val else "NA",
            "Reason":     _reasons_str(row.get("reasons", [])),
            # hidden metadata for preview (not shown as column)
            "_orig_filepath": row.get("original_filepath", ""),
            "_orig_filename": row.get("original_filename", ""),
        })

    df = pd.DataFrame(display_data)
    # Drop hidden metadata columns before passing to data_editor
    visible_df = df.drop(columns=["_orig_filepath", "_orig_filename"])

    # ── Editable table (§4) ────────────────────────────────────────────────────
    st.markdown("### 📋 Extracted Data")
    st.caption("Edit any cell (Status and Reason are read-only). Select a row to preview the receipt.")

    edited_df = st.data_editor(
        visible_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Status": st.column_config.TextColumn("Status", disabled=True, width="small"),
            "Vendor": st.column_config.TextColumn("Vendor", width="medium"),
            "Sub Total": st.column_config.NumberColumn("Sub Total", format="%.2f"),
            "Tax": st.column_config.NumberColumn("Tax", format="%.2f"),
            "Total": st.column_config.NumberColumn("Total", format="%.2f"),
            "Currency": st.column_config.TextColumn("Currency", width="small"),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=allowed_cats,
                width="medium",
            ),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Reason": st.column_config.TextColumn("Reason", disabled=True, width="large"),
        },
        key="results_editor",
        hide_index=False,
    )

    # Apply edits back → recompute status & reason (§4)
    for i, (_, edited_row) in enumerate(edited_df.iterrows()):
        if i < len(filtered_results):
            orig = filtered_results[i]
            # Find this row's index in the full results list
            orig_idx = results.index(orig)

            # Detect edits (anything except Status and Reason)
            changed = False
            for col in ["Vendor", "Sub Total", "Tax", "Total", "Currency", "Category", "Date"]:
                new_val = edited_row.get(col)
                # Map display column names back to internal keys
                internal_key = {
                    "Vendor": "vendor",
                    "Sub Total": "subtotal",
                    "Tax": "tax",
                    "Total": "total",
                    "Currency": "currency",
                    "Category": "category",
                    "Date": "date",
                }[col]
                old_val = orig.get(internal_key)
                # Handle "NA" date display
                if internal_key == "date" and new_val == "NA":
                    new_val = None
                if str(new_val) != str(old_val):
                    results[orig_idx][internal_key] = new_val
                    changed = True

            if changed:
                # Recompute status and reason
                dupe_set = find_duplicates(results)
                v_row = validate_row(
                    results[orig_idx],
                    allowed_categories=list(categories_dict.keys()),
                    is_duplicate=(orig_idx in dupe_set),
                    retried=results[orig_idx].get("_retried", False),
                )
                results[orig_idx].update(v_row)

    # ── Receipt preview beside table (§4) ─────────────────────────────────────
    st.markdown("### 🔍 Receipt Preview")
    max_row = max(0, len(filtered_results) - 1)
    row_idx = st.number_input(
        "Select row number to preview (0-indexed)",
        min_value=0,
        max_value=max_row,
        value=min(st.session_state.get("selected_row_idx", 0), max_row),
        step=1,
        key="preview_row_idx",
    )
    st.session_state["selected_row_idx"] = row_idx

    if 0 <= row_idx < len(filtered_results):
        selected = filtered_results[row_idx]
        col_preview, col_details = st.columns([1, 1])

        with col_preview:
            orig_path = Path(df.at[row_idx, "_orig_filepath"] if row_idx < len(df) else "")
            if orig_path.exists():
                try:
                    if orig_path.suffix.lower() == ".pdf":
                        pages = render_pdf_pages(orig_path, dpi=100, all_pages=False)
                        if pages:
                            _, tmp = pages[0]
                            img = Image.open(tmp)
                            st.image(img, caption=orig_path.name, use_container_width=True)
                            tmp.unlink(missing_ok=True)
                    else:
                        img = Image.open(orig_path)
                        st.image(img, caption=orig_path.name, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not preview: {e}")
            else:
                st.info("No preview available.")

        with col_details:
            st.markdown(f"**File:** `{selected.get('original_filename', '')}`")
            s = selected.get("status", "OK")
            badge = (
                '<span class="badge-ok">✅ OK</span>'
                if s == "OK"
                else '<span class="badge-review">⚠️ Needs review</span>'
            )
            st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)

            reasons = selected.get("reasons", [])
            if reasons:
                if isinstance(reasons, list) and reasons:
                    st.warning("**Issues:**\n" + "\n".join(f"• {r}" for r in reasons))
                elif isinstance(reasons, str) and reasons:
                    st.warning(f"**Issues:** {reasons}")

            vendor = selected.get("vendor") or ""
            category = selected.get("category") or "Uncategorized"
            if vendor:
                if st.button(f"💾 Remember '{vendor}' → {category}", key=f"remember_{row_idx}"):
                    save_example(vendor, category)
                    st.success(f"Saved: {vendor} → {category}")

    # ── Category Summary (§5) ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Category Summary")
    st.caption("Updates live when you edit the table. Currencies are never mixed.")

    summary_df = compute_category_summary(results)
    if not summary_df.empty:
        # Style grand-total rows
        def _style_totals(row):
            if "TOTAL" in str(row.get("Category", "")):
                return ["font-weight: bold; background-color: #1e293b22"] * len(row)
            return [""] * len(row)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sub Total": st.column_config.NumberColumn("Sub Total", format="%.2f"),
                "Tax":       st.column_config.NumberColumn("Tax",       format="%.2f"),
                "Total":     st.column_config.NumberColumn("Total",     format="%.2f"),
            },
        )
    else:
        st.info("No data to summarize yet.")

    # ── Export (§6) ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📤 Export")

    if st.button("📥 Export CSV + Categorized Files", key="export_btn"):
        out_path = Path(output_folder)
        export_rows = [dict(r) for r in results]

        with st.spinner("Exporting…"):
            try:
                csv_path, summary_df_out, summary_csv = run_export(export_rows, out_path)
                st.success(
                    f"✅ Exported:\n"
                    f"- `{csv_path}`\n"
                    f"- `{summary_csv}`\n"
                    f"- Files copied to `{out_path}/<Category>/`"
                )

                st.markdown("#### 📊 Category Summary")
                st.dataframe(summary_df_out, use_container_width=True, hide_index=True)

                with open(csv_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download expenses.csv",
                        data=f,
                        file_name="expenses.csv",
                        mime="text/csv",
                        key="dl_csv",
                    )
                with open(summary_csv, "rb") as f:
                    st.download_button(
                        "⬇️ Download category_summary.csv",
                        data=f,
                        file_name="category_summary.csv",
                        mime="text/csv",
                        key="dl_summary_csv",
                    )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

elif st.session_state.get("processing_done"):
    st.info("No results to show.")
else:
    st.info("Select an input folder and click **Process folder** to start.")
