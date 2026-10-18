# SlipSense

**Privacy-first local AI receipt & invoice sorter for freelancers and small businesses.**  
*All processing happens on your own machine. Your financial documents never leave localhost.*

---

## What It Does

SlipSense drops a folder of receipt photos and PDFs into a locally-running vision AI model (via Ollama), extracts structured financial fields (vendor, date, subtotal, tax, total, currency, category), validates the math with deterministic rules, flags questionable rows for human review, and exports an accountant-ready `expenses.csv` plus neatly categorized copies of your originals.

**Privacy is the core feature.** No cloud APIs, no external telemetry. The AI runs entirely on your CPU/GPU, and works with Wi-Fi switched off.

---

## Features

- 📸 **Universal Document Ingestion**: Accepts JPG, PNG, WebP, and multi-format PDF receipts and invoices.
- 🖼️ **In-Memory Image Preparation (`image_prep.py`)**: Automatically fixes phone EXIF rotation, rasterizes PDF pages with PyMuPDF at 110 DPI, converts to RGB, downscales to 1000px max side (preserving aspect ratio, never upscaling), and encodes in-memory JPEG (quality 85). Never writes temporary files to disk.
- 🤖 **Local Vision AI**: Vision model served locally by Ollama (`qwen2.5vl:3b`). Swap models with a single line in `config.yaml`.
- 📐 **Standardized 9-Column Output**: `Status | Vendor | Sub Total | Tax | Total | Currency | Category | Date | Reason`.
- 🧮 **Deterministic Verification (`validation.py`)**: Rule-based arithmetic validation (1% tolerance check between subtotal + tax and total), tax summation (GST, CGST, SGST, IGST, VAT, cess), duplicate detection, and future-date prevention. Code rules decide trust, not model confidence scores.
- 📝 **Human-in-the-Loop Review**: Streamlit UI sorts `Needs review` rows to the top. Clicking any row opens an inspector card showing the receipt image side-by-side with editable fields.
- 📊 **Multi-Currency Category Summary**: Generates `category_summary.csv` aggregating expenses by category and currency. Currencies are never mixed together.
- 📤 **Non-Destructive Export (`exporter.py`)**: Copies files to `output/<Category>/YYYY-MM-DD_Vendor.ext` with collision-safe suffixes. Original files are never modified, overwritten, or deleted.
- ⚡ **Disk Caching**: Already-processed receipts are cached by SHA-256 hash to eliminate redundant computation.
- 🧪 **Full Test Suite & Benchmark**: 84 pytest test suite covering all modules, plus `benchmark_sizes.py` to measure latency and accuracy across image resolutions.

---

## Quick Start

### 1. Install Ollama

- **macOS / Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download installer from [https://ollama.ai](https://ollama.ai)

### 2. Pull the default vision model

```bash
ollama pull qwen2.5vl:3b
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the SlipSense app

```bash
streamlit run app.py
```

The application opens automatically at `http://localhost:8501`.

---

## Using SlipSense

1. **Place receipts** in a folder (e.g. `synthetic_receipts` or `~/receipts/`).
2. **Set the input folder** in the Streamlit sidebar.
3. Click **⚡ Process folder** to run in-memory preparation and local extraction.
4. **Inspect flagged rows** at the top of the table; edit fields side-by-side with the receipt image if needed.
5. Click **📥 Export** to save `expenses.csv`, `category_summary.csv`, and organized folders under `./output/`.

---

## Testing & Benchmarks

### Run Unit & Integration Tests

```bash
pytest tests/ -v
```

84 tests cover validation rules, date parsing, in-memory image preparation, EXIF transposition, filename sanitization, CSV export, and Ollama client mocking.

### Run Resolution Benchmark

To measure real extraction latency and accuracy across image resolutions (`max_side` = 1400, 1000, 800):

```bash
python benchmark_sizes.py --receipts synthetic_receipts --ground-truth synthetic_receipts/ground_truth.csv
```

Measured results are saved to `benchmark_report.md`.

### Run Accuracy Evaluation

```bash
python evaluate.py --receipts synthetic_receipts --ground-truth synthetic_receipts/ground_truth.csv
```

Evaluates field-level accuracy (vendor, date, subtotal, tax, total, currency, category) against ground truth.

---

## Project Structure

```
SlipSense/
├── app.py                      # Streamlit interactive UI & inspector
├── image_prep.py               # In-memory image preprocessing & PDF rendering
├── extraction.py               # Ollama orchestration & Pydantic parsing
├── validation.py               # Deterministic business rules & status logic
├── exporter.py                 # Safe export, renaming, and category summaries
├── ollama_client.py            # Local Ollama HTTP API client
├── benchmark_sizes.py          # Multi-resolution latency & accuracy benchmark
├── evaluate.py                 # Ground-truth accuracy evaluation script
├── make_synthetic_receipts.py  # Synthetic receipt generator for testing
├── config.yaml                 # Central configuration (model, URL, image settings)
├── categories.yaml             # Expense category definitions
├── examples.yaml               # Few-shot vendor → category hints
├── benchmark_report.md         # Measured resolution benchmark report
├── requirements.txt            # Python dependencies
├── submissions/                # DEV.to challenge submission & screenshots
│   ├── submission.md
│   └── screenshots/
├── tests/
│   ├── test_image_prep.py
│   ├── test_validation.py
│   ├── test_exporter.py
│   └── test_ollama_client.py
├── LICENSE                     # MIT License
└── README.md
```

---

## Configuration Reference

Edit `config.yaml` to configure runtime options:

```yaml
ollama_url: "http://localhost:11434"
model_name: "qwen2.5vl:3b"

image:
  max_side: 1000       # Downscale longest side to at most 1000px
  jpeg_quality: 85     # In-memory JPEG encoding quality
  pdf_dpi: 110         # PyMuPDF rasterization DPI for PDFs

temperature: 0         # Deterministic extraction
output_folder: "./output"
```

---

## Privacy Guarantee

- **Zero Cloud Calls**: SlipSense connects only to `localhost`. Verify by disconnecting Wi-Fi; the entire app continues to operate.
- **Original Documents are Sacred**: Files in your input folder are never altered, moved, or deleted.
- **No Telemetry**: No tracking or data collection.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 SlipSense Contributors.
