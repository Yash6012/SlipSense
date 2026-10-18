"""
ollama_client.py
────────────────
Thin wrapper around the Ollama local HTTP API.

Responsibilities:
  - Check if Ollama is running (GET /api/tags)
  - Check if the configured model is available
  - Send a single image (base64) to /api/chat with a JSON-schema format field
  - Return the raw JSON string from the model response

All network calls go to localhost only. No cloud APIs are used.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Timeouts (seconds) ────────────────────────────────────────────────────────
CONNECT_TIMEOUT = 5     # Time to establish the TCP connection
READ_TIMEOUT = 300      # Max time to wait for the model to finish (300s for cold-start)


# ── Health Checks ─────────────────────────────────────────────────────────────

def check_ollama(ollama_url: str) -> dict:
    """
    Call GET /api/tags to see if Ollama is reachable.

    Returns a dict with:
      {
        "running": bool,
        "models": list[str],   # available model names (empty if not running)
        "error": str | None    # human-readable error if not running
      }
    """
    try:
        resp = requests.get(
            f"{ollama_url}/api/tags",
            timeout=CONNECT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"running": True, "models": models, "error": None}
    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "models": [],
            "error": "Cannot reach Ollama. Is it running? Start with: ollama serve",
        }
    except requests.exceptions.Timeout:
        return {
            "running": False,
            "models": [],
            "error": "Ollama health check timed out.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "models": [], "error": str(exc)}


def is_model_available(ollama_url: str, model_name: str) -> tuple[bool, list[str]]:
    """
    Return (available: bool, all_model_names: list[str]).
    The comparison strips the ':latest' tag that Ollama sometimes appends.
    """
    result = check_ollama(ollama_url)
    if not result["running"]:
        return False, []
    available = result["models"]
    # Normalise: strip ':latest' for comparison
    def _norm(n: str) -> str:
        return n.replace(":latest", "").strip()

    found = any(_norm(m) == _norm(model_name) for m in available)
    return found, available


# ── Image encoding ────────────────────────────────────────────────────────────

def encode_image_b64(image_path: Path) -> str:
    """Read a file and return its base64-encoded string (no data URI prefix)."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


# ── Main extraction call ───────────────────────────────────────────────────────

def extract_from_image(
    ollama_url: str,
    model_name: str,
    image_data: Path | str,
    system_prompt: str,
    user_message: str,
    json_schema: dict,
    temperature: float = 0,
) -> tuple[Optional[str], Optional[str]]:
    """
    Send `image_data` (either a base64 string, PreparedImage, or a file Path)
    to Ollama /api/chat with the given prompts and schema.

    If a file path is provided, it is automatically prepared in memory via
    image_prep.prepare_image so that raw un-processed images are never sent.

    Returns (json_text: str | None, error: str | None).
    """
    # If image_data is a file path, prepare it in memory first
    if isinstance(image_data, Path) or (isinstance(image_data, str) and len(image_data) < 1000 and Path(image_data).is_file()):
        try:
            from image_prep import prepare_image
            b64 = str(prepare_image(image_data))
        except Exception as exc:
            return None, f"Could not read file: {exc}"
    else:
        b64 = str(image_data)

    payload = {
        "model": model_name,
        "stream": False,
        "keep_alive": "10m",      # §8: keep model warm for 10 minutes
        "options": {
            "temperature": temperature,
            "num_ctx": 2048,      # §8: low-memory context window
            "num_predict": 200,   # §8: max tokens to generate
        },
        "format": json_schema,   # Ollama structured-output field
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
                "images": [b64],  # Ollama vision: list of base64 strings
            },
        ],
    }

    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["message"]["content"]
        logger.debug("Ollama raw response: %s", raw_text[:500])
        return raw_text, None
    except requests.exceptions.Timeout:
        return None, "Ollama request timed out. Try a smaller model or increase READ_TIMEOUT."
    except requests.exceptions.ConnectionError as exc:
        return None, f"Connection error: {exc}"
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:  # noqa: BLE001
            pass
        return None, f"HTTP {exc.response.status_code}: {body}"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick command-line test:
        python ollama_client.py <image_path> [--config config.yaml]
    Extracts one image and prints the raw JSON response.
    """
    import argparse
    import sys
    import yaml

    parser = argparse.ArgumentParser(description="Test Ollama extraction on a single image.")
    parser.add_argument("image", type=Path, help="Path to a receipt image or PDF-rendered PNG")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    url = cfg["ollama_url"]
    model = cfg["model_name"]

    print(f"Ollama URL : {url}")
    print(f"Model      : {model}")

    # Health check
    status = check_ollama(url)
    if not status["running"]:
        print(f"\n❌ Ollama is not running: {status['error']}")
        sys.exit(1)

    found, all_models = is_model_available(url, model)
    if not found:
        print(f"\n❌ Model '{model}' not found. Run: ollama pull {model}")
        print(f"   Available: {all_models}")
        sys.exit(1)

    print(f"✅ Ollama running. Model '{model}' found.\n")

    # Minimal schema for the CLI test
    from extraction import RECEIPT_JSON_SCHEMA, build_system_prompt  # noqa: E402

    sys_prompt = build_system_prompt()
    user_msg = "Extract all receipt/invoice fields from this image."

    if not args.image.exists():
        print(f"❌ File not found: {args.image}")
        sys.exit(1)

    print(f"Processing: {args.image.name}")
    raw, err = extract_from_image(url, model, args.image, sys_prompt, user_msg, RECEIPT_JSON_SCHEMA)

    if err:
        print(f"\n❌ Error: {err}")
        sys.exit(1)

    # Pretty-print
    try:
        parsed = json.loads(raw)
        print("\n── Extracted JSON ──────────────────────────────────────")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("\n⚠️  Model returned non-JSON output:")
        print(raw)
