"""Fetch one or both QLoRA models (adapter + GGUF) from Hugging Face Hub.

One HF repo holds both fine-tunes in sibling subfolders. Files land in the
local layout the rest of the project expects (no `qwen/` / `llama32/` prefix
locally), so Streamlit's pipeline router and the Ollama Modelfiles resolve
their paths with no extra config.

Mapping:
  HF repo                          Local destination
  qwen/qwen-medqa-adapter/      -> models/qwen-medqa-adapter/
  qwen/qwen-medqa-gguf/         -> models/qwen-medqa-gguf/
  llama32/llama32-medqa-adapter -> models/llama32-medqa-adapter/
  llama32/llama32-medqa-gguf/   -> models/llama32-medqa-gguf/

Usage (from project root):
  python scripts/download_model_from_hf.py                       # both, full (adapter + GGUF)
  python scripts/download_model_from_hf.py --target qwen
  python scripts/download_model_from_hf.py --target llama32 --only-gguf
  python scripts/download_model_from_hf.py --target both --only-adapter

Prereq: pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Davis426/COMP8420-Healthcare-LLM-Assistant"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Each target maps to the HF subfolder prefix + the two top-level dirs inside it.
TARGETS: dict[str, dict] = {
    "qwen": {
        "remote_dir":  "qwen",
        "adapter_dir": "qwen-medqa-adapter",
        "gguf_dir":    "qwen-medqa-gguf",
        "ollama_tag":  "medqa-qwen",
    },
    "llama32": {
        "remote_dir":  "llama32",
        "adapter_dir": "llama32-medqa-adapter",
        "gguf_dir":    "llama32-medqa-gguf",
        "ollama_tag":  "medqa-llama32",
    },
}


def _patterns(target_key: str, only: str | None) -> list[str]:
    cfg = TARGETS[target_key]
    rd = cfg["remote_dir"]
    if only == "gguf":
        return [f"{rd}/{cfg['gguf_dir']}/*"]
    if only == "adapter":
        return [f"{rd}/{cfg['adapter_dir']}/*"]
    return [f"{rd}/{cfg['adapter_dir']}/*", f"{rd}/{cfg['gguf_dir']}/*"]


def _flatten_remote_layout(target_key: str) -> None:
    """snapshot_download preserves the HF folder tree (`qwen/qwen-medqa-adapter/...`).
    Move each subfolder up one level so the local layout matches what
    Streamlit + Ollama expect (`qwen-medqa-adapter/...` at MODELS_DIR root)."""
    cfg = TARGETS[target_key]
    remote_root = MODELS_DIR / cfg["remote_dir"]
    if not remote_root.exists():
        return
    for sub in (cfg["adapter_dir"], cfg["gguf_dir"]):
        src = remote_root / sub
        dst = MODELS_DIR / sub
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    # Drop the now-empty `qwen/` (or `llama32/`) wrapper.
    if remote_root.exists() and not any(remote_root.iterdir()):
        remote_root.rmdir()


def fetch(target_key: str, only: str | None) -> None:
    cfg = TARGETS[target_key]
    patterns = _patterns(target_key, only)
    print(f"\n=== {target_key} ===")
    print(f"Patterns: {patterns}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=patterns,
        local_dir=str(MODELS_DIR),
    )
    _flatten_remote_layout(target_key)
    if only != "adapter":
        print(f"Next: cd models/{cfg['gguf_dir']} && ollama create {cfg['ollama_tag']} -f Modelfile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        choices=["qwen", "llama32", "both"],
        default="both",
        help="Which fine-tune to download (default: both)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-gguf", action="store_true",
                       help="Download only the GGUF (for Ollama)")
    group.add_argument("--only-adapter", action="store_true",
                       help="Download only the LoRA adapter (for transformers+peft)")
    args = parser.parse_args()

    only = "gguf" if args.only_gguf else ("adapter" if args.only_adapter else None)

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading from https://huggingface.co/{REPO_ID}")
    print(f"Target dir: {MODELS_DIR}")

    targets = ["qwen", "llama32"] if args.target == "both" else [args.target]
    for t in targets:
        fetch(t, only)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
