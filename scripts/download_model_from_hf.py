"""Fetch the QLoRA adapter + Q4_K_M GGUF from Hugging Face Hub.

For users cloning the GitHub repo who want to run the QLoRA path without
retraining. Mirrors the local layout (drops files into models/qwen-medqa-adapter/
and models/qwen-medqa-gguf/), so src/app.py and the Ollama Modelfile resolve
their paths with no extra config.


Usage (from project root):
  python scripts/download_model_from_hf.py             # both adapter + GGUF (~1.0 GB)
  python scripts/download_model_from_hf.py --only-gguf # GGUF only, for Ollama (~941 MB)
  python scripts/download_model_from_hf.py --only-adapter  # adapter only (~82 MB)

Prereq: pip install huggingface_hub
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Davis426/COMP8420-Healthcare-LLM-Assistant"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-gguf", action="store_true",
                       help="Download only the GGUF (for Ollama)")
    group.add_argument("--only-adapter", action="store_true",
                       help="Download only the LoRA adapter (for transformers+peft)")
    args = parser.parse_args()

    if args.only_gguf:
        patterns = ["qwen-medqa-gguf/*"]
    elif args.only_adapter:
        patterns = ["qwen-medqa-adapter/*"]
    else:
        patterns = ["qwen-medqa-adapter/*", "qwen-medqa-gguf/*"]

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading from https://huggingface.co/{REPO_ID}")
    print(f"Target: {MODELS_DIR}")
    print(f"Patterns: {patterns}")
    print()

    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=patterns,
        local_dir=str(MODELS_DIR),
    )

    print()
    print(f"Done. Files in {local_path}")
    if not args.only_adapter:
        print()
        print("Next: register the GGUF with Ollama:")
        print("  cd models/qwen-medqa-gguf")
        print("  ollama create medqa-qwen -f Modelfile")


if __name__ == "__main__":
    main()
