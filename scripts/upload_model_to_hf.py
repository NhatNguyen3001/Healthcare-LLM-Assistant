"""Upload one or both QLoRA models (adapter + Q4_K_M GGUF) to Hugging Face Hub.

One HF repo holds both fine-tunes in sibling subfolders:
  Davis426/COMP8420-Healthcare-LLM-Assistant
    ├── qwen/qwen-medqa-adapter/
    ├── qwen/qwen-medqa-gguf/
    ├── llama32/llama32-medqa-adapter/
    └── llama32/llama32-medqa-gguf/

The combined model card lives at the repo root as README.md (sourced from
models/MODEL_CARD.md).

Total upload size per model: ~1.0 GB (mostly the GGUF). Both: ~2 GB.

Usage (from project root):
  python scripts/upload_model_to_hf.py                 # uploads both
  python scripts/upload_model_to_hf.py --target qwen
  python scripts/upload_model_to_hf.py --target llama32
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Davis426/COMP8420-Healthcare-LLM-Assistant"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Each target maps to (local-subfolder-name, repo-subfolder-name).
# Local layout is unchanged from before — the subfolder lives in the repo
# layout only, written by `path_in_repo`. Locally `models/qwen-medqa-adapter`
# stays at top level so the Streamlit / Ollama paths keep resolving.
TARGETS: dict[str, dict] = {
    "qwen": {
        "adapter_dir": MODELS_DIR / "qwen-medqa-adapter",
        "gguf_path":   MODELS_DIR / "qwen-medqa-gguf" / "model.Q4_K_M.gguf",
        "remote_dir":  "qwen",
        "patterns":    ["qwen-medqa-adapter/*", "qwen-medqa-gguf/*"],
        "ignore":      [
            "qwen-medqa-gguf/model.safetensors",
            "qwen-medqa-gguf/model.safetensors.*",
        ],
        "label": "QLoRA Qwen2.5-1.5B",
    },
    "llama32": {
        "adapter_dir": MODELS_DIR / "llama32-medqa-adapter",
        "gguf_path":   MODELS_DIR / "llama32-medqa-gguf" / "model.Q4_K_M.gguf",
        "remote_dir":  "llama32",
        "patterns":    ["llama32-medqa-adapter/*", "llama32-medqa-gguf/*"],
        "ignore":      [
            "llama32-medqa-gguf/model.safetensors",
            "llama32-medqa-gguf/model.safetensors.*",
        ],
        "label": "QLoRA Llama-3.2-1B",
    },
}


def upload_one(api: HfApi, target_key: str) -> None:
    cfg = TARGETS[target_key]
    if not cfg["adapter_dir"].exists():
        raise SystemExit(f"adapter folder missing: {cfg['adapter_dir']}")
    if not cfg["gguf_path"].exists():
        raise SystemExit(f"GGUF missing: {cfg['gguf_path']}")

    print(f"\n=== {cfg['label']} -> {REPO_ID}/{cfg['remote_dir']}/ ===")
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=cfg["remote_dir"],
        # Top-level files in each subfolder only. Single * does NOT recurse,
        # so checkpoint-* subfolders are naturally excluded.
        allow_patterns=cfg["patterns"],
        # Explicit skip for the merged 2-3 GB safetensors. Redundant once the
        # GGUF + adapter are both up.
        ignore_patterns=cfg["ignore"],
        commit_message=f"Upload {cfg['label']} adapter + Q4_K_M GGUF",
    )


def upload_model_card(api: HfApi) -> None:
    # Combined model card lives at the repo root as README.md. Kept as a
    # separate upload_file call so the source filename can stay MODEL_CARD.md
    # locally (avoids colliding with the other README.md files in the repo).
    model_card = MODELS_DIR / "MODEL_CARD.md"
    if not model_card.exists():
        print(f"WARN: {model_card} not found; HF repo will have no model card.")
        return
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update combined model card",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        choices=["qwen", "llama32", "both"],
        default="both",
        help="Which fine-tune to upload (default: both)",
    )
    parser.add_argument(
        "--skip-card",
        action="store_true",
        help="Skip uploading models/MODEL_CARD.md as the repo README",
    )
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        raise SystemExit(f"models/ not found at {MODELS_DIR}")

    api = HfApi()

    print(f"Source: {MODELS_DIR}")
    print(f"Target: https://huggingface.co/{REPO_ID}")
    print("Each GGUF is ~0.8-1.0 GB; an upload can take 10-30 min per model")
    print("on a typical home connection. Progress bars below come from hf_hub.")

    if args.target == "both":
        upload_one(api, "qwen")
        upload_one(api, "llama32")
    else:
        upload_one(api, args.target)

    if not args.skip_card:
        upload_model_card(api)

    print()
    print(f"Done. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
