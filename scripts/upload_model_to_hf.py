"""One-shot upload of the QLoRA adapter + Q4_K_M GGUF to Hugging Face Hub.


Total upload size: ~1.0 GB (mostly the 941 MB GGUF).

Usage (from project root):
  python scripts/upload_model_to_hf.py

"""
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Davis426/COMP8420-Healthcare-LLM-Assistant"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    if not MODELS_DIR.exists():
        raise SystemExit(f"models/ not found at {MODELS_DIR}")

    adapter_dir = MODELS_DIR / "qwen-medqa-adapter"
    gguf_path = MODELS_DIR / "qwen-medqa-gguf" / "model.Q4_K_M.gguf"
    if not adapter_dir.exists():
        raise SystemExit(f"adapter folder missing: {adapter_dir}")
    if not gguf_path.exists():
        raise SystemExit(f"GGUF missing: {gguf_path}")

    api = HfApi()

    print(f"Source: {MODELS_DIR}")
    print(f"Target: https://huggingface.co/{REPO_ID}")
    print()
    print("Starting upload. The GGUF is ~941 MB, so this can take 10-30 minutes")
    print("on a typical home connection. Progress bars below come from hf_hub.")
    print()

    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        # Top-level files in each subfolder. Single * does NOT recurse, which
        # is what we want: training checkpoints in qwen-medqa-adapter/checkpoint-*
        # are naturally excluded because they are not at the top level.
        allow_patterns=[
            "qwen-medqa-adapter/*",
            "qwen-medqa-gguf/*",
        ],
        # Explicit skip for the merged 2.9 GB safetensors. Redundant once the
        # GGUF + adapter are both up.
        ignore_patterns=[
            "qwen-medqa-gguf/model.safetensors",
            "qwen-medqa-gguf/model.safetensors.*",
        ],
        commit_message="Upload QLoRA adapter + Q4_K_M GGUF for Ollama deploy",
    )

    # Push the HF model card as README.md at the repo root. Kept as a
    # separate upload_file call so the source filename can stay MODEL_CARD.md
    # locally (avoids colliding with the other README.md files in the repo)
    # while landing at the canonical README.md path on the HF side.
    model_card = MODELS_DIR / "MODEL_CARD.md"
    if model_card.exists():
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
    else:
        print(f"WARN: {model_card} not found; HF repo will have no model card.")

    print()
    print(f"Done. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
