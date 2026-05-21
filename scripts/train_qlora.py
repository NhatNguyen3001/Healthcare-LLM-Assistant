"""
QLoRA fine-tune of Qwen2.5-1.5B-Instruct on the medical Q/A mix.

Reads data/qlora_training/{train,val}.jsonl, applies the Qwen2.5 chat template,
runs SFT with LoRA adapters on top of a 4-bit-quantized base (QLoRA), and saves
the adapter to models/qwen-medqa-adapter/.

Expected wall time on RTX 4060 (8 GB): ~30-50 min for 3 epochs over ~27.5k pairs.
Peak VRAM during training: ~5-6 GB.

Run:
    python scripts/train_qlora.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# IMPORTANT: Unsloth must be imported BEFORE transformers / peft / trl so it
# can monkey-patch their attention layers and optimizer paths. Don't reorder.
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"   / "qlora_training"
OUTPUT_DIR = REPO_ROOT / "models" / "qwen-medqa-adapter"

# Pre-quantized 4-bit Qwen2.5 hosted by Unsloth. Saves ~30s of on-the-fly
# quantization vs loading Qwen/Qwen2.5-1.5B-Instruct and quantizing locally.
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

# Covers ~99% of MedQuAD answers (we capped answers to 1500 chars in prep,
# which is ~375 tokens; plus question + chat-template overhead stays well under).
MAX_SEQ_LEN = 1024


def require_cuda() -> None:
    """QLoRA needs a CUDA GPU. Bail early with a clear message if absent."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required. Unsloth does not support CPU-only training. "
            "Check `nvidia-smi` -- if blank, NVIDIA driver / CUDA toolkit is missing."
        )
    print(f"[train] CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"[train] bf16 ok     : {is_bfloat16_supported()}")
    print(f"[train] torch       : {torch.__version__}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int,   default=2)
    ap.add_argument("--grad-accum", type=int,   default=4)
    ap.add_argument("--lora-r",     type=int,   default=16)
    ap.add_argument("--lora-alpha", type=int,   default=32)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--output-dir", type=Path,  default=OUTPUT_DIR)
    args = ap.parse_args()

    require_cuda()

    # ---- Load 4-bit-quantized base + tokenizer ----------------------------------
    print(f"[train] loading base : {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = MAX_SEQ_LEN,
        dtype           = None,        # Unsloth picks bf16 on Ampere+, fp16 otherwise
        load_in_4bit    = True,        # the "Q" in QLoRA
    )

    # ---- Pin the Qwen2.5 chat template ------------------------------------------
    # This MUST match what Ollama serves with at inference time. The stock Qwen2.5
    # Modelfile that Unsloth's `save_pretrained_gguf` writes uses this same template,
    # so train-time and deploy-time stay aligned. Don't hand-roll the template string.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ---- Attach LoRA adapters to the 4-bit base ---------------------------------
    # All 7 projection layers (attention + MLP). For 1.5B models, adapting the MLP
    # layers in addition to attention gives a noticeable quality lift over the
    # attention-only configuration that the original QLoRA paper used on 65B Llama.
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = 0,
        bias           = "none",
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",   # Unsloth's more VRAM-efficient variant
        random_state   = args.seed,
        use_rslora     = False,
        loftq_config   = None,
    )

    # ---- Load the prepared messages-format dataset ------------------------------
    print(f"[train] loading data : {DATA_DIR}")
    if not (DATA_DIR / "train.jsonl").exists():
        raise SystemExit(
            f"missing {DATA_DIR/'train.jsonl'} -- run scripts/prepare_qlora_dataset.py first"
        )

    ds = load_dataset(
        "json",
        data_files = {
            "train": str(DATA_DIR / "train.jsonl"),
            "val":   str(DATA_DIR / "val.jsonl"),
        },
    )
    print(f"[train]   train rows : {len(ds['train']):>6}")
    print(f"[train]   val rows   : {len(ds['val']):>6}")

    # Apply the Qwen2.5 chat template to each row, producing a single "text" string.
    # add_generation_prompt=False because we want the full Q+A turn in the training
    # text -- the model needs to see both sides to learn to produce the assistant turn.
    def format_row(row):
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize               = False,
            add_generation_prompt  = False,
        )
        return {"text": text}

    ds = ds.map(format_row, remove_columns=["messages", "source"])

    # ---- SFT config -------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir                   = str(args.output_dir),
        per_device_train_batch_size  = args.batch_size,
        gradient_accumulation_steps  = args.grad_accum,    # effective batch = batch * accum
        num_train_epochs             = args.epochs,
        learning_rate                = args.lr,
        warmup_ratio                 = 0.03,
        lr_scheduler_type            = "cosine",
        optim                        = "adamw_8bit",       # quantized optimizer, saves ~1 GB VRAM
        bf16                         = is_bfloat16_supported(),
        fp16                         = not is_bfloat16_supported(),
        logging_steps                = 25,
        save_strategy                = "epoch",
        eval_strategy                = "steps",
        eval_steps                   = 200,
        seed                         = args.seed,
        report_to                    = "none",             # no wandb / tensorboard
        dataset_text_field           = "text",
        max_seq_length               = MAX_SEQ_LEN,
        packing                      = False,              # keep examples separate
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = ds["train"],
        eval_dataset  = ds["val"],
        args          = sft_config,
    )

    # Loss-mask user tokens so gradients only flow on assistant tokens. Without this,
    # the model wastes capacity learning to predict the user's question -- a measurable
    # quality hit on QA fine-tunes. Token markers below are Qwen2.5-specific.
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # ---- Train ------------------------------------------------------------------
    print("[train] starting training (expect ~30-50 min on RTX 4060)")
    train_result = trainer.train()
    runtime_s = train_result.metrics["train_runtime"]
    print(f"[train] done in {runtime_s:.1f}s ({runtime_s/60:.1f} min)")

    # ---- Save the final adapter -------------------------------------------------
    # Note: SFTTrainer's save_strategy="epoch" has already saved checkpoints during
    # training. This final save guarantees the latest weights land at output_dir/
    # with the names export_to_ollama.py expects.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] saving adapter to {args.output_dir}")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Snapshot the run config + metrics for the training report.
    metrics_path = args.output_dir / "training_metrics.json"
    metrics = {
        "base_model":             BASE_MODEL,
        "chat_template":          "qwen-2.5",
        "epochs":                 args.epochs,
        "learning_rate":          args.lr,
        "per_device_batch_size":  args.batch_size,
        "gradient_accumulation":  args.grad_accum,
        "effective_batch_size":   args.batch_size * args.grad_accum,
        "lora_r":                 args.lora_r,
        "lora_alpha":             args.lora_alpha,
        "max_seq_length":         MAX_SEQ_LEN,
        "seed":                   args.seed,
        "n_train":                len(ds["train"]),
        "n_val":                  len(ds["val"]),
        "train_runtime_seconds":  runtime_s,
        "final_train_loss":       train_result.metrics.get("train_loss"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] wrote {metrics_path.name}")

    print()


if __name__ == "__main__":
    main()
