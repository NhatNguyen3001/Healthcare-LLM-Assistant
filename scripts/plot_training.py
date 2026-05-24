"""
Produces:
  - results/qlora_loss_curve.png  -- train + eval loss vs step
  - results/qlora_source_mix.png  -- pairs per source in train.jsonl

The loss-curve plot needs `trainer_state.json` from a completed (or in-progress)
training run; if absent, it's skipped silently so the script is safe to re-run
both before and after training.

Run:
    python scripts/plot_training.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
QWEN_ADAPTER_DIR = REPO_ROOT / "models" / "qwen-medqa-adapter"
LLAMA_ADAPTER_DIR = REPO_ROOT / "models" / "llama32-medqa-adapter"
TRAIN_JSONL = REPO_ROOT / "data" / "qlora_training" / "train.jsonl"

WARM = "#c47a4d"
COOL = "#3b6e8f"
LLAMA_WARM = "#8b5e3c"
LLAMA_COOL = "#2d5470"


def find_trainer_state(adapter_dir: Path) -> Path | None:
    """Prefer the root trainer_state.json; fall back to the latest checkpoint."""
    direct = adapter_dir / "trainer_state.json"
    if direct.exists():
        return direct
    candidates = list(adapter_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.parent.name.split("-")[1]))


def _load_history(adapter_dir: Path):
    state_path = find_trainer_state(adapter_dir)
    if state_path is None:
        return None, None
    history = json.loads(state_path.read_text(encoding="utf-8")).get("log_history", [])
    train = [(e["step"], e["loss"]) for e in history if "loss" in e and "eval_loss" not in e]
    eval_ = [(e["step"], e["eval_loss"]) for e in history if "eval_loss" in e]
    return train, eval_


def plot_loss_curve() -> None:
    qwen_train, qwen_eval = _load_history(QWEN_ADAPTER_DIR)
    llama_train, llama_eval = _load_history(LLAMA_ADAPTER_DIR)

    if qwen_train is None and llama_train is None:
        print("[plot] loss curve skipped (no trainer_state.json for either model)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Qwen subplot
    ax = axes[0]
    if qwen_train:
        xs, ys = zip(*qwen_train)
        ax.plot(xs, ys, label="train loss", color=WARM, linewidth=1.8)
    if qwen_eval:
        xs, ys = zip(*qwen_eval)
        ax.plot(xs, ys, label="eval loss", color=COOL, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Qwen2.5-1.5B-Instruct")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Llama subplot
    ax = axes[1]
    if llama_train:
        xs, ys = zip(*llama_train)
        ax.plot(xs, ys, label="train loss", color=LLAMA_WARM, linewidth=1.8)
    if llama_eval:
        xs, ys = zip(*llama_eval)
        ax.plot(xs, ys, label="eval loss", color=LLAMA_COOL, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("step")
    ax.set_title("Llama-3.2-1B-Instruct")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("QLoRA training loss curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "qlora_loss_curve.png", dpi=150)
    plt.close(fig)
    print("[plot] loss curve done (both models)")


def plot_source_mix() -> None:
    if not TRAIN_JSONL.exists():
        print("[plot] source mix skipped (train.jsonl not found)")
        return

    counts: Counter = Counter()
    with TRAIN_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            counts[json.loads(line)["source"]] += 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    sources, values = zip(*items)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sources, values, color=WARM)
    ax.invert_yaxis()
    ax.set_xlabel("training pairs")
    ax.set_title(f"QLoRA training dataset mix (total: {sum(values):,})")
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "qlora_source_mix.png", dpi=150)
    plt.close(fig)
    print("[plot] source mix done")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_source_mix()
    plot_loss_curve()


if __name__ == "__main__":
    main()
