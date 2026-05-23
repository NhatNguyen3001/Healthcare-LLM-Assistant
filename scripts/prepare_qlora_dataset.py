"""
build the QLoRA training dataset.

Reads the three cleaned Q/A-shaped knowledge bases (MedQuAD + BioASQ + DrugBank),
templates DrugBank entries into Q/A pairs, normalizes text, deterministically
shuffles, and writes 90/5/5 train/val/test splits as JSONL in messages format.

Output schema (one JSON object per line):
  {
    "messages": [
      {"role": "user",      "content": "<question>"},
      {"role": "assistant", "content": "<answer>"}
    ],
    "source": "medquad" | "bioasq" | "drugbank-description"
              | "drugbank-indication" | "drugbank-side_effects"
              | "drugbank-mechanism_of_action"
  }

No chat template is applied here. We keep messages-format on disk so each
train_qlora_{qwen,llama32}.py script can call tokenizer.apply_chat_template
at train time with its own base-specific template. This is what lets the two
QLoRA fine-tunes share one prepared dataset.

Run:
    python scripts/prepare_qlora_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- resolved relative to repo root (this file lives in scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

MEDQUAD_PATH  = REPO_ROOT / "knowledge_bases" / "medquad"  / "medquad_cleaned.jsonl"
BIOASQ_PATH   = REPO_ROOT / "knowledge_bases" / "bioasq"   / "bioasq_cleaned.jsonl"
DRUGBANK_PATH = REPO_ROOT / "knowledge_bases" / "drugbank" / "drugbank_cleaned.jsonl"

OUT_DIR = REPO_ROOT / "data" / "qlora_training"

# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------
# Drop answers shorter than this -- too thin to teach the model anything useful.
MIN_ANSWER_CHARS = 30

# Cap answers at this length. Roughly 350 tokens, leaves room for the question
# + Qwen2.5 chat-template overhead inside max_seq_length=1024.
MAX_ANSWER_CHARS = 1500

# Cap questions too -- a few MedQuAD entries are absurdly long.
MAX_QUESTION_CHARS = 400

# Citation-style markers like [L41539] in DrugBank are noise for an end-user model.
DRUGBANK_CITATION_RE = re.compile(r"\[[A-Z]\d+(?:,\s*[A-Z]\d+)*\]")

# Markdown emphasis underscores around drug names look weird stripped of context.
DRUGBANK_UNDERSCORE_RE = re.compile(r"_([^_]+)_")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Strip, collapse whitespace, undo Windows line endings."""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_drugbank_field(text: str) -> str:
    """DrugBank-specific scrub: drop citation tags and unwrap _emphasis_."""
    text = normalize_text(text)
    text = DRUGBANK_CITATION_RE.sub("", text)
    text = DRUGBANK_UNDERSCORE_RE.sub(r"\1", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def truncate_on_sentence_boundary(text: str, max_chars: int) -> str:
    """Cut to <= max_chars, prefer ending at the last sentence boundary."""
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    # Walk back to the last period/question/exclamation followed by space/eol.
    last_boundary = max(snippet.rfind(". "), snippet.rfind("? "), snippet.rfind("! "))
    if last_boundary > max_chars * 0.6:  # only honor boundary if it's not too far back
        snippet = snippet[: last_boundary + 1]
    return snippet.strip()


def make_pair(question: str, answer: str, source: str) -> dict | None:
    """Build a single training pair, or None if it fails the quality bar."""
    q = normalize_text(question)
    a = normalize_text(answer)

    if not q or not a:
        return None
    if len(a) < MIN_ANSWER_CHARS:
        return None
    if len(q) > MAX_QUESTION_CHARS:
        q = truncate_on_sentence_boundary(q, MAX_QUESTION_CHARS)
    if len(a) > MAX_ANSWER_CHARS:
        a = truncate_on_sentence_boundary(a, MAX_ANSWER_CHARS)

    return {
        "messages": [
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},
        ],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------
def load_medquad(path: Path) -> list[dict]:
    """MedQuAD: one row per Q/A, fields `question` and `answer` already split."""
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pair = make_pair(row.get("question", ""), row.get("answer", ""), "medquad")
            if pair is not None:
                pairs.append(pair)
    return pairs


def load_bioasq(path: Path) -> list[dict]:
    """BioASQ: question is `body`, answer is `ideal_answer`. Some are blank."""
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pair = make_pair(row.get("body", ""), row.get("ideal_answer", ""), "bioasq")
            if pair is not None:
                pairs.append(pair)
    return pairs


# DrugBank produces multiple Q/A pairs per record, one per templated field.
DRUGBANK_TEMPLATES: list[tuple[str, str]] = [
    # (question template, drugbank field name)
    ("What is {drug}?",                          "description"),
    ("What is {drug} used for?",                 "indication"),
    ("What are the side effects of {drug}?",     "side_effects"),
    ("How does {drug} work?",                    "mechanism_of_action"),
]


def load_drugbank(path: Path) -> list[dict]:
    """DrugBank: 1 record -> up to 4 templated Q/A pairs."""
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            drug = (row.get("drug_name") or "").strip()
            if not drug:
                continue
            for q_template, field_name in DRUGBANK_TEMPLATES:
                raw_answer = row.get(field_name) or ""
                if not raw_answer:
                    continue
                question = q_template.format(drug=drug)
                answer = clean_drugbank_field(raw_answer)
                pair = make_pair(question, answer, f"drugbank-{field_name}")
                if pair is not None:
                    pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------
def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(path: Path, stats: dict) -> None:
    """Drop a README.md next to the splits with the exact mix for reproducibility."""
    lines = [
        "# QLoRA training dataset",
        "",
        "Built by `scripts/prepare_qlora_dataset.py`. Schema is messages format:",
        "",
        "```json",
        '{"messages": [{"role": "user", "content": "..."},'
        ' {"role": "assistant", "content": "..."}], "source": "..."}',
        "```",
        "",
        "## Counts",
        "",
        "| Split | Pairs |",
        "|---|---|",
        f"| train | {stats['train']} |",
        f"| val | {stats['val']} |",
        f"| test | {stats['test']} |",
        f"| **total** | **{stats['total']}** |",
        "",
        "## Source breakdown (training split)",
        "",
        "| Source | Pairs |",
        "|---|---|",
    ]
    for source, count in sorted(stats["by_source"].items()):
        lines.append(f"| {source} | {count} |")
    lines += [
        "",
        "## Cleaning thresholds",
        "",
        f"- `MIN_ANSWER_CHARS = {MIN_ANSWER_CHARS}`",
        f"- `MAX_ANSWER_CHARS = {MAX_ANSWER_CHARS}` (truncated on sentence boundary)",
        f"- `MAX_QUESTION_CHARS = {MAX_QUESTION_CHARS}`",
        f"- `--per-source-limit = {stats['per_source_limit']}` (0 = no cap)",
        "",
        "## Reproducibility",
        "",
        "- Shuffle seed: 42",
        "- Split ratios: 90 / 5 / 5",
        "- Chat template (Qwen2.5) applied at train time, NOT here. Swap the base",
        "  model without rebuilding this dataset.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="Where to write train/val/test jsonl (default: data/qlora_training/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac",  type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--per-source-limit", type=int, default=1500,
                    help="Cap pairs per `source` tag (medquad, bioasq, and each "
                         "drugbank-<field>) after merge. Deterministic shuffle "
                         "then slice. Use 0 to disable. Default: 1500.")
    args = ap.parse_args()

    # Sanity: all three cleaned files must exist.
    for path in (MEDQUAD_PATH, BIOASQ_PATH, DRUGBANK_PATH):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    rng = random.Random(args.seed)

    print(f"[prep] loading MedQuAD from {MEDQUAD_PATH.name}")
    medquad  = load_medquad(MEDQUAD_PATH)
    print(f"[prep]   {len(medquad):>6} pairs after filtering")

    print(f"[prep] loading BioASQ from {BIOASQ_PATH.name}")
    bioasq   = load_bioasq(BIOASQ_PATH)
    print(f"[prep]   {len(bioasq):>6} pairs after filtering")

    print(f"[prep] loading DrugBank from {DRUGBANK_PATH.name}")
    drugbank = load_drugbank(DRUGBANK_PATH)
    print(f"[prep]   {len(drugbank):>6} templated pairs after filtering")

    all_pairs = medquad + bioasq + drugbank
    print(f"[prep] total before per-source cap: {len(all_pairs)} pairs")

    # Cap each `source` tag (medquad, bioasq, drugbank-<field>) at --per-source-limit.
    if args.per_source_limit and args.per_source_limit > 0:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in all_pairs:
            buckets[row["source"]].append(row)
        capped: list[dict] = []
        for source in sorted(buckets):
            rows = buckets[source]
            original = len(rows)
            if original > args.per_source_limit:
                rng.shuffle(rows)
                rows = rows[: args.per_source_limit]
            capped.extend(rows)
            print(f"[prep]   {source:<30} {original:>6} -> {len(rows):>6}")
        all_pairs = capped
        print(f"[prep] total after per-source cap: {len(all_pairs)} pairs")

    rng.shuffle(all_pairs)

    n = len(all_pairs)
    n_test = int(round(n * args.test_frac))
    n_val  = int(round(n * args.val_frac))
    test_rows  = all_pairs[:n_test]
    val_rows   = all_pairs[n_test : n_test + n_val]
    train_rows = all_pairs[n_test + n_val :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, args.out_dir / "train.jsonl")
    write_jsonl(val_rows,   args.out_dir / "val.jsonl")
    write_jsonl(test_rows,  args.out_dir / "test.jsonl")

    by_source = Counter(row["source"] for row in train_rows)
    stats = {
        "train": len(train_rows),
        "val":   len(val_rows),
        "test":  len(test_rows),
        "total": n,
        "by_source": dict(by_source),
        "per_source_limit": args.per_source_limit,
    }
    write_readme(args.out_dir / "README.md", stats)

    print()
    print(f"[prep] wrote splits to {args.out_dir}")
    print(f"[prep]   train: {stats['train']:>6}")
    print(f"[prep]   val:   {stats['val']:>6}")
    print(f"[prep]   test:  {stats['test']:>6}")
    print(f"[prep] training-split source mix:")
    for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"[prep]   {source:<30} {count:>6}")


if __name__ == "__main__":
    main()
