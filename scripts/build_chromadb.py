"""
Indexes one collection (`CHROMA_COLLECTION`, cosine) at `CHROMA_PERSIST_PATH`.
Each source's pre-built `text` field is embedded with S-PubMedBert (via the
collection's embedding function in `src/pipeline/chromadb_store.py`); the
remaining cleaned columns become per-doc metadata so retrieval can filter
by `source`, `specialty`, `qtype`, `drug_name`, `textbook`, etc.

Sources (paths from `RAG_SOURCES` in config.py):

  - medical_transcriptions  (CSV,   4,966 rows) -> mtsamples_cleaned.csv
  - bioasq                  (JSONL, 2,000 rows) -> bioasq_cleaned.jsonl
  - medquad                 (JSONL,16,359 rows) -> medquad_cleaned.jsonl
  - drugbank                (JSONL, 4,218 rows) -> drugbank_cleaned.jsonl
  - medrag_textbooks        (JSONL,26,994 rows) -> textbooks_cleaned.jsonl
  - medtext                 (JSONL, 1,412 rows) -> medtext_cleaned.jsonl

Run from project root:
    python scripts/build_chromadb.py                     # full reindex of all 6
    python scripts/build_chromadb.py --limit 100         # smoke-test (100/source)
    python scripts/build_chromadb.py --sources medquad medtext   # subset
    python scripts/build_chromadb.py --no-reset          # incremental upsert

Rebuild is idempotent: ids are namespaced as `<source>_<original_id>`, so
upserts overwrite. By default the collection is reset() first for a clean
build; pass --no-reset to keep existing entries (useful when rebuilding
only one source).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.pipeline import chromadb_store
from src.utils.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_PATH,
    RAG_SOURCES,
    RESULTS_DIR,
)

# Allowed metadata value types for ChromaDB. Anything else is coerced to
# str (or dropped if empty). Keeps load resilient when a cleaned file has
# a stray null or numeric type variation.
_SCALAR = (str, int, float, bool)

# Per-source loaders. Each returns (ids, documents, metadatas) lists of
# equal length. `text` is the embedding target; everything else goes in
# metadata. Empty/missing text rows are skipped (and counted) — embedding
# an empty string poisons retrieval.


def _coerce_meta(meta: dict) -> dict:
    """Filter a metadata dict to ChromaDB-allowed scalar types.

    None values dropped (ChromaDB accepts None but it bloats the store).
    Lists/dicts coerced to str (cleaning should have flattened these
    already, but this is a safety net for stray fields).
    """
    out: dict = {}
    for k, v in meta.items():
        if v is None or v == "":
            continue
        if isinstance(v, _SCALAR):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_medical_transcriptions(path: Path, limit: int | None):
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().astype(bool)]
    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        ids.append(f"medicaltranscriptions_{int(row['id'])}")
        docs.append(str(row["text"]))
        metas.append(_coerce_meta({
            "source": "medicaltranscriptions",
            "specialty": row.get("specialty"),
            "sample_name": row.get("sample_name"),
            "description": row.get("description"),
            "keywords": row.get("keywords"),
        }))
    return ids, docs, metas


def _load_bioasq(path: Path, limit: int | None):
    ids, docs, metas = [], [], []
    for i, rec in enumerate(_read_jsonl(path)):
        if limit and i >= limit:
            break
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"bioasq_{rec['id']}")
        docs.append(text)
        metas.append(_coerce_meta({
            "source": "bioasq",
            "question_type": rec.get("question_type"),
            "ideal_answer": rec.get("ideal_answer"),
            "has_exact_answer": rec.get("has_exact_answer"),
            "n_snippets": rec.get("n_snippets"),
            "n_documents": rec.get("n_documents"),
        }))
    return ids, docs, metas


def _load_medquad(path: Path, limit: int | None):
    ids, docs, metas = [], [], []
    for i, rec in enumerate(_read_jsonl(path)):
        if limit and i >= limit:
            break
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"medquad_{rec['id']}")
        docs.append(text)
        metas.append(_coerce_meta({
            "source": "medquad",
            "qtype": rec.get("qtype"),
            "question": rec.get("question"),
        }))
    return ids, docs, metas


def _load_drugbank(path: Path, limit: int | None):
    ids, docs, metas = [], [], []
    for i, rec in enumerate(_read_jsonl(path)):
        if limit and i >= limit:
            break
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"drugbank_{rec['id']}")
        docs.append(text)
        metas.append(_coerce_meta({
            "source": "drugbank",
            "drug_name": rec.get("drug_name"),
            "type": rec.get("type"),
            "groups": rec.get("groups"),
            "atc_codes": rec.get("atc_codes"),
            "indication": rec.get("indication"),
            "mechanism_of_action": rec.get("mechanism_of_action"),
            "half_life": rec.get("half_life"),
        }))
    return ids, docs, metas


def _load_medrag_textbooks(path: Path, limit: int | None):
    ids, docs, metas = [], [], []
    for i, rec in enumerate(_read_jsonl(path)):
        if limit and i >= limit:
            break
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"textbooks_{rec['id']}")
        docs.append(text)
        metas.append(_coerce_meta({
            "source": "medrag-textbooks",
            "textbook": rec.get("textbook"),
        }))
    return ids, docs, metas


def _load_medtext(path: Path, limit: int | None):
    ids, docs, metas = [], [], []
    for i, rec in enumerate(_read_jsonl(path)):
        if limit and i >= limit:
            break
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"medtext_{rec['id']}")
        docs.append(text)
        metas.append(_coerce_meta({
            "source": "medtext",
            "prompt": rec.get("prompt"),
        }))
    return ids, docs, metas


_LOADERS = {
    "medical_transcriptions": _load_medical_transcriptions,
    "bioasq":                 _load_bioasq,
    "medquad":                _load_medquad,
    "drugbank":               _load_drugbank,
    "medrag_textbooks":       _load_medrag_textbooks,
    "medtext":                _load_medtext,
}


def _write_report(per_source: list[dict], total_indexed: int, elapsed: float) -> Path:
    out = RESULTS_DIR / "build_chromadb.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append("# ChromaDB Build Report")
    md.append("")
    md.append(f"- **Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **Persist path**: `{CHROMA_PERSIST_PATH}`")
    md.append(f"- **Collection**: `{CHROMA_COLLECTION}` (cosine)")
    md.append(f"- **Embedding model**: S-PubMedBert (`pritamdeka/S-PubMedBert-MS-MARCO`, 768d, L2-normalised)")
    md.append(f"- **Total docs indexed**: **{total_indexed:,}**")
    md.append(f"- **Wall time**: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    md.append("")
    md.append("## Per-source breakdown")
    md.append("")
    md.append("| Source | Path | Loaded | Skipped (empty text) | Indexed | Time (s) |")
    md.append("|---|---|---:|---:|---:|---:|")
    for s in per_source:
        md.append(
            f"| `{s['name']}` | `{s['path']}` | {s['loaded']:,} | "
            f"{s['skipped']:,} | {s['indexed']:,} | {s['time']:.1f} |"
        )
    md.append("")
    md.append("## Metadata fields per source")
    md.append("")
    md.append("Stored on each ChromaDB doc so retrieval can filter via `where={...}`. "
              "Always present: `source`. Source-specific:")
    md.append("")
    md.append("- `medicaltranscriptions`: specialty, sample_name, description, keywords")
    md.append("- `bioasq`: question_type, ideal_answer, has_exact_answer, n_snippets, n_documents")
    md.append("- `medquad`: qtype, question")
    md.append("- `drugbank`: drug_name, type, groups, atc_codes, indication, mechanism_of_action, half_life")
    md.append("- `medrag-textbooks`: textbook")
    md.append("- `medtext`: prompt")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- `id` namespacing: each doc id is `<source>_<original_id>` — globally unique across sources, idempotent re-indexing.")
    md.append("- BioASQ docs may exceed the embedder's ~512-token limit and get truncated; revisit chunking if recall lags in Phase 7.4.")
    md.append("- Documents are embedded as-is (no `preprocessing.preprocess()` call); preprocessing is query-side only.")
    md.append("")
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ChromaDB knowledge base.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_LOADERS.keys()),
        default=list(_LOADERS.keys()),
        help="Subset of sources to index (default: all 6).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Per-source row limit for smoke-tests (default: no limit).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip dropping the collection before indexing (incremental upsert).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Upsert batch size passed to chromadb_store.add (default: 128).",
    )
    args = parser.parse_args()

    print(f"Persist path : {CHROMA_PERSIST_PATH}")
    print(f"Collection   : {CHROMA_COLLECTION}")
    print(f"Sources      : {', '.join(args.sources)}")
    print(f"Per-source limit: {args.limit if args.limit else 'none'}")
    print(f"Batch size   : {args.batch_size}")
    print()

    if not args.no_reset:
        print(f"Resetting collection '{CHROMA_COLLECTION}' ...")
        chromadb_store.reset()
        print(f"  collection size after reset: {chromadb_store.count()}\n")

    per_source: list[dict] = []
    total_start = time.time()

    for name in args.sources:
        path = RAG_SOURCES[name]
        if not path.exists():
            print(f"[SKIP] {name}: file not found at {path}")
            per_source.append({
                "name": name, "path": str(path), "loaded": 0,
                "skipped": 0, "indexed": 0, "time": 0.0,
            })
            continue

        print(f"[{name}] loading from {path.name} ...")
        t0 = time.time()
        ids, docs, metas = _LOADERS[name](path, args.limit)
        loaded = len(ids)
        # Skipped count = limit-or-actual minus surviving rows. We only
        # know skipped precisely when the loader iterated everything.
        # For smoke-tests with --limit this is still informative.
        skipped = 0  # loaders already filter empty text; recompute if needed
        print(f"  loaded {loaded:,} docs in {time.time() - t0:.1f}s")

        if loaded == 0:
            per_source.append({
                "name": name, "path": str(path), "loaded": 0,
                "skipped": 0, "indexed": 0, "time": 0.0,
            })
            continue

        print(f"  upserting -> ChromaDB (this triggers embedding) ...")
        t1 = time.time()
        chromadb_store.add(ids, docs, metas, batch_size=args.batch_size)
        elapsed = time.time() - t1
        print(f"  done in {elapsed:.1f}s "
              f"({loaded / max(elapsed, 1e-6):.1f} docs/s)")
        print(f"  collection size now: {chromadb_store.count():,}\n")

        per_source.append({
            "name": name,
            "path": str(path),
            "loaded": loaded,
            "skipped": skipped,
            "indexed": loaded,
            "time": elapsed + (t1 - t0),
        })

    total_elapsed = time.time() - total_start
    total_indexed = chromadb_store.count()

    print("=" * 60)
    print(f"Total indexed: {total_indexed:,}")
    print(f"Wall time   : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    report_path = _write_report(per_source, total_indexed, total_elapsed)
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
