"""ChromaDB client + single-collection helpers.

One persistent collection (`CHROMA_COLLECTION`) at `CHROMA_PERSIST_PATH`,
cosine distance. The collection's embedding function delegates to
`src.pipeline.embeddings.encode()` so the same S-PubMedBert is used for
both indexing (scripts/build_chromadb.py, Phase 5) and query time.
"""
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from src.pipeline import embeddings
from src.utils.config import (
    CHROMA_COLLECTION,
    CHROMA_K,
    CHROMA_PERSIST_PATH,
)


class _PubMedBertEF(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "pubmedbert"

    def __call__(self, input: Documents) -> Embeddings:
        return embeddings.encode(list(input)).tolist()


_ef = _PubMedBertEF()
_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        CHROMA_PERSIST_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
    return _client


def get_collection():
    """Get-or-create the single project collection. Embedding function is
    bound automatically — callers should never pass `embeddings=` directly."""
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=_ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add(
    ids: list[str],
    documents: list[str],
    metadatas: Optional[list[dict]] = None,
    batch_size: int = 128,
) -> None:
    """Upsert documents in batches. ChromaDB calls our embedding function
    on each batch. Use stable ids so re-indexing is idempotent."""
    coll = get_collection()
    n = len(ids)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        coll.upsert(
            ids=ids[i:j],
            documents=documents[i:j],
            metadatas=(metadatas[i:j] if metadatas else None),
        )


def query(
    text: str,
    k: int = CHROMA_K,
    where: Optional[dict] = None,
) -> list[dict]:
    """Top-k retrieval, deduped by document text.

    Different IDs across sources can hold near-identical content (e.g. the
    same diabetes Q&A in BioASQ and MedQuAD), so we over-fetch 3x and drop
    duplicates on a whitespace+case-normalized fingerprint before truncating
    to k. Returns [{id, document, metadata, distance}, ...]."""
    coll = get_collection()
    res = coll.query(query_texts=[text], n_results=k * 3, where=where)
    ids, docs, metas, dists = (
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0] if res.get("metadatas") else [None] * len(res["ids"][0]),
        res["distances"][0],
    )
    seen: set[str] = set()
    out: list[dict] = []
    for i, d, m, dist in zip(ids, docs, metas, dists):
        fp = " ".join((d or "").lower().split())
        if fp in seen:
            continue
        seen.add(fp)
        out.append({"id": i, "document": d, "metadata": m, "distance": dist})
        if len(out) >= k:
            break
    return out


def count() -> int:
    return get_collection().count()


def reset() -> None:
    """Drop the collection. `scripts/build_chromadb.py` calls this before
    a full reindex."""
    global _collection
    try:
        _get_client().delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    _collection = None


if __name__ == "__main__":
    sample_docs = [
        ("doc-cardio-1",
         "Acute myocardial infarction presents with chest pain, ECG changes, and elevated troponin.",
         {"source": "smoke_test", "topic": "cardiology"}),
        ("doc-cardio-2",
         "Hypertension is managed with lifestyle changes and antihypertensives like lisinopril.",
         {"source": "smoke_test", "topic": "cardiology"}),
        ("doc-ortho-1",
         "ACL tears occur in sports with sudden direction changes; MRI is the standard confirmation.",
         {"source": "smoke_test", "topic": "orthopedics"}),
        ("doc-neuro-1",
         "Migraines present with unilateral throbbing pain, photophobia, and may include visual aura.",
         {"source": "smoke_test", "topic": "neurology"}),
    ]
    ids, docs, metas = (list(t) for t in zip(*sample_docs))
    add(ids, docs, metas)
    print(f"Collection size after upsert: {count()}")

    hits = query("patient with crushing chest pain and diaphoresis", k=3)
    print(f"\nTop {len(hits)} hits:")
    for h in hits:
        topic = (h["metadata"] or {}).get("topic", "?")
        print(f"  [{h['distance']:.3f}] {h['id']}  ({topic})")
        print(f"    {h['document']}")
