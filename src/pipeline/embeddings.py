"""Sentence embeddings — `pritamdeka/S-PubMedBert-MS-MARCO` via sentence-transformers.

Used by `chromadb_store.py` for indexing the six RAG sources and for
query-time retrieval. The model is loaded lazily so importing this module is cheap.
"""
from typing import Union

import numpy as np

from src.utils.config import EMBEDDING_MODEL

_model = None


def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(
    texts: Union[str, list[str]],
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode one string or a list of strings.

    Returns a 1-D vector for a single string input, otherwise a 2-D array
    of shape (n_texts, embedding_dim). Vectors are L2-normalised so cosine
    similarity equals dot product — what ChromaDB's default metric wants.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    vecs = _load().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs[0] if single else vecs


def dim() -> int:
    """Embedding dimension. Useful for ChromaDB collection setup."""
    return _load().get_sentence_embedding_dimension()


if __name__ == "__main__":
    samples = [
        "patient presents with chest pain and shortness of breath",
        "metformin is used to treat type 2 diabetes mellitus",
        "MRI of the lumbar spine shows L4-L5 disc herniation",
        "history of myocardial infarction in 2018",
        "EEG findings consistent with generalized epilepsy",
        "lisinopril 10mg daily for hypertension",
        "fracture of the distal radius following a fall",
        "patient reports recurrent migraines with aura",
        "chronic obstructive pulmonary disease, on home oxygen",
        "elevated liver enzymes on routine bloodwork",
    ]
    vecs = encode(samples)
    print(f"Encoded {len(samples)} texts -> shape {vecs.shape}, dtype {vecs.dtype}")
    print(f"Embedding dim: {dim()}")
    print(f"First vector norm: {np.linalg.norm(vecs[0]):.4f}  (should be ~1.0)")

    # Quick sanity check: clinically related items should be closer than unrelated ones
    sim_related = float(vecs[3] @ vecs[0])   # MI history vs chest pain
    sim_random  = float(vecs[3] @ vecs[2])   # MI history vs lumbar MRI
    print(f"cos(MI hx, chest pain) = {sim_related:.3f}")
    print(f"cos(MI hx, lumbar MRI) = {sim_random:.3f}")
