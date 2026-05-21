"""Medical NER — scispaCy `en_ner_bc5cdr_md` wrapper.

Extracts DISEASE and CHEMICAL entities from clinical text. Run on the
output of `preprocessing.preprocess()` (case preserved so the model sees
natural text). spaCy 3.7.x is required
"""
import spacy

from src.utils.config import NER_MODEL

_nlp = spacy.load(
    NER_MODEL,
    disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
)

# Railguard 1 inserts these literal placeholders; scispaCy can false-positive
# their constituent words (e.g. "NAME" -> CHEMICAL). Drop spans whose text
# is entirely placeholder words.
_PLACEHOLDER_WORDS = {"PATIENT", "NAME", "PHONE", "EMAIL", "ID", "LOCATION", "DATE"}


def _is_placeholder_ent(span) -> bool:
    tokens = [t for t in span.text.upper().replace("[", " ").replace("]", " ").split() if t]
    return bool(tokens) and all(t in _PLACEHOLDER_WORDS for t in tokens)


def _filter_ents(ents):
    return [e for e in ents if not _is_placeholder_ent(e)]


def to_doc(text: str):
    """Return the raw spaCy Doc. Use this in the UI for displacy rendering."""
    doc = _nlp(text)
    doc.ents = tuple(_filter_ents(doc.ents))
    return doc


def extract(text: str) -> list[dict]:
    """Extract DISEASE and CHEMICAL entities.

    Returns a list of {"text", "label", "start", "end"} dicts. Character
    offsets are into the input `text` so the UI can highlight in-place.
    """
    doc = _nlp(text)
    return [
        {
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        }
        for ent in _filter_ents(doc.ents)
        if ent.label_ in {"DISEASE", "CHEMICAL"}
    ]


if __name__ == "__main__":
    sample = (
        "Patient presents with hypertension and type 2 diabetes mellitus. "
        "Started on metformin and lisinopril. History of myocardial "
        "infarction in 2018. Reports chest pain and shortness of breath."
    )
    print("INPUT:", sample, "\n")
    for ent in extract(sample):
        print(f"  {ent['label']:8} {ent['text']!r}  [{ent['start']}:{ent['end']}]")
