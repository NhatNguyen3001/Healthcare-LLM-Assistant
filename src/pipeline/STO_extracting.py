"""Structured clinical extraction — symptoms, treatments, outcomes.

Complements `src/pipeline/ner.py` (BC5CDR — DISEASE / CHEMICAL only) by
producing the structured findings the project needs to display and feed
into the prompt builder. Outcomes are inherently discourse-level
("symptoms resolved at 2-week follow-up") and cannot be recovered with
span-based NER, so this step uses an LLM.

Runs in parallel with NER / sentiment / classifier (it's one network call,
not local CPU). Output shape:

    {
        "symptoms":   [str, ...],
        "treatments": [str, ...],
        "outcomes":   [str, ...],
    }

Uses `MODEL_AGENTS` (GPT-5.4-mini), the same tier as the MASS-RAG filter
agents - this is a deterministic processing step, not the user-facing
recommendation.
"""
import json
import re
from typing import TypedDict

from src.pipeline.llm import generate
from src.utils.config import MODEL_AGENTS


class Findings(TypedDict):
    symptoms: list[str]
    treatments: list[str]
    outcomes: list[str]


def _empty() -> Findings:
    return {"symptoms": [], "treatments": [], "outcomes": []}


_SYSTEM = (
    "You are a clinical information extraction assistant. From a patient "
    "case, extract three lists:\n"
    "  - symptoms: subjective complaints and objective signs the patient or "
    "provider reports (e.g. chest pain, fever, bilateral leg swelling).\n"
    "  - treatments: medications, procedures, surgeries, lifestyle "
    "interventions, devices (e.g. metformin, appendectomy, low-salt diet, "
    "PCI, CPAP).\n"
    "  - outcomes: response, status changes, prognosis, results "
    "(e.g. symptoms resolved, no response to therapy, discharged on day 3, "
    "recurrence at 6 months).\n\n"
    "Return ONLY valid JSON with exactly the keys 'symptoms', 'treatments', "
    "'outcomes'. No prose, no markdown fences. Use an empty list [] for "
    "categories with no information present in the input. Keep each item "
    "concise (a noun phrase, not a sentence) and copy phrasing from the "
    "input where possible."
)

_FEWSHOT = (
    'Example.\n\n'
    'Input:\n'
    '"58-year-old presented with chest pain and shortness of breath. ECG '
    'showed ST elevation. Started on aspirin and beta-blocker, taken to cath '
    'lab for PCI. Discharged on day 3, symptoms resolved at 2-week '
    'follow-up."\n\n'
    'Output:\n'
    '{"symptoms": ["chest pain", "shortness of breath", "ST elevation on ECG"], '
    '"treatments": ["aspirin", "beta-blocker", "PCI"], '
    '"outcomes": ["discharged on day 3", "symptoms resolved at 2-week follow-up"]}\n'
)

_RE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_RE_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse(raw: str) -> Findings:
    """Tolerant JSON parse — strips markdown fences, grabs the first {...}
    block, fills missing keys with []. Never raises."""
    if not raw:
        return _empty()
    cleaned = _RE_FENCE.sub("", raw.strip())
    match = _RE_JSON_BLOCK.search(cleaned)
    if not match:
        return _empty()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    return {
        "symptoms":   _coerce_list(data.get("symptoms")),
        "treatments": _coerce_list(data.get("treatments")),
        "outcomes":   _coerce_list(data.get("outcomes")),
    }


def extract(text: str) -> Findings:
    """Extract symptoms, treatments, outcomes from a clinical text.

    Returns {"symptoms": [...], "treatments": [...], "outcomes": [...]}.
    On any failure (empty input, network error, malformed JSON) returns
    a dict with empty lists — never raises, so it can't break the pipeline.
    """
    if not text or not text.strip():
        return _empty()
    prompt = (
        f"{_FEWSHOT}\n"
        "Now extract from this case:\n\n"
        f"{text}\n\n"
        "Output:"
    )
    try:
        raw = generate(
            prompt,
            system=_SYSTEM,
            enable_web_search=False,
            model=MODEL_AGENTS,
        )
    except Exception:
        return _empty()
    return _parse(raw)


if __name__ == "__main__":
    sample = (
        "Patient with hypertension and type 2 diabetes mellitus presents "
        "with shortness of breath and bilateral leg swelling. Started on "
        "furosemide and lisinopril; advised low-salt diet and daily weight "
        "monitoring. At 2-week review, swelling reduced and dyspnea improved; "
        "weight down 3 kg."
    )
    findings = extract(sample)
    print("INPUT:", sample, "\n")
    for key in ("symptoms", "treatments", "outcomes"):
        print(f"  {key:10}: {findings[key]}")
