"""MASS-RAG — multi-agent retrieval evidence synthesis.

Reference: Xiao et al. (2026) arXiv:2604.18509. Answer agent is NOT used;
Synthesis works directly from the three filter agents' outputs.

Three filter agents (Summarizer / Extractor / Reasoner) run in parallel
on retrieved ChromaDB documents using GPT-5.4-mini (`MODEL_AGENTS`) via
`asyncio.gather`. A Synthesis agent (GPT-5.5, `MODEL_GENERATION`, no
tools) reconciles the three outputs into a single evidence block.

This module returns the evidence block only. The orchestrator (app.py)
is responsible for the FINAL user-facing LLM call which combines
NER + extracted findings + this evidence + few-shot + CoT, with
`enable_web_search=True` so the model can supplement guidelines.

Routing:
  - Drop hits with cosine distance > `CHROMA_DISTANCE_THRESHOLD` (0.8).
  - If 0 hits survive, skip MASS-RAG entirely. The caller passes
    `result["fallback_prompt"]` to `llm.generate(enable_web_search=True)`
    instead of using `result["evidence"]`.

Each filter prompt follows ReAct (reason -> act -> reason) — one of the
five required prompt techniques.
"""
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from src.pipeline import llm
from src.utils.config import CHROMA_DISTANCE_THRESHOLD, MODEL_AGENTS


class MassRAGResult(TypedDict):
    evidence: str               # synthesised evidence block ("" when fallback)
    agents: dict                # raw per-agent outputs ({} when fallback)
    retained_hits: list[dict]   # hits that survived the distance threshold
    fallback: bool              # True when 0 hits survived
    fallback_prompt: str        # populated when fallback; pass to generate(enable_web_search=True)


# --- agent prompts --------------------------------------------------------

_SYS_SUMMARIZER = (
    "You are the Summarizer agent in a clinical retrieval-augmented system. "
    "Produce an abstractive, query-relevant summary of the provided "
    "documents. Do NOT answer the user's query directly — only summarise "
    "what the retrieved passages say that is relevant to it."
)

_SYS_EXTRACTOR = (
    "You are the Extractor agent in a clinical retrieval-augmented system. "
    "Copy verbatim factual spans from the retrieved documents that are "
    "relevant to the query — exact text only, no paraphrasing. Tag each "
    "span with its document id in the form [doc_id]."
)

_SYS_REASONER = (
    "You are the Reasoner agent in a clinical retrieval-augmented system. "
    "Identify implicit connections, agreements, and contradictions ACROSS "
    "the retrieved documents that are relevant to the query. Do NOT "
    "introduce external knowledge — every inference must be grounded in "
    "the retrieved text."
)

_SYS_SYNTHESIS = (
    "You are the Synthesis agent. You receive three parallel analyses of "
    "the same retrieved documents (a summary, verbatim extracted spans, "
    "and cross-document reasoning) and reconcile them into a single "
    "coherent evidence block for a downstream clinical recommender. Your "
    "output is evidence ONLY — do not give a recommendation or diagnosis."
)

# ReAct scaffold — reason -> act -> reason. The model is asked to reason
# internally and only emit the final output, so the visible answer stays
# clean for the synthesis step.
_REACT_TEMPLATE = (
    "Reason: think step by step about what the user's query is asking and "
    "which retrieved passages bear on it.\n"
    "Act: produce your output as instructed in your system message.\n"
    "Reason again: briefly check your output covers the query and is "
    "grounded in the passages.\n\n"
    "User query:\n{query}\n\n"
    "Retrieved documents:\n{docs}\n\n"
    "Now respond with the final output ONLY — do not include your "
    "intermediate reasoning."
)

_FALLBACK_PROMPT_TEMPLATE = (
    "No documents in the local clinical knowledge base were sufficiently "
    "similar to the user's query (cosine distance > {threshold} for every "
    "retrieved hit). Use web_search to find current clinical guidance to "
    "address the query below. Be cautious, cite sources where possible, "
    "and remind the user this is not a definitive diagnosis.\n\n"
    "User query:\n{query}"
)


# --- helpers --------------------------------------------------------------

# Truncate per-doc text before sending to filter agents. Long BioASQ /
# DrugBank entries can blow up filter-agent token counts (and wall-clock)
# when k retrieved docs are all large. 400 chars covers a tight summary —
# enough for the agents to extract / cite without bloating the prompt.
_MAX_DOC_CHARS = 400


def _format_docs(hits: list[dict]) -> str:
    """Render retained hits for prompt inclusion. One block per hit with id,
    distance, and metadata so the agents can cite. Doc text is truncated to
    `_MAX_DOC_CHARS` to keep agent prompts small."""
    blocks = []
    for h in hits:
        meta = h.get("metadata") or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) or "-"
        text = h["document"]
        if len(text) > _MAX_DOC_CHARS:
            text = text[:_MAX_DOC_CHARS] + " [...truncated]"
        blocks.append(
            f"[{h['id']}] (distance={h.get('distance', 0.0):.3f}, {meta_str})\n"
            f"{text}"
        )
    return "\n\n".join(blocks)


def filter_hits(
    hits: list[dict],
    threshold: float = CHROMA_DISTANCE_THRESHOLD,
) -> list[dict]:
    """Drop hits with cosine distance above `threshold`."""
    return [h for h in hits if h.get("distance", 1.0) <= threshold]


# --- agents ---------------------------------------------------------------

def _run_agent_sync(system: str, query: str, docs_block: str) -> str:
    prompt = _REACT_TEMPLATE.format(query=query, docs=docs_block)
    return llm.generate(
        prompt, system=system, enable_web_search=False, model=MODEL_AGENTS,
    )


def _run_filters_sync(query: str, docs_block: str) -> dict:
    """Run the 3 filter agents in parallel via threads.

    Was async/asyncio.gather, but Streamlit's per-script `asyncio.run()` calls
    caused the module-level AsyncOpenAI client's httpx pool to bind to the
    first event loop and hang on subsequent runs. Threads share the sync
    OpenAI client cleanly — each call is one HTTP request, so threads are a
    fine substitute for true async I/O.
    """
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_sum = ex.submit(_run_agent_sync, _SYS_SUMMARIZER, query, docs_block)
        f_ext = ex.submit(_run_agent_sync, _SYS_EXTRACTOR,  query, docs_block)
        f_rea = ex.submit(_run_agent_sync, _SYS_REASONER,   query, docs_block)
        return {
            "summarizer": f_sum.result(),
            "extractor":  f_ext.result(),
            "reasoner":   f_rea.result(),
        }


def _synthesize(query: str, agents: dict) -> str:
    prompt = (
        f"User query:\n{query}\n\n"
        f"Summarizer output:\n{agents['summarizer']}\n\n"
        f"Extractor output (verbatim spans):\n{agents['extractor']}\n\n"
        f"Reasoner output (cross-document connections):\n{agents['reasoner']}\n\n"
        "Reconcile these into a single evidence block. Resolve any "
        "contradictions explicitly. Preserve any document ids cited by the "
        "Extractor. Output evidence only — no recommendation."
    )
    return llm.generate(prompt, system=_SYS_SYNTHESIS, enable_web_search=False)


# --- entry points ---------------------------------------------------------

def run_sync(
    query: str,
    hits: list[dict],
    threshold: float = CHROMA_DISTANCE_THRESHOLD,
) -> MassRAGResult:
    """Sync entry point. `hits` is the raw output of `chromadb_store.query()`.

    On the normal path: filters hits, runs the 3 agents in parallel via
    threads, then synthesises. On the fallback path (0 hits survive
    `threshold`): returns a populated `fallback_prompt` and empty
    agents/evidence.
    """
    retained = filter_hits(hits, threshold)
    if not retained:
        return {
            "evidence":        "",
            "agents":          {},
            "retained_hits":   [],
            "fallback":        True,
            "fallback_prompt": _FALLBACK_PROMPT_TEMPLATE.format(
                threshold=threshold, query=query,
            ),
        }
    docs_block = _format_docs(retained)
    t0 = time.perf_counter()
    agents = _run_filters_sync(query, docs_block)
    t_filters = time.perf_counter() - t0
    print(f"[massrag] filters (3 parallel gpt-5.4-mini): {t_filters:.2f}s", file=sys.stderr)
    t0 = time.perf_counter()
    evidence = _synthesize(query, agents)
    t_synth = time.perf_counter() - t0
    print(f"[massrag] synthesis (gpt-5.5): {t_synth:.2f}s", file=sys.stderr)
    print(f"[massrag] total agent calls: {t_filters + t_synth:.2f}s", file=sys.stderr)
    agent_chars = sum(len(v) for v in agents.values())
    print(f"[massrag] agent output chars: {agent_chars} (feeds synthesis prompt)", file=sys.stderr)
    return {
        "evidence":        evidence,
        "agents":          agents,
        "retained_hits":   retained,
        "fallback":        False,
        "fallback_prompt": "",
    }


async def run(
    query: str,
    hits: list[dict],
    threshold: float = CHROMA_DISTANCE_THRESHOLD,
) -> MassRAGResult:
    """Async wrapper for any future async caller. Offloads `run_sync` to a
    thread so the caller's event loop is not blocked. The Streamlit UI uses
    `run_sync` directly — see the note in `_run_filters_sync` for why."""
    return await asyncio.to_thread(run_sync, query, hits, threshold)


if __name__ == "__main__":
    fake_hits = [
        {
            "id":       "doc-cardio-1",
            "document": "Acute myocardial infarction presents with chest pain, "
                        "ECG changes (ST elevation), and elevated troponin. "
                        "Immediate aspirin and reperfusion (PCI preferred) are standard.",
            "metadata": {"source": "smoke", "topic": "cardiology"},
            "distance": 0.32,
        },
        {
            "id":       "doc-cardio-2",
            "document": "Unstable angina shares symptoms with NSTEMI but lacks "
                        "troponin elevation. Risk stratification with TIMI or "
                        "GRACE score guides management.",
            "metadata": {"source": "smoke", "topic": "cardiology"},
            "distance": 0.55,
        },
    ]
    user_query = (
        "55-year-old with crushing substernal chest pain radiating to the "
        "left arm, diaphoretic, ECG pending — what should I think about?"
    )

    print("--- normal path ---")
    result = run_sync(user_query, fake_hits)
    print(f"fallback: {result['fallback']}")
    print(f"retained: {len(result['retained_hits'])}")
    print("\nEVIDENCE:")
    print(result["evidence"])

    print("\n--- fallback path (all hits exceed threshold) ---")
    bad_hits = [{**h, "distance": 0.95} for h in fake_hits]
    fb = run_sync(user_query, bad_hits)
    print(f"fallback: {fb['fallback']}")
    print("\nFALLBACK PROMPT (caller passes this to generate(enable_web_search=True)):")
    print(fb["fallback_prompt"])
