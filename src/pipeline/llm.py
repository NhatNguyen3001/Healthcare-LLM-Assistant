"""OpenAI LLM wrappers via the Responses API, plus a router to the local QLoRA models.

Three roles, three models on the cloud side:
- `generate()`        GPT-5.5 (`MODEL_GENERATION`) for Synthesis + final LLM generation
- `generate_async()`  GPT-5.4-mini (`MODEL_AGENTS`) for the three parallel MASS-RAG filter agents
- `judge()`           GPT-5.4 (`MODEL_JUDGE`) for eval notebooks (judge must differ from generator)

`generate()` / `generate_stream()` take a `model_choice` flag with three values:
- "cloud"          legacy Responses-API path (GPT-5.5)
- "local_qwen"     QLoRA Qwen2.5-1.5B via Ollama (`LOCAL_MODEL_QWEN`)
- "local_llama32"  QLoRA Llama-3.2-1B via Ollama (`LOCAL_MODEL_LLAMA32`)

Both local options hit the same Ollama daemon on a different tag; the daemon
handles model lifecycle. MASS-RAG filter agents (`generate_async`),
`rewrite_query`, and `judge` always stay cloud — they're never routed.

Web search is a Responses API built-in tool. Enable it at the final
LLM-generation step (the model decides whether to actually call it) and
in the 0-hit retrieval fallback. Synthesis stays tool-free. `enable_web_search`
is silently ignored on a local choice since local models have no web tool.
"""
from typing import Iterator, Literal, Optional

from openai import AsyncOpenAI, OpenAI

from src.pipeline import local_llm
from src.utils.config import (
    LOCAL_MODEL_LLAMA32,
    LOCAL_MODEL_QWEN,
    MODEL_AGENTS,
    MODEL_GENERATION,
    MODEL_JUDGE,
    OPENAI_API_KEY,
)

ModelChoice = Literal["cloud", "local_qwen", "local_llama32"]

# Map each local UI choice to the Ollama tag served by the daemon.
_LOCAL_TAGS: dict[str, str] = {
    "local_qwen":    LOCAL_MODEL_QWEN,
    "local_llama32": LOCAL_MODEL_LLAMA32,
}

_client = OpenAI(api_key=OPENAI_API_KEY)
_aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _input(prompt: str, system: Optional[str]) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def generate(
    prompt: str,
    system: Optional[str] = None,
    enable_web_search: bool = False,
    model: str = MODEL_GENERATION,
    model_choice: ModelChoice = "cloud",
) -> str:
    """Sync chat completion. Routes to cloud Responses API (default) or one of
    two local Ollama-served QLoRA models when `model_choice` is "local_qwen"
    or "local_llama32". `enable_web_search` and `model` are cloud-only and
    silently ignored on either local path."""
    if model_choice in _LOCAL_TAGS:
        return local_llm.generate(prompt, system=system, model=_LOCAL_TAGS[model_choice])
    tools = [{"type": "web_search"}] if enable_web_search else None
    resp = _client.responses.create(
        model=model,
        input=_input(prompt, system),
        tools=tools,
    )
    return resp.output_text


async def generate_async(
    prompt: str,
    system: Optional[str] = None,
    model: str = MODEL_AGENTS,
) -> str:
    """Async call for the three parallel MASS-RAG filter agents
    (`asyncio.gather`). No tools — agents are deterministic processors."""
    resp = await _aclient.responses.create(
        model=model,
        input=_input(prompt, system),
    )
    return resp.output_text


def generate_stream(
    prompt: str,
    system: Optional[str] = None,
    enable_web_search: bool = False,
    model: str = MODEL_GENERATION,
    model_choice: ModelChoice = "cloud",
) -> Iterator[str]:
    """Stream text deltas. Routes to cloud Responses API or one of the two
    local Ollama-served QLoRA models based on `model_choice`. Yields plain
    text chunks suitable for `st.write_stream`. `enable_web_search` and
    `model` are cloud-only."""
    if model_choice in _LOCAL_TAGS:
        yield from local_llm.generate_stream(
            prompt, system=system, model=_LOCAL_TAGS[model_choice],
        )
        return
    tools = [{"type": "web_search"}] if enable_web_search else None
    stream = _client.responses.create(
        model=model,
        input=_input(prompt, system),
        tools=tools,
        stream=True,
    )
    for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            if delta:
                yield delta


def rewrite_query(current_input: str, last_turns: list[dict]) -> str:
    """Rewrite a follow-up query as a standalone search query using prior turns.

    `last_turns` is a list of {"input": str, "recommendation": str} dicts,
    oldest-first. Returns `current_input` unchanged when history is empty —
    no LLM call. On any failure (network, parse) returns `current_input`
    so retrieval still runs.

    Used before ChromaDB retrieval so pronoun-only follow-ups like
    "what about side effects?" become "side effects of <prior medication>".
    """
    if not last_turns:
        return current_input

    def _trim(text: str, n: int) -> str:
        text = (text or "").strip()
        return text if len(text) <= n else text[:n] + " [...]"

    history = "\n\n".join(
        f"Previous user input:\n{_trim(t.get('input',''), 200)}\n"
        f"Assistant recommendation:\n{_trim(t.get('recommendation',''), 400)}"
        for t in last_turns
    )
    system = (
        "You rewrite a follow-up clinical question into a single standalone "
        "search query suitable for a medical retrieval system. Resolve "
        "pronouns and elliptical references using the conversation. Output "
        "ONLY the rewritten query as a single concise line (max ~25 words) "
        "— no prose, no quotes, no preamble."
    )
    prompt = (
        f"{history}\n\n"
        f"New user input:\n{current_input}\n\n"
        "Standalone search query:"
    )
    try:
        out = generate(prompt, system=system, enable_web_search=False, model=MODEL_AGENTS)
    except Exception:
        return current_input
    out = (out or "").strip().strip('"').strip("'")
    # Hard cap the rewrite so a verbose mini response can't bloat the
    # downstream MASS-RAG filter agent prompts.
    if len(out) > 300:
        out = out[:300]
    return out or current_input


def judge(
    question: str,
    answer: str,
    criteria: str,
    model: str = MODEL_JUDGE,
) -> str:
    """LLM-as-judge for the evaluation notebooks. Uses a different model
    than `MODEL_GENERATION` so the generator isn't grading itself."""
    system = (
        "You are an impartial evaluator. Given a question, a candidate "
        "answer, and evaluation criteria, return a short structured "
        "assessment with a numerical score from 0 to 10 and a brief rationale."
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Evaluation criteria:\n{criteria}"
    )
    resp = _client.responses.create(
        model=model,
        input=_input(prompt, system),
    )
    return resp.output_text


if __name__ == "__main__":
    print("--- generate (no tools) ---")
    print(generate(
        "In one sentence, how does aspirin work?",
        system="You are a concise medical educator.",
    ))

    print("\n--- generate (web_search enabled) ---")
    print(generate(
        "What is the current first-line treatment for uncomplicated hypertension?",
        system="You are a medical assistant. Use web_search if you need current guidelines.",
        enable_web_search=True,
    ))

    print("\n--- judge ---")
    print(judge(
        question="What causes type 2 diabetes?",
        answer="Type 2 diabetes is caused by insulin resistance and a relative insulin deficiency.",
        criteria="Medical accuracy, clarity, and completeness.",
    ))
