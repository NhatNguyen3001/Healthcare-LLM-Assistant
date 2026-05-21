"""Local LLM wrapper — talks to Ollama serving our QLoRA-fine-tuned Qwen2.5-1.5B.

Mirrors `llm.generate()` / `llm.generate_stream()` in signature so `llm.py` can
route between cloud and local on a single `model_choice` flag without the caller
caring which path actually runs.

`enable_web_search` from `llm.py` is intentionally absent here — local models
have no web tool. The router in `llm.py` drops that argument when routing local.
MASS-RAG filter agents (`generate_async`) and `judge` stay on the cloud path
unconditionally; we don't reimplement them here.

Uses sync `requests` only. Streamlit's script-rerun model + an async HTTP client
deadlocks (httpx pool binds to the first event loop and hangs on the next call);
"""
from typing import Iterator, Optional

import json
import requests

from src.utils.config import LOCAL_MODEL_NAME, OLLAMA_BASE_URL


_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

# A demo request shouldn't hang forever if Ollama is wedged.
_REQUEST_TIMEOUT = (5, 120)  # (connect, read) seconds


def _messages(prompt: str, system: Optional[str]) -> list[dict]:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _connection_hint(exc: Exception) -> str:
    return (
        f"Could not reach Ollama at {OLLAMA_BASE_URL}: {exc}\n"
        "Hints:\n"
        "  - Is the Ollama daemon running? (Windows: tray icon; Linux: `systemctl status ollama`)\n"
        f"  - Is the model registered? Try `ollama list` — should include `{LOCAL_MODEL_NAME}`.\n"
        "  - If not, from models/qwen-medqa-gguf/: `ollama create medqa-qwen -f Modelfile`."
    )


def generate(
    prompt: str,
    system: Optional[str] = None,
    model: str = LOCAL_MODEL_NAME,
) -> str:
    """Non-streaming chat completion via Ollama. Returns the full assistant reply."""
    payload = {
        "model":    model,
        "messages": _messages(prompt, system),
        "stream":   False,
    }
    try:
        resp = requests.post(_CHAT_ENDPOINT, json=payload, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(_connection_hint(e)) from e
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def generate_stream(
    prompt: str,
    system: Optional[str] = None,
    model: str = LOCAL_MODEL_NAME,
) -> Iterator[str]:
    """Stream content deltas from Ollama. Yields plain text chunks suitable
    for `st.write_stream` in the UI."""
    payload = {
        "model":    model,
        "messages": _messages(prompt, system),
        "stream":   True,
    }
    try:
        resp = requests.post(
            _CHAT_ENDPOINT, json=payload, stream=True, timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise RuntimeError(_connection_hint(e)) from e
    resp.raise_for_status()
    # Ollama streams newline-delimited JSON. Each line is a chat-chunk event
    # whose `message.content` carries the new token text; `done: true` terminates.
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        chunk = event.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if event.get("done"):
            break


if __name__ == "__main__":
    print("--- generate (non-streaming) ---")
    print(generate(
        "What are common side effects of metformin?",
        system="You are a concise medical educator.",
    ))

    print("\n--- generate_stream ---")
    for chunk in generate_stream(
        "Explain in two sentences how aspirin works.",
        system="You are a concise medical educator.",
    ):
        print(chunk, end="", flush=True)
    print()
