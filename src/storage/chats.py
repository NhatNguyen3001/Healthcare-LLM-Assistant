"""Chat persistence - one JSON file per user under `data/chats/<slug>.json`.

Schema:

    {
      "user": "<slug>",
      "chats": [
        {
          "id":      "<chat-id>",
          "title":   "<editable title>",
          "created": "<iso-ts>",
          "updated": "<iso-ts>",
          "turns":   [ { "input", "panels", "ts" }, ... ]
        },
        ...
      ]
    }

The login screen is name-only — slugify the name to derive the file. No
password, no security boundary. Atomic writes via tempfile + os.replace.
"""
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT

CHATS_DIR = PROJECT_ROOT / "data" / "chats"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Lowercase, ASCII-only, single-dash separator. Empty / all-symbol input
    falls back to 'user' so we never produce an empty filename."""
    s = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return s or "user"


def _user_path(slug: str) -> Path:
    return CHATS_DIR / f"{slug}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_chat_id() -> str:
    return f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def load_user(slug: str) -> dict:
    """Load a user's chat file, or return an empty shell if it doesn't exist."""
    path = _user_path(slug)
    if not path.exists():
        return {"user": slug, "chats": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt file — start fresh rather than crash the UI.
        return {"user": slug, "chats": []}


def save_user(slug: str, data: dict) -> None:
    """Atomic write to data/chats/<slug>.json."""
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    path = _user_path(slug)
    fd, tmp = tempfile.mkstemp(dir=CHATS_DIR, prefix=f".{slug}.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_chat(data: dict, chat_id: str) -> dict | None:
    for c in data.get("chats", []):
        if c.get("id") == chat_id:
            return c
    return None


def new_chat(data: dict, title: str = "New chat") -> dict:
    """Create a new chat, prepend to the user's list, return it."""
    chat = {
        "id":      _new_chat_id(),
        "title":   title or "New chat",
        "created": _now(),
        "updated": _now(),
        "turns":   [],
    }
    data.setdefault("chats", []).insert(0, chat)
    return chat


def rename_chat(data: dict, chat_id: str, new_title: str) -> bool:
    chat = get_chat(data, chat_id)
    if chat is None:
        return False
    chat["title"] = (new_title or "").strip() or "Untitled"
    chat["updated"] = _now()
    return True


def delete_chat(data: dict, chat_id: str) -> bool:
    chats = data.get("chats", [])
    before = len(chats)
    data["chats"] = [c for c in chats if c.get("id") != chat_id]
    return len(data["chats"]) < before


def reset_chat(data: dict, chat_id: str) -> bool:
    """Clear all turns but keep the chat entry."""
    chat = get_chat(data, chat_id)
    if chat is None:
        return False
    chat["turns"] = []
    chat["updated"] = _now()
    return True


def delete_user(slug: str) -> bool:
    """Delete the user's chat file. Returns True if a file was removed.
    Irreversible — the caller is responsible for confirming with the user."""
    path = _user_path(slug)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def add_turn(data: dict, chat_id: str, turn: dict) -> bool:
    """Append a turn dict to the chat. Caller is responsible for the turn
    shape — typically {"input": str, "panels": {...}, "ts": iso-ts}."""
    chat = get_chat(data, chat_id)
    if chat is None:
        return False
    turn.setdefault("ts", _now())
    chat.setdefault("turns", []).append(turn)
    chat["updated"] = _now()
    # Auto-title from first user input if still on default.
    if chat["title"] in ("New chat", "Untitled") and turn.get("input"):
        chat["title"] = turn["input"][:48].strip() or chat["title"]
    return True


if __name__ == "__main__":
    slug = slugify("Nhat Demo!")
    print(f"slug: {slug!r}")
    data = load_user(slug)
    print(f"loaded: {len(data['chats'])} chats")
    chat = new_chat(data, "Smoke test")
    add_turn(data, chat["id"], {"input": "Patient with chest pain", "panels": {}})
    rename_chat(data, chat["id"], "Chest pain consult")
    save_user(slug, data)
    print(f"saved -> {_user_path(slug)}")
    delete_chat(data, chat["id"])
    save_user(slug, data)
    print(f"after delete: {len(data['chats'])} chats")
