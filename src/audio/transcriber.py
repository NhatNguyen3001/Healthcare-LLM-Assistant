"""Audio transcription — thin wrapper around OpenAI's transcription API.

Always uses `gpt-4o-transcribe`. Accepts
either a file path or raw audio bytes — the latter is what
`streamlit-audiorecorder` produces from a live recording.
"""
import io
from pathlib import Path
from typing import Union

from openai import OpenAI

from src.utils.config import OPENAI_API_KEY, TRANSCRIPTION_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

AudioInput = Union[str, Path, bytes, io.IOBase]


def transcribe(audio: AudioInput, *, filename: str = "audio.wav") -> str:
    """Transcribe audio to plain text via `gpt-4o-transcribe`.

    Args:
        audio: filesystem path, raw bytes (e.g. from streamlit-audiorecorder),
            or any binary file-like object.
        filename: only used when `audio` is bytes — the OpenAI SDK infers the
            audio format from this name, so the extension must match the data
            (.wav, .mp3, .m4a, .webm, …).
    """
    if isinstance(audio, (str, Path)):
        with open(audio, "rb") as f:
            resp = _client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=f,
            )
    elif isinstance(audio, bytes):
        buf = io.BytesIO(audio)
        buf.name = filename
        resp = _client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=buf,
        )
    else:
        # file-like object — caller is responsible for .name + binary mode
        resp = _client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio,
        )

    return resp.text.strip()


if __name__ == "__main__":
    import sys

    from src.utils.config import PROJECT_ROOT

    sample_dir = PROJECT_ROOT / "data" / "sample_audio"
    samples = sorted(sample_dir.glob("*")) if sample_dir.exists() else []
    samples = [p for p in samples if p.is_file()]

    if not samples:
        print(f"No sample audio in {sample_dir}. Drop a .wav/.mp3 there to smoke-test.")
        sys.exit(0)

    target = samples[0]
    print(f"INPUT:  {target.name}")
    print(f"OUTPUT: {transcribe(target)}")
