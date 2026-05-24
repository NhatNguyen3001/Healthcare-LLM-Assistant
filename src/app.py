"""Streamlit UI — healthcare NLP demo entry point.

Flow: login (name only) -> sidebar (chats + input + sample queries + Run) ->
main column (live pipeline trace, 8 panels rendered top-to-bottom).

Each completed turn persists to data/chats/<slug>.json with the full
8-panel output. The last 3 turns drive contextual query rewriting before
ChromaDB retrieval and are injected into the final GPT-5.5 prompt.

Run from the project root: `streamlit run src/app.py`
"""
import sys
from pathlib import Path

# This file lives inside the `src` package; running it as a script means
# `src/` is on sys.path but its parent (the project root) is not — so the
# absolute imports below (`from src.storage import ...`) would fail. Insert
# the project root first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Quiet the noisy startup stack BEFORE any transitive ML import. Order
# matters: TF_CPP_MIN_LOG_LEVEL must be set in the environment before
# `import tensorflow` (which transformers/sentence-transformers probe for
# even though we never run TF compute). The warnings filters must beat
# torch's _pytree register_constant() warning and spaCy's [W095] model
# version mismatch — both fire at module/model load time, not at use.
import os
import shutil
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 3 = hide INFO/WARNING/ERROR; only FATAL remains
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # silence the oneDNN custom-ops INFO line at the source

warnings.filterwarnings(
    "ignore",
    message=r"Couldn't find (ffmpeg|ffprobe|avconv|avprobe)",
)
warnings.filterwarnings(
    "ignore",
    message=r"<enum '[A-Za-z_]+'> is an Enum subclass and is now natively supported",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"\[W095\] Model '.+'.+may not be 100% compatible",
    category=UserWarning,
)

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()  # idempotent; downloads on first call only
    _FFMPEG_EXE = shutil.which("ffmpeg")
    _FFPROBE_EXE = shutil.which("ffprobe")
    if not _FFMPEG_EXE or not _FFPROBE_EXE:
        raise FileNotFoundError(
            "static_ffmpeg.add_paths() did not put ffmpeg / ffprobe on PATH"
        )
    from pydub import AudioSegment
    AudioSegment.converter = _FFMPEG_EXE
    AudioSegment.ffmpeg = _FFMPEG_EXE
    AudioSegment.ffprobe = _FFPROBE_EXE
    # Surfaced in the audio expander so users can see whether the recorder
    # backend is alive before they ever click record.
    _FFMPEG_BACKEND = "static-ffmpeg (bundled)"
except Exception as _ffmpeg_import_err:
    # static-ffmpeg not installed / network-blocked on first download.
    # Recording will fail until a system ffmpeg + ffprobe is on PATH.
    _FFMPEG_BACKEND = f"unavailable ({type(_ffmpeg_import_err).__name__})"

import html
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from spacy import displacy

from src.storage import chats as chat_store
from src.utils.config import CHROMA_K, TRANSCRIPTION_MODEL


# ------------------------------------------------------------- Page config

# Logo lives at project-root /assets/logo.png. Falls back to an emoji until
# the file is dropped in, so a missing asset never breaks startup.
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
st.set_page_config(
    page_title="Healthcare NLP Assistant",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🩺",
    layout="wide",
)


# Sidebar selector keys -> display labels. Single source of truth so the
# selectbox and the "via X" recommendation chip never drift apart.
MODEL_LABELS: dict[str, str] = {
    "cloud":         "GPT-5.5 (cloud)",
    "local_qwen":    "QLoRA Qwen2.5-1.5B (local)",
    "local_llama32": "QLoRA Llama-3.2-1B (local)",
}


# ------------------------------------------------------------- Cached loaders

@st.cache_resource(show_spinner="Warming up clinical NLP models (one-time)")
def load_pipeline():
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    from src.pipeline import (
        chromadb_store,
        classifier,
        llm,
        massrag,
        ner,
        preprocessing,
        railguard,
        sentiment,
        STO_extracting,
    )
    print(f"[warmup] module imports: {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    pipe = {
        "rg": railguard, "pre": preprocessing, "ner": ner,
        "sent": sentiment, "clf": classifier, "sto": STO_extracting,
        "cdb": chromadb_store, "mr": massrag, "llm": llm,
    }

    # Parallel warmup: sentiment .pkl and classifier .pkl + S-PubMedBert
    # are I/O-heavy (disk reads + weight deserialization) so threads overlap.
    def _warm_sentiment():
        t = time.perf_counter()
        try:
            sentiment.predict("warmup")
        except Exception as e:
            print(f"[warmup] sentiment skipped: {e}", file=sys.stderr)
        return f"sentiment .pkl: {time.perf_counter() - t:.2f}s"

    def _warm_classifier():
        t = time.perf_counter()
        try:
            classifier.predict("warmup")
        except Exception as e:
            print(f"[warmup] classifier skipped: {e}", file=sys.stderr)
        return f"classifier .pkl + S-PubMedBert: {time.perf_counter() - t:.2f}s"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(fn): fn.__name__ for fn in [_warm_sentiment, _warm_classifier]}
        for fut in as_completed(futs):
            print(f"[warmup] {fut.result()}", file=sys.stderr)

    print(f"[warmup] TOTAL: {time.perf_counter() - t_total:.2f}s", file=sys.stderr)
    return pipe


@st.cache_resource(show_spinner="Loading audio module...")
def load_audio():
    from src.audio import transcriber
    return transcriber


# ------------------------------------------------------------- Constants

SAMPLE_QUERIES = [
    ("Headache + nausea",
     "I've had a bad headache and felt nauseous for two days. "
     "What could be wrong?"),
    ("Ibuprofen safety",
     "I take medication for high blood pressure. Is it safe to take "
     "ibuprofen for back pain?"),
    # STO-rich: sx (sinus infection, stomach pain, diarrhea), tx (amoxicillin),
    # ox (improved on day 3 then new GI side effects). Tests the S/T/O panel.
    ("Antibiotic side effects",
     "I had a sinus infection last week. The doctor gave me amoxicillin and "
     "I felt much better after three days, but now I have stomach pain and "
     "diarrhea. Is this normal?"),
    ("Heartburn at night",
     "I often get heartburn at night after dinner. "
     "What can I do to prevent it?"),
    # STO-rich: sx (sprain, swelling, pain on walking), tx (ice, rest),
    # ox (swelling improving, still painful). Tests the S/T/O panel.
    ("Sprained ankle",
     "I sprained my ankle playing soccer three days ago. I've been using "
     "ice and resting it, and the swelling is going down, but it still "
     "hurts to walk. Should I see a doctor?"),
]

DISCLAIMER = (
    ":warning: **Clinical decision support demo only - not medical advice.** "
    "Always consult a licensed healthcare professional."
)

SOURCE_COLORS = {
    "medicaltranscriptions": "#ffadad",
    "bioasq":                "#ffd6a5",
    "medquad":               "#fdffb6",
    "drugbank":              "#caffbf",
    "medrag-textbooks":      "#9bf6ff",
    "medtext":               "#bdb2ff",
}


# ------------------------------------------------------------- Helpers

def diff_html(before: str, after: str):
    return (
        "<div style='display:flex;gap:1em'>"
        f"<div style='flex:1'><b>Before</b>"
        f"<pre style='white-space:pre-wrap'>{html.escape(before)}</pre></div>"
        f"<div style='flex:1'><b>After</b>"
        f"<pre style='white-space:pre-wrap'>{html.escape(after)}</pre></div>"
        "</div>"
    )


def ner_html(doc):
    return displacy.render(
        doc, style="ent",
        options={"colors": {"DISEASE": "#ffadad", "CHEMICAL": "#a0c4ff"}},
        page=False, minify=True,
    )


def source_badge(src: str):
    bg = SOURCE_COLORS.get(src, "#dddddd")
    return (
        f"<span style='background:{bg};padding:2px 8px;border-radius:8px;"
        f"font-size:0.8em'>{html.escape(src)}</span>"
    )


def last_turns(chat: dict, n: int = 3) -> list[dict]:
    return [
        {"input": t.get("input", ""),
         "recommendation": (t.get("panels", {}).get("recommendation") or "")}
        for t in chat.get("turns", [])[-n:]
    ]


# ------------------------------------------------------------- Login

def inject_css():
    """Light visual polish, rounded buttons, soft cards, sidebar tone.

    The `:has()` selectors below modern-CSS (Baseline 2023) and key the layout
    off invisible marker divs we plant in specific places (e.g. the active
    chat row, the destructive Delete button, the sample-query chip row).
    Streamlit doesn't expose widget keys as CSS hooks, so markers + :has() is
    the cleanest way to scope styles to one widget instead of all of them.
    """
    st.markdown(
        """
        <style>
        /* === Buttons: rounded, subtle lift on hover ===================== */
        [data-testid="stButton"] > button {
            border-radius: 10px;
            transition: transform 0.12s ease, box-shadow 0.12s ease,
                        background 0.12s ease, border-color 0.12s ease;
        }
        [data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(28, 25, 23, 0.08);
        }
        [data-testid="stPopover"] > button { border-radius: 10px; }
        [data-testid="stExpander"] details { border-radius: 12px; }
        [data-testid="stSidebar"] { background: #e6e3de; }

        /* === Bordered containers (live panels + history + active chat) == */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #d6d2cb !important;
            background: #faf8f5;
        }

        /* Active chat row: primary-tinted accent + soft gradient */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.active-chat-marker) {
            border: 1px solid #475569 !important;
            border-left: 3px solid #475569 !important;
            background: linear-gradient(180deg, #f5f2ed 0%, #ebe7e0 100%);
            box-shadow: 0 1px 4px rgba(71, 85, 105, 0.10);
        }

        /* === Typography hierarchy ====================================== */
        h3 {
            color: #1c1917;
            letter-spacing: 0.01em;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #d6d2cb;
        }
        h4 {
            color: #475569;
            letter-spacing: 0.005em;
            margin-bottom: 0.5rem !important;
        }

        /* Sidebar section headers ('Chats', 'Model') as eyebrow caps */
        [data-testid="stSidebar"] h3 {
            border-bottom: none;
            font-size: 0.78rem;
            padding-bottom: 0;
            text-transform: uppercase;
            font-weight: 700;
            color: #57534e;
            letter-spacing: 0.10em;
            margin-top: 1.2rem !important;
            margin-bottom: 0.4rem !important;
        }

        /* === Sample query chips: pill-shaped, smaller ==================
           Adjacent-sibling so only the row IMMEDIATELY after the anchor
           gets pill styling. Earlier descendant-form selector leaked into
           every later horizontal block (including the audio expander's
           Send/Re-record row), making the primary Send button white. */
        [data-testid="stElementContainer"]:has(.sample-chips-anchor)
            + [data-testid="stElementContainer"]
            [data-testid="stHorizontalBlock"]
            [data-testid="stButton"] > button {
            border-radius: 999px !important;
            font-size: 0.85em;
            padding: 0.35em 1.1em;
            background: #faf8f5;
            border: 1px solid #d6d2cb;
            font-weight: 500;
        }
        [data-testid="stElementContainer"]:has(.sample-chips-anchor)
            + [data-testid="stElementContainer"]
            [data-testid="stHorizontalBlock"]
            [data-testid="stButton"] > button:hover {
            background: #f0ede8;
            border-color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login_screen():
    inject_css()
    st.title("Healthcare NLP Assistant")
    st.caption("Name-only login. Same name = same chat history.")
    st.info(DISCLAIMER)
    with st.form("login"):
        name = st.text_input("Your name", placeholder="e.g. Davis")
        ok = st.form_submit_button("Continue", type="primary")
    if ok and name.strip():
        slug = chat_store.slugify(name)
        st.session_state.user_slug = slug
        st.session_state.user_name = name.strip()
        st.session_state.user_data = chat_store.load_user(slug)
        if not st.session_state.user_data["chats"]:
            chat_store.new_chat(st.session_state.user_data, "New chat")
            chat_store.save_user(slug, st.session_state.user_data)
        st.session_state.current_chat_id = (
            st.session_state.user_data["chats"][0]["id"]
        )
        st.rerun()


# Sidebar

def sidebar():
    s = st.session_state
    data = s.user_data

    with st.sidebar:
        st.markdown(f":bust_in_silhouette: **{s.user_name}**")
        if st.button("Log out", use_container_width=True):
            for k in (
                "user_slug", "user_name", "user_data", "current_chat_id",
                "pending_run", "confirming_delete_account",
                "audio_buffer", "audio_widget_nonce",
            ):
                s.pop(k, None)
            st.rerun()

        st.divider()
        st.markdown("### Model")
        # Only the final-answer generator switches. MASS-RAG filter agents,
        # rewrite_query, and judge stay on GPT-5.4-mini / GPT-5.4 regardless
        # (see llm.py router). Persisted across turns via session_state.
        st.selectbox(
            "Generator",
            options=["cloud", "local_qwen", "local_llama32"],
            format_func=lambda c: MODEL_LABELS[c],
            key="model_choice",
            label_visibility="collapsed",
            help=(
                "cloud = OpenAI GPT-5.5 via the Responses API. "
                "local_qwen / local_llama32 = your QLoRA fine-tunes served by "
                "one Ollama daemon at 127.0.0.1:11434."
            ),
        )

        st.divider()
        st.markdown("### Chats")

        if st.button("➕ New chat", use_container_width=True):
            chat = chat_store.new_chat(data, "New chat")
            chat_store.save_user(s.user_slug, data)
            s.current_chat_id = chat["id"]
            st.rerun()

        for chat in list(data.get("chats", [])):
            active = chat["id"] == s.current_chat_id
            label = ("▸ " if active else "  ") + chat["title"]
            row = st.container(border=True) if active else st.container()
            with row:
                col_label, col_more = st.columns([6, 1])
                if col_label.button(
                    label, key=f"sel-{chat['id']}", use_container_width=True,
                ):
                    s.current_chat_id = chat["id"]
                    st.rerun()

                # Popover holds the rename input + delete. Outside-click
                # dismisses, so no explicit Cancel button is needed.
                with col_more.popover("", use_container_width=True):
                    new_title = st.text_input(
                        "Title",
                        value=chat["title"],
                        key=f"ren-input-{chat['id']}",
                        label_visibility="collapsed",
                    )
                    save_col, del_col = st.columns(2)
                    if save_col.button(
                        "Save", icon=":material/check:",
                        key=f"ren-save-{chat['id']}",
                        type="primary", use_container_width=True,
                    ):
                        chat_store.rename_chat(data, chat["id"], new_title)
                        chat_store.save_user(s.user_slug, data)
                        st.rerun()
                    if del_col.button(
                        "Delete",
                        key=f"act-del-{chat['id']}",
                        use_container_width=True,
                    ):
                        chat_store.delete_chat(data, chat["id"])
                        if not data["chats"]:
                            chat_store.new_chat(data, "New chat")
                        chat_store.save_user(s.user_slug, data)
                        s.current_chat_id = data["chats"][0]["id"]
                        st.rerun()

                # Invisible marker keys the bordered container's CSS via
                # :has(.active-chat-marker) so only THIS row gets the accent.
                if active:
                    st.markdown(
                        '<span class="active-chat-marker"></span>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        if st.button("↻ Reset current chat", use_container_width=True):
            chat_store.reset_chat(data, s.current_chat_id)
            chat_store.save_user(s.user_slug, data)
            st.rerun()

        st.divider()
        if s.get("confirming_delete_account"):
            st.warning(
                f"Delete account **{s.user_name}** and ALL its chats? "
                "This cannot be undone."
            )
            c1, c2 = st.columns(2)
            if c1.button("Yes, delete", type="primary", use_container_width=True):
                chat_store.delete_user(s.user_slug)
                for k in list(s.keys()):
                    s.pop(k, None)
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                s.pop("confirming_delete_account", None)
                st.rerun()
        else:
            if st.button("⚠️ Delete account", use_container_width=True):
                s.confirming_delete_account = True
                st.rerun()


# Pipeline

def safe(label: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{label} failed: {type(e).__name__}: {e}"


def run_pipeline(user_input: str, audio_bytes, audio_name, chat: dict) -> dict:
    """Execute the full pipeline; render each panel inline as it completes;
    return a panels dict for persistence. Per-panel try/except — failures show
    a warning and the pipeline continues."""
    pipe = load_pipeline()
    panels: dict = {}

    # Pre-allocate placeholders for every panel up-front. Each `st.empty()`
    # immediately replaces the prior turn's DOM at its script-position with
    # an empty container, so old panels don't sit faded while this turn's
    # blocking steps (STO LLM call, MASS-RAG) are still running. Each panel
    # later fills its placeholder via `with ph_X.container(border=True):`.
    ph_input      = st.empty()
    ph_rg1        = st.empty()
    ph_ner        = st.empty()
    ph_sto        = st.empty()
    ph_mr         = st.empty()
    ph_ret        = st.empty()
    ph_rec        = st.empty()

    # 0) Input panel: transcript card for audio, typed-input card for text.
    # Always present so the live trace's first card matches the history
    # replay's `#### 📝 Input` / `#### 🎤 Transcript` cards.
    with ph_input.container(border=True):
        if audio_bytes is not None:
            try:
                t_tr = time.perf_counter()
                user_input = load_audio().transcribe(
                    audio_bytes, filename=audio_name or "audio.wav",
                )
                tr_elapsed = time.perf_counter() - t_tr
                # Mirror the Recommendation panel: tail in the header for UI,
                # a one-line stderr log for terminal inspection. Persist both
                # to panels so render_history can replay the same metadata.
                tail = f" *(via {TRANSCRIPTION_MODEL} · {tr_elapsed:.2f}s)*"
                st.markdown(f"#### 🎤 Transcript{tail}")
                st.write(user_input)
                print(
                    f"[pipeline] transcribe done "
                    f"({TRANSCRIPTION_MODEL}, {tr_elapsed:.2f}s, "
                    f"chars={len(user_input)})",
                    file=sys.stderr,
                )
                panels["transcript"]        = user_input
                panels["transcript_model"]  = TRANSCRIPTION_MODEL
                panels["transcript_time_s"] = tr_elapsed
            except Exception as e:
                st.markdown("#### 🎤 Transcript")
                st.warning(f"Transcription failed: {e}")
                panels["transcript_error"] = str(e)
                return panels
        else:
            st.markdown("#### 📝 Input")
            st.write(user_input)

    panels["input"] = user_input

    # 1) Railguard 1
    with ph_rg1.container(border=True):
        st.markdown("#### 🛡️ Railguard: PII stripped / generalised")
        rg1, err = safe("Railguard 1", pipe["rg"].railguard, user_input)
        if err:
            st.warning(err)
            rg1 = user_input
            panels["rg1_error"] = err
        st.markdown(diff_html(user_input, rg1), unsafe_allow_html=True)
        panels["rg1"] = rg1

    # 2) Preprocessing (no panel — feeds the rest)
    pre, err = safe("Preprocessing", pipe["pre"].preprocess, rg1)
    if err:
        pre = rg1
        panels["preprocessing_error"] = err
    panels["preprocessed"] = pre

    # 3) Parallel block: NER, sentiment, classifier, S/T/O extraction
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_ner  = ex.submit(lambda: (pipe["ner"].to_doc(pre), pipe["ner"].extract(pre)))
        f_sent = ex.submit(pipe["sent"].predict, pre)
        f_clf  = ex.submit(pipe["clf"].predict, pre)
        f_sto  = ex.submit(pipe["sto"].extract, pre)

        ner_pair, ner_err   = safe("NER",        f_ner.result)
        sent_res, sent_err  = safe("Sentiment",  f_sent.result)
        clf_res,  clf_err   = safe("Classifier", f_clf.result)
        sto_res,  sto_err   = safe("Extraction", f_sto.result)

    # 3a) NER panel + display-only badges
    with ph_ner.container(border=True):
        st.markdown("#### 🩺 NER + Signal badges")
        b1, b2 = st.columns(2)
        if sent_res:
            b1.metric("Sentiment", sent_res["label"], f"{sent_res['score']:.2f}")
        else:
            b1.warning(sent_err or "Sentiment unavailable")
        if clf_res:
            b2.metric("Predicted specialty",
                      clf_res["label"], f"{clf_res['score']:.2f}")
        else:
            b2.warning(clf_err or "Classifier unavailable")
        if ner_pair:
            doc, ents = ner_pair
            ner_html_str = ner_html(doc)
            st.markdown(ner_html_str, unsafe_allow_html=True)
            panels["ner"] = ents
            panels["ner_html"] = ner_html_str
        else:
            st.warning(ner_err)
            panels["ner_error"] = ner_err
        panels["sentiment"]  = sent_res or {}
        panels["classifier"] = clf_res or {}

    # 3b) Extracted findings panel
    with ph_sto.container(border=True):
        st.markdown("#### 📋 Extracted findings (Symptoms / Treatments / Outcomes)")
        if sto_err:
            st.warning(sto_err)
            panels["extraction_error"] = sto_err
        else:
            sx = sto_res.get("symptoms")   or []
            tx = sto_res.get("treatments") or []
            ox = sto_res.get("outcomes")   or []
            if sx or tx or ox:
                cs, ct, co = st.columns(3)
                cs.markdown("**Symptoms**\n"   + "\n".join(f"- {x}" for x in (sx or ["—"])))
                ct.markdown("**Treatments**\n" + "\n".join(f"- {x}" for x in (tx or ["—"])))
                co.markdown("**Outcomes**\n"   + "\n".join(f"- {x}" for x in (ox or ["—"])))
            elif panels.get("ner"):
                st.info(
                    "Informational query detected — no clinical findings to "
                    "extract. NER entities above are used for retrieval."
                )
            else:
                st.write("No clinical findings extracted.")
            panels["extraction"] = sto_res

    # 4) Query rewriting (uses last 3 turns) + retrieval + MASS-RAG
    history3 = last_turns(chat, 3)
    ner_terms = " ".join(e["text"] for e in (panels.get("ner") or []))
    sx_terms  = " ".join((sto_res or {}).get("symptoms", []))
    base_q    = " ".join(filter(None, [pre, ner_terms, sx_terms])).strip()
    print(f"[pipeline] rewrite_query start (history_turns={len(history3)}, base_q_len={len(base_q)})", file=sys.stderr)
    rewritten, _ = safe("Query rewriting", pipe["llm"].rewrite_query, base_q, history3)
    rewritten = rewritten or base_q
    print(f"[pipeline] rewrite_query done (rewritten_len={len(rewritten)})", file=sys.stderr)
    panels["rewritten_query"] = rewritten

    massrag_result = None
    with ph_mr.container(border=True):
        st.markdown("#### 🤖 MASS-RAG")
        with st.status("Retrieving + running agents...", expanded=True) as status:
            st.write(f"🔎 Search query (rewritten): `{rewritten[:200]}`")
            try:
                print(f"[pipeline] chromadb query start", file=sys.stderr)
                t_retrieval = time.perf_counter()
                hits = pipe["cdb"].query(rewritten, k=CHROMA_K)
                t_retrieval = time.perf_counter() - t_retrieval
                dists_preview = ", ".join(f"{h.get('distance', 0):.3f}" for h in hits[:7])
                print(f"[pipeline] chromadb query done ({len(hits)} hits, {t_retrieval:.2f}s)", file=sys.stderr)
                st.write(f"📚 Retrieved {len(hits)} docs from ChromaDB in `{t_retrieval:.2f}s` — distances: `{dists_preview}`")
                status.update(label="Summarizer / Extractor / Reasoner (parallel)...")
                print(f"[pipeline] mass-rag start", file=sys.stderr)
                t_massrag = time.perf_counter()
                massrag_result = pipe["mr"].run_sync(rewritten, hits)
                t_massrag = time.perf_counter() - t_massrag
                print(f"[pipeline] mass-rag done (fallback={massrag_result['fallback']}, {t_massrag:.2f}s)", file=sys.stderr)
                if massrag_result["fallback"]:
                    st.write(f"⚠️ No docs passed similarity threshold - falling back to web search. (`{t_massrag:.2f}s`)")
                    status.update(label="Fallback to web search", state="complete")
                else:
                    st.write(f"✓ Synthesis complete on {len(massrag_result['retained_hits'])} retained docs in `{t_massrag:.2f}s` (3 parallel filters + synthesis)")
                    status.update(label=f"MASS-RAG complete ({t_retrieval + t_massrag:.1f}s)", state="complete")
                panels["massrag"] = {
                    "fallback": massrag_result["fallback"],
                    "evidence": massrag_result.get("evidence", ""),
                    "agents":   massrag_result.get("agents", {}),
                    "retained": [
                        {"id": h["id"], "distance": h["distance"],
                         "metadata": h["metadata"], "document": h["document"]}
                        for h in massrag_result["retained_hits"]
                    ],
                }
            except Exception as e:
                st.warning(f"MASS-RAG failed: {e}")
                panels["massrag_error"] = str(e)
                status.update(label="MASS-RAG failed", state="error")

        if massrag_result and not massrag_result["fallback"]:
            with st.expander("Agent outputs"):
                for name, out in massrag_result["agents"].items():
                    st.markdown(f"**{name.title()}**")
                    st.write(out)
            with st.expander("Synthesised evidence"):
                st.write(massrag_result["evidence"])

    # 5) Retrieved cases panel (after MASS-RAG, per user request)
    with ph_ret.container(border=True):
        st.markdown("#### 📄 Retrieved cases")
        retained = (panels.get("massrag") or {}).get("retained") or []
        if retained:
            for h in retained:
                src = (h.get("metadata") or {}).get("source", "?")
                with st.container(border=True):
                    st.markdown(
                        f"{source_badge(src)} &nbsp; `dist={h['distance']:.3f}` "
                        f"&nbsp; `{h['id']}`",
                        unsafe_allow_html=True,
                    )
                    doc = h["document"]
                    st.write(doc[:600] + ("..." if len(doc) > 600 else ""))
        elif massrag_result and massrag_result["fallback"]:
            st.info("No KB matches — recommendation will use web search.")
        else:
            st.write("No retrieval results.")

    # 6) Recommendation — streamed
    model_choice = st.session_state.get("model_choice", "cloud")
    model_display = MODEL_LABELS.get(model_choice, model_choice)
    with ph_rec.container(border=True):
        st.markdown(
            "<div style='background:#e2dfd9;border-left:4px solid #475569;"
            "padding:0.6rem 0.9rem;border-radius:6px;margin-bottom:0.4rem;"
            "font-size:1.05rem;font-weight:600'>"
            "💡 Clinical recommendation "
            f"<span style='font-size:0.85rem;font-weight:400;opacity:0.75'>"
            f"· via {html.escape(model_display)}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if massrag_result is None:
            st.warning("Skipping recommendation — MASS-RAG failed.")
            panels["recommendation_error"] = "no massrag"
            return panels

        system_prompt = (
            "You are a clinical decision support assistant. Never provide a "
            "definitive diagnosis. Always recommend professional consultation "
            "for serious symptoms. Think step by step before giving your "
            "recommendation."
        )
        if massrag_result["fallback"]:
            prompt = massrag_result["fallback_prompt"]
        else:
            history_str = "\n\n".join(
                f"Previous user input:\n{t['input']}\n"
                f"Prior recommendation:\n{t['recommendation']}"
                for t in history3
            )
            ner_str = ", ".join(f"{e['label']}={e['text']}" for e in (panels.get("ner") or []))
            sx = ", ".join((sto_res or {}).get("symptoms", []))
            tx = ", ".join((sto_res or {}).get("treatments", []))
            ox = ", ".join((sto_res or {}).get("outcomes", []))
            few_shot = (
                "Example.\n"
                "Patient: 55yo with chest pain and SOB.\n"
                "Recommendation: Given chest pain and SOB, urgent cardiac "
                "evaluation is indicated to rule out acute coronary syndrome. "
                "Call emergency services. Do not delay.\n"
            )
            prompt = (
                f"{few_shot}\n"
                + (f"Conversation so far:\n{history_str}\n\n" if history_str else "")
                + f"Current case (railguarded):\n{rg1}\n\n"
                f"Clinical entities: {ner_str or '—'}\n"
                f"Symptoms: {sx or '—'}\n"
                f"Treatments mentioned: {tx or '—'}\n"
                f"Outcomes / response: {ox or '—'}\n\n"
                f"Synthesised evidence:\n{massrag_result['evidence']}\n\n"
                "Think step by step and provide a single coherent clinical "
                "recommendation. Cite document ids from the evidence where "
                "relevant."
            )

        try:
            print(f"[pipeline] generate_stream start ({model_display})", file=sys.stderr)
            t_inference = time.perf_counter()
            final_text = st.write_stream(
                pipe["llm"].generate_stream(
                    prompt, system=system_prompt, enable_web_search=True,
                    model_choice=model_choice,
                )
            )
            t_inference = time.perf_counter() - t_inference
            print(
                f"[pipeline] generate_stream done "
                f"({model_display}, {t_inference:.2f}s, "
                f"chars={len(final_text or '')})",
                file=sys.stderr,
            )
            st.caption(f"⏱ Generated by **{model_display}** in `{t_inference:.2f}s`")
            panels["recommendation"]    = final_text
            panels["model_used"]        = model_display
            panels["inference_time_s"]  = round(t_inference, 3)
        except Exception as e:
            st.warning(f"Generation failed: {e}")
            panels["recommendation_error"] = str(e)
            return panels

    return panels


# ------------------------------------------------------------- Main view

def _turn_label(i: int, t: dict):
    from datetime import datetime
    snippet = (t.get("input") or "").strip().splitlines()
    snippet = snippet[0][:60] if snippet else f"Turn {i+1}"
    icon = "🎤" if t.get("panels", {}).get("transcript") else "⌨"
    ts = t.get("ts", "")
    try:
        ts_str = datetime.fromisoformat(ts).strftime("%H:%M")
    except (ValueError, TypeError):
        ts_str = ""
    parts = [f"Turn {i+1}", icon]
    if ts_str:
        parts.append(ts_str)
    parts.append(snippet)
    return " · ".join(parts)


def render_history(chat: dict, expand_last: bool = True):
    turns = chat.get("turns") or []
    if not turns:
        return
    st.markdown("#### Conversation history")
    last_idx = len(turns) - 1
    for i, t in enumerate(turns):
        with st.expander(_turn_label(i, t), expanded=(expand_last and i == last_idx)):
            p = t.get("panels", {})

            with st.container(border=True):
                # Audio turns get the same model + time tail in the header
                # as the live Transcript panel; typed turns stay as "Input".
                if p.get("transcript_model"):
                    tr_model = p["transcript_model"]
                    tr_time  = p.get("transcript_time_s")
                    tail_bits = [f"via {tr_model}"]
                    if tr_time is not None:
                        tail_bits.append(f"{tr_time:.2f}s")
                    st.markdown(
                        f"#### 🎤 Transcript *({' · '.join(tail_bits)})*"
                    )
                else:
                    st.markdown("#### 📝 Input")
                st.write(t.get("input", ""))

            sent = p.get("sentiment") or {}
            clf  = p.get("classifier") or {}
            if sent or clf or p.get("ner_html"):
                with st.container(border=True):
                    st.markdown("#### 🩺 NER + Signal badges")
                    if sent or clf:
                        bs, bc = st.columns(2)
                        if sent.get("label"):
                            bs.metric("Sentiment", sent["label"], f"{sent.get('score', 0):.2f}")
                        if clf.get("label"):
                            bc.metric("Specialty (display only)", clf["label"], f"{clf.get('score', 0):.2f}")
                    if p.get("ner_html"):
                        st.markdown(p["ner_html"], unsafe_allow_html=True)

            ext = p.get("extraction") or {}
            sx = ext.get("symptoms")   or []
            tx = ext.get("treatments") or []
            ox = ext.get("outcomes")   or []
            if sx or tx or ox:
                with st.container(border=True):
                    st.markdown("#### 📋 Findings (Symptoms / Treatments / Outcomes)")
                    cs, ct, co = st.columns(3)
                    cs.markdown("**Symptoms**\n"   + "\n".join(f"- {x}" for x in (sx or ["—"])))
                    ct.markdown("**Treatments**\n" + "\n".join(f"- {x}" for x in (tx or ["—"])))
                    co.markdown("**Outcomes**\n"   + "\n".join(f"- {x}" for x in (ox or ["—"])))
            elif p.get("ner_html"):
                st.caption("No structured findings — informational query.")

            retained = (p.get("massrag") or {}).get("retained") or []
            if retained:
                with st.container(border=True):
                    st.markdown("#### 📄 Retrieved cases")
                    for r in retained:
                        src = (r.get("metadata") or {}).get("source", "?")
                        st.markdown(
                            f"- {source_badge(src)} &nbsp; `{r['id']}` "
                            f"(dist={r['distance']:.3f})",
                            unsafe_allow_html=True,
                        )

            if p.get("recommendation"):
                with st.container(border=True):
                    model = p.get("model_used")
                    inf   = p.get("inference_time_s")
                    tail_bits = []
                    if model:
                        tail_bits.append(f"via {model}")
                    if inf is not None:
                        tail_bits.append(f"{inf:.2f}s")
                    tail = f" *({' · '.join(tail_bits)})*" if tail_bits else ""
                    st.markdown(f"#### 💬 Recommendation{tail}")
                    st.write(p["recommendation"])


def _render_input_row(s):
    """Sample-query chips + audio expander. Rendered just above the pinned
    `st.chat_input`. Each control sets `s.pending_run` and triggers a rerun
    so the pipeline executes on the NEXT script run."""
    # Marker scopes the pill-shape CSS to JUST this chip row, not every
    # button on the page. CSS hooks via :has(.sample-chips-anchor).
    st.markdown(
        '<span class="sample-chips-anchor"></span>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(SAMPLE_QUERIES))
    for col, (label, text) in zip(cols, SAMPLE_QUERIES):
        if col.button(label, key=f"sample-{label}", use_container_width=True):
            s.pending_run = {"text": text, "audio_bytes": None, "audio_name": None}
            st.rerun()

    with st.expander("🎤 Audio input"):
        # Backend status surfaced before the user clicks anything. If the
        # imageio-ffmpeg shim at the top of this file failed at startup,
        # recording will fail at .export(format="wav"). Show a real warning
        # so the user is not staring at a silent failure.
        if _FFMPEG_BACKEND.startswith("unavailable"):
            st.warning(
                f"Audio backend {_FFMPEG_BACKEND}. "
                "Run `pip install -r requirements.txt` and restart Streamlit."
            )
        else:
            st.caption(f"Audio backend: {_FFMPEG_BACKEND}")

        # Bumped after Send / Re-record so the audiorecorder + file_uploader
        # widgets remount fresh. Without it audiorecorder keeps returning the
        # same AudioSegment across reruns and re-buffers it every render.
        s.setdefault("audio_widget_nonce", 0)

        try:
            from audiorecorder import audiorecorder
            rec = audiorecorder(
                "Click to record",
                "Recording... click to stop",
                key=f"audiorec_{s.audio_widget_nonce}",
            )
            if rec is not None and len(rec) > 0 and s.get("audio_buffer") is None:
                wav_bytes = rec.export(format="wav").read()
                s.audio_buffer = {
                    "wav_bytes": wav_bytes,
                    "duration_ms": len(rec),
                    "size_bytes": len(wav_bytes),
                }
        except Exception as e:
            st.caption(f"Live recorder unavailable ({type(e).__name__}) - use upload below.")

        if s.get("audio_buffer"):
            buf = s.audio_buffer
            st.audio(buf["wav_bytes"], format="audio/wav")
            dur_s = buf["duration_ms"] / 1000.0
            size_kb = buf["size_bytes"] / 1024.0
            st.caption(f"Captured {dur_s:.1f}s · {size_kb:.1f} kB · wav")
            send_col, redo_col = st.columns(2)
            if send_col.button(
                "Send for transcription",
                type="primary",
                use_container_width=True,
                key=f"audio_send_{s.audio_widget_nonce}",
            ):
                s.pending_run = {
                    "text": "",
                    "audio_bytes": buf["wav_bytes"],
                    "audio_name": "recording.wav",
                }
                s.audio_buffer = None
                s.audio_widget_nonce += 1
                st.rerun()
            if redo_col.button(
                "Re-record",
                use_container_width=True,
                key=f"audio_redo_{s.audio_widget_nonce}",
            ):
                s.audio_buffer = None
                s.audio_widget_nonce += 1
                st.rerun()

        up = st.file_uploader(
            "Or upload audio",
            type=["wav", "mp3", "m4a", "webm"],
            key=f"audio_upload_{s.audio_widget_nonce}",
        )
        if up is not None:
            s.pending_run = {
                "text": "", "audio_bytes": up.getvalue(), "audio_name": up.name,
            }
            s.audio_widget_nonce += 1
            st.rerun()


def main_view():
    inject_css()
    s = st.session_state
    data = s.user_data
    chat = chat_store.get_chat(data, s.current_chat_id)
    if chat is None:
        if not data["chats"]:
            chat_store.new_chat(data, "New chat")
            chat_store.save_user(s.user_slug, data)
        s.current_chat_id = data["chats"][0]["id"]
        st.rerun()

    # Capture chat_input submission EARLY so the empty-state hero is hidden
    # within the same script run. st.chat_input pins to the viewport bottom
    # visually regardless of where it's called in the script.
    typed = st.chat_input("Ask about a patient case...")
    if typed:
        s.pending_run = {"text": typed, "audio_bytes": None, "audio_name": None}

    # Chat title at the very top of the main area, full width.
    st.markdown(
        f"<h3 style='margin:0.2rem 0 0.5rem 0'>"
        f"{html.escape(chat['title'])}</h3>",
        unsafe_allow_html=True,
    )

    # ChatGPT-style narrow centered column for everything except the pinned
    # chat_input below.
    _, center, _ = st.columns([1, 4, 1])
    with center:
        # When a new turn is about to run live, collapse prior turns so the
        # fresh live-trace panels are the focal point — otherwise the most
        # recent past turn stays auto-expanded above the new run.
        render_history(chat, expand_last=(s.get("pending_run") is None))

        # Empty state hero — only when no turns and no pending run.
        if not chat.get("turns") and not s.get("pending_run"):
            st.markdown(
                "<div style='text-align:center;padding:0.6rem 0;color:#6b7280'>"
                "<div style='font-size:1.15rem;font-weight:600'>"
                "What patient case are you working on?</div>"
                "<div style='font-size:0.9rem;margin-top:0.3rem'>"
                "Type a question below, tap a sample, or attach audio."
                "</div></div>",
                unsafe_allow_html=True,
            )

        # Execute a pending run (set by a sample button, audio, or chat_input
        # on the previous rerun). Panels render live inline here.
        payload = s.pop("pending_run", None)
        if payload:
            if payload.get("text") and payload["text"].strip():
                user_input = payload["text"].strip()
                audio_bytes, audio_name = None, None
            elif payload.get("audio_bytes"):
                user_input = ""
                audio_bytes = payload["audio_bytes"]
                audio_name = payload["audio_name"]
            else:
                user_input = None
            if user_input is not None:
                st.markdown("---")
                st.markdown("#### Current turn")
                panels = run_pipeline(user_input, audio_bytes, audio_name, chat)
                chat_store.add_turn(data, chat["id"], {
                    "input": (
                        panels.get("transcript") or panels.get("input")
                        or user_input
                    ),
                    "panels": panels,
                })
                chat_store.save_user(s.user_slug, data)

        st.markdown("---")
        st.markdown("##### Try a sample or attach audio")
        _render_input_row(s)


def main():
    if "user_slug" not in st.session_state:
        login_screen()
        return
    # Warm up heavy models right after login so the first user message is
    # instant. Cached via @st.cache_resource — runs once per Streamlit process.
    load_pipeline()
    sidebar()
    main_view()


main()
