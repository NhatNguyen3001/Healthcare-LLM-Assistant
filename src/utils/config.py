"""Project configuration loader.

Single source of environment loading and constants for the healthcare NLP system.

Project rule: NO other module calls `load_dotenv()` or `os.getenv()`. Import from here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- OpenAI ---
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
MODEL_GENERATION    = os.getenv("OPENAI_MODEL_GENERATION", "gpt-5.5-2026-04-23")
MODEL_AGENTS        = os.getenv("OPENAI_MODEL_AGENTS", "gpt-5.4-mini")
MODEL_JUDGE         = os.getenv("OPENAI_MODEL_JUDGE", "gpt-5.4")  # must differ from MODEL_GENERATION
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-transcribe")

# --- Local LLM (Ollama, Phase 6.5) ---
# Ollama binds to localhost only by default; the Streamlit app reaches the
# daemon over loopback. Never expose 11434 publicly.
# Two QLoRA backends are served by the same `ollama serve` daemon; the router
# in src/pipeline/llm.py picks which tag to hit based on UI selection.
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LOCAL_MODEL_QWEN    = os.getenv("LOCAL_MODEL_QWEN", "medqa-qwen")
LOCAL_MODEL_LLAMA32 = os.getenv("LOCAL_MODEL_LLAMA32", "medqa-llama32")

# --- ChromaDB ---
CHROMA_PERSIST_PATH      = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_PATH", "data/chromadb")
CHROMA_COLLECTION        = os.getenv("CHROMA_COLLECTION", "healthcare_knowledge_base")
CHROMA_K                 = 7
CHROMA_DISTANCE_THRESHOLD = 0.8  # cosine distance; drop hits above this (i.e. cosine_sim < 0.2). 0-survivor triggers web-search fallback.

# --- Models ---
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
NER_MODEL       = "en_ner_bc5cdr_md"

# --- Privacy filter ---
AGE_BANDS = {
    (0, 17):   "paediatric patient",
    (18, 35):  "young adult patient",
    (36, 55):  "middle-aged patient",
    (56, 75):  "older adult patient",
    (76, 999): "elderly patient",
}

# --- Cleaned dataset paths ---

# 6 sources indexed into ChromaDB (Phase 5)
RAG_SOURCES = {
    "medical_transcriptions": PROJECT_ROOT / "knowledge_bases/medical_transcriptions/mtsamples_cleaned.csv",
    "bioasq":                 PROJECT_ROOT / "knowledge_bases/bioasq/bioasq_cleaned.jsonl",
    "medquad":                PROJECT_ROOT / "knowledge_bases/medquad/medquad_cleaned.jsonl",
    "drugbank":               PROJECT_ROOT / "knowledge_bases/drugbank/drugbank_cleaned.jsonl",
    "medrag_textbooks":       PROJECT_ROOT / "knowledge_bases/medrag_textbooks/textbooks_cleaned.jsonl",
    "medtext":                PROJECT_ROOT / "knowledge_bases/medtext/medtext_cleaned.jsonl",
}

# Sentiment training data (Phase 4.1) — NOT indexed
SENTIMENT_TRAINING_DATA = {
    "train":      PROJECT_ROOT / "knowledge_bases/drug_review/drug_review_train_cleaned.csv",
    "test":       PROJECT_ROOT / "knowledge_bases/drug_review/drug_review_test_cleaned.csv",
    "validation": PROJECT_ROOT / "knowledge_bases/drug_review/drug_review_validation_cleaned.csv",
}

# NER evaluation data — NOT indexed
NER_EVAL_DATA = {
    "test": PROJECT_ROOT / "knowledge_bases/bc5cdr/bc5cdr_test_cleaned.jsonl",
}

# --- Output paths ---
MODELS_DIR            = PROJECT_ROOT / "models"
SENTIMENT_MODEL_PATH  = MODELS_DIR / "sentiment_model.pkl"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier_model.pkl"
RESULTS_DIR           = PROJECT_ROOT / "results"
