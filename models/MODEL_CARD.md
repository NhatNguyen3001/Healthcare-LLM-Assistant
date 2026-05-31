---
license: cc-by-nc-4.0
language:
  - en
library_name: peft
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen2.5-1.5B-Instruct
  - meta-llama/Llama-3.2-1B-Instruct
tags:
  - medical
  - healthcare
  - clinical
  - qlora
  - peft
  - lora
  - qwen
  - qwen2.5
  - llama
  - llama-3.2
  - ollama
  - gguf
---

# Healthcare LLM Assistant - QLoRA fine-tunes

Two parallel QLoRA fine-tunes of small instruct models on the same 9,000-pair mix of public biomedical Q&A, served side-by-side in the parent project's Streamlit UI for a 3-way bake-off against GPT-5.5.

| Variant | Subfolder | Base | Adapter | GGUF (Q4_K_M) |
|---|---|---|---|---|
| **Qwen** | `qwen/` | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | `qwen/qwen-medqa-adapter/` (~82 MB) | `qwen/qwen-medqa-gguf/model.Q4_K_M.gguf` (~941 MB) |
| **Llama-3.2** | `llama32/` | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `llama32/llama32-medqa-adapter/` (~50 MB) | `llama32/llama32-medqa-gguf/model.Q4_K_M.gguf` (~770 MB) |

Both variants were trained with the same dataset, the same LoRA shape (r=16, α=32, all 7 projection layers) and the same SFT recipe, so any quality gap isolates the base-model effect.

Built as part of the COMP8420 (Macquarie University) main project on a healthcare NLP assistant. Companion code: **https://github.com/NhatNguyen3001/Healthcare-LLM-Assistant**
(see the GitHub README for the full system: voice input, PII railguard, multi-agent RAG, evaluation notebooks.)

## What is in this repo

```
.
├── qwen/
│   ├── qwen-medqa-adapter/                  # PEFT LoRA adapter
│   └── qwen-medqa-gguf/
│       ├── model.Q4_K_M.gguf                # Ollama-ready GGUF
│       └── Modelfile                        # Ollama registration recipe
└── llama32/
    ├── llama32-medqa-adapter/               # PEFT LoRA adapter
    └── llama32-medqa-gguf/
        ├── model.Q4_K_M.gguf
        └── Modelfile
```

The merged-but-unquantized `safetensors` is intentionally not uploaded for either variant; it is redundant for end users (use the GGUF for Ollama OR the adapter for transformers+peft).

## Training data

9,000 question-answer pairs (train 8,100 / val 450 / test 450) drawn from six public sources, capped at 1,500 pairs per source for balance:

| Source | Pairs | Notes |
|---|---|---|
| BioASQ (subset of training14b) | ~1,500 | factoid / list / summary biomedical Q&A |
| MedQuAD | ~1,500 | consumer-facing medical questions |
| DrugBank `description` | ~1,500 | "What is X?" templates |
| DrugBank `indication` | ~1,500 | indication / contraindication |
| DrugBank `side_effects` | ~1,500 | side-effect summaries |
| DrugBank `mechanism_of_action` | ~1,500 | MoA explanations |

90 / 5 / 5 random split with `seed=42`. The OpenAI messages format is used at JSONL level; each variant's chat template (Qwen2.5 or Llama-3.1) is applied at training time, not stored in the JSONL.

## Training setup

Same hyperparameters across both variants:

| Hyperparameter | Value |
|---|---|
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA target modules | all 7 projection layers (q, k, v, o, gate, up, down) |
| Max sequence length | 1024 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule |
| Optimizer | `adamw_8bit` |
| Seed | 42 |
| Hardware | RTX 4060 (8 GB, bf16) |

Per-variant differences:

| | Qwen | Llama-3.2 |
|---|---|---|
| Base id | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4) | `meta-llama/Llama-3.2-1B-Instruct` (4-bit NF4) |
| Chat template | `qwen-2.5` | `llama-3.2` |
| Wall time (3 epochs) | ~95 min | ~58 min (smaller base) |
| Final train loss | 1.3646 | 1.4843 |
| Best val loss | 1.5536 (~epoch 1.97) | 1.6955 (~epoch 1.97) |

Deployed checkpoints are end-of-epoch-3 for both (the "what a full QLoRA run gives you" baseline, not early-stopped).

## Evaluation

Evaluated on the held-out 450-pair test set, with 100 stratified pairs (~17 per source) used as the common comparison sample across all evaluation notebooks.

Two evaluation passes:

1. **Surface metrics**: ROUGE-1/2/L + BERTScore-F1 (with the PubMedBERT backbone)
2. **LLM-as-judge**: GPT-5.4 scoring blind on Accuracy / Completeness / Clarity / Safety (0-10), reference-aware

**3-way results (100 stratified test pairs, `seed=42`):**

Surface metrics (ROUGE + BERTScore with PubMedBERT backbone):

| Metric | GPT-5.5 | QLoRA Qwen | QLoRA Llama-3.2 |
|---|---|---|---|
| ROUGE-1 | 0.2955 | 0.2997 | **0.3049** |
| ROUGE-2 | 0.0907 | **0.1087** | 0.1105 |
| ROUGE-L | 0.1921 | **0.2101** | 0.2046 |
| BERTScore-F1 | 0.8221 | **0.8293** | 0.8272 |

LLM-as-judge (GPT-5.4, 0-10 scale):

| Dimension | GPT-5.5 | QLoRA Qwen | QLoRA Llama-3.2 |
|---|---|---|---|
| Accuracy | **9.26** | 3.57 | 2.77 |
| Completeness | **8.24** | 3.08 | 2.70 |
| Clarity | **9.35** | 6.69 | 6.41 |
| Safety | **9.56** | 5.01 | 4.47 |

Latency:

| Model | Mean latency |
|---|---|
| GPT-5.5 (cloud) | 7.22 s |
| QLoRA Qwen (local, RTX 4060) | 0.98 s |
| QLoRA Llama-3.2 (local, RTX 4060) | **0.63 s** |

**Key findings:**

- Both QLoRA models edge out GPT-5.5 on surface metrics via **template substitution** on DrugBank-style entries (71+ sibling templates in train share the same skeleton). The fine-tunes learn the template and slot-fill entities at inference. Verified 0/450 literal Q+A pair overlap between train and test, so this is template generalization, not memorization.
- GPT-5.5 dominates on all judge dimensions. The Accuracy gap is the headline finding: the 1B-scale fine-tunes hallucinate plausible-sounding but factually wrong medical content that ROUGE and BERTScore (even with PubMedBERT) cannot detect.
- Between the two locals, Qwen edges Llama-3.2 on every judge dimension. Llama-3.2 is faster (0.63 s vs 0.98 s) due to its smaller parameter count.
- Both local models are 7-11x faster than the cloud path.

Detailed numbers and charts live in the parent repo:

- `results/llm_generation_evaluation.csv` + `llm_generation_eval_chart.png` + `llm_generation_bertscore_chart.png`
- `results/llm_judge_evaluation.csv` + `llm_judge_eval_chart.png`
- `results/model_comparison.csv` + `model_comparison_chart.png`
- `results/qlora_loss_curve.png` + `results/qlora_source_mix.png`

## How to use

The examples below use the **Qwen** variant. For the **Llama-3.2** variant, swap every `qwen` for `llama32` in the paths and use the Ollama tag `medqa-llama32`.

### Option 1: Ollama (recommended for local serving)

```bash
# Install Ollama (https://ollama.com/download) first. On Windows it auto-starts as a service.

# Fetch one variant's GGUF + Modelfile
pip install huggingface_hub
huggingface-cli download Davis426/Healthcare-LLM-Assistant \
  --include "qwen/qwen-medqa-gguf/*" \
  --local-dir ./models

# Register with Ollama
cd ./models/qwen/qwen-medqa-gguf
ollama create medqa-qwen -f Modelfile

# Try it
ollama run medqa-qwen "What are the side effects of amoxicillin?"
```

For the Llama variant, swap every `qwen` for `llama32` (paths) and the Ollama tag to `medqa-llama32`.

You can register both side-by-side; one `ollama serve` daemon handles both tags concurrently (`OLLAMA_MAX_LOADED_MODELS` defaults to 3).

### Option 2: transformers + peft (Python)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# pick a variant
base_id    = "Qwen/Qwen2.5-1.5B-Instruct"
subfolder  = "qwen/qwen-medqa-adapter"
# or:
# base_id   = "meta-llama/Llama-3.2-1B-Instruct"
# subfolder = "llama32/llama32-medqa-adapter"
adapter_id = "Davis426/Healthcare-LLM-Assistant"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id, subfolder=subfolder)

messages = [{"role": "user", "content": "What are the side effects of amoxicillin?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### Option 3: llama.cpp directly

```bash
pip install huggingface_hub
huggingface-cli download Davis426/Healthcare-LLM-Assistant \
  --include "qwen/qwen-medqa-gguf/model.Q4_K_M.gguf" --local-dir .

./llama-cli -m qwen/qwen-medqa-gguf/model.Q4_K_M.gguf \
  -p "What are the side effects of amoxicillin?" -n 256
```

## Limitations

Both models are teaching / research artifacts. **Do not use for real clinical decisions.** Specifically:

- **Catastrophic forgetting on out-of-distribution prompts.** Fine-tuning on a narrow Q&A distribution at the 1-1.5B parameter scale shifts each base model hard. Casual / non-medical questions get answered in MedQA-style; the base model's general conversational ability is degraded.
- **Weakened in-context grounding.** Every training pair has shape `user_question -> answer`, with no retrieved-context block. As a result both fine-tuned models partly lose the ability to read RAG passages in the prompt and tend to answer from parametric memory even when correct evidence is supplied. The parent repo's MASS-RAG pipeline retains GPT-5.5 for cases where grounded answers matter; the local models are sidebar-selectable for the comparison experience.
- **No factual safety net.** Both training data and evaluation rely on existing biomedical corpora; the models have no live knowledge cutoff or up-to-date drug-interaction database. The parent repo applies a regex-based PII railguard on user input, but model output itself is not safety-filtered beyond what each base model already does.
- **English only.**
- **Llama-3.2 base licence:** Llama-3.2 community licence applies to the Llama variant (acceptance via the gated HF repo); see the Meta licence for permitted uses.

## License

The fine-tuned adapters and GGUFs in this repo are released under `cc-by-nc-4.0` (research and non-commercial use). Base model licences override where stricter: Qwen2.5 is Apache-2.0; Llama-3.2 is under the Meta Llama 3.2 Community Licence. Downstream dataset licences may impose additional restrictions; please consult each source (BioASQ, MedQuAD, DrugBank, MedRAG textbooks) before redistribution.

## Citation

If you use or build on this work, please reference:

```bibtex
@misc{comp8420-2026-medqa,
  title  = {Healthcare NLP Assistant: parallel QLoRA fine-tunes of Qwen2.5-1.5B and Llama-3.2-1B for medical Q&A},
  author = {Davis426},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/Davis426/Healthcare-LLM-Assistant}}
}
```

Built on top of:

- Qwen2.5 (Alibaba): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Llama-3.2 (Meta): https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- QLoRA (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314
- MASS-RAG (Xiao, Huang, Liu, Xie, 2026): https://arxiv.org/abs/2604.18509 (used by the parent repo's retrieval pipeline that these models plug into)
- Generalist embedding models for clinical semantic search (Excoffier et al., 2024): https://arxiv.org/abs/2401.01943
- Healthcare NER using language model pretraining (Tarcar et al., 2019): https://arxiv.org/abs/1910.11241
- Unsloth: https://github.com/unslothai/unsloth
- llama.cpp + Ollama for GGUF serving
