---
license: cc-by-nc-4.0
language:
  - en
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - medical
  - healthcare
  - clinical
  - qlora
  - peft
  - lora
  - qwen
  - qwen2.5
  - ollama
  - gguf
---

# Qwen2.5-1.5B Medical QA (QLoRA)

QLoRA fine-tune of [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on a 9,000-pair mix of six public biomedical Q&A sources. Built as part of the COMP8420 (Macquarie University) main project on a healthcare NLP assistant. The fine-tuned model is served locally via Ollama and benchmarked head-to-head against GPT-5.5 in the parent GitHub repo.

**Companion code**: https://github.com/NhatNguyen3001/COMP8420-Healthcare-LLM-Assistant
(see the GitHub README for the full system: voice input, PII railguard, multi-agent RAG, evaluation notebooks.)

## What is in this repo

| Path | Size | What |
|---|---|---|
| `qwen-medqa-adapter/` | ~82 MB | PEFT LoRA adapter (re-apply to base Qwen2.5-1.5B-Instruct with `peft`) |
| `qwen-medqa-gguf/model.Q4_K_M.gguf` | ~941 MB | Merged + Q4_K_M quantized model, ready for Ollama or llama.cpp |
| `qwen-medqa-gguf/Modelfile` | <1 KB | Ollama registration recipe |

The merged-but-unquantized `safetensors` is intentionally not uploaded; it is redundant for end users (use the GGUF for Ollama OR the adapter for transformers+peft).

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

90 / 5 / 5 random split with `seed=42`. The OpenAI messages format was used at JSONL level; the Qwen2.5 chat template is applied at training time, not stored in the JSONL.

## Training setup

| Hyperparameter | Value |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4 via bitsandbytes) |
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
| Wall time | ~5,667 seconds (~95 minutes) |

Best validation loss: 1.5536 around epoch 1.98. The deployed checkpoint is end-of-epoch-3 (the "what a full QLoRA run gives you" baseline, not early-stopped).

## Evaluation

Evaluated on the held-out 450-pair test set, with 100 stratified pairs (~17 per source) used as the common comparison sample across all evaluation notebooks.

Two evaluation passes:

1. **Surface metrics**: ROUGE-1/2/L + BERTScore-F1 (with the PubMedBERT backbone)
2. **LLM-as-judge**: GPT-5.4 scoring blind on Accuracy / Completeness / Clarity / Safety (0-10), reference-aware

**Headline findings (vs GPT-5.5):**

- This QLoRA model wins ROUGE-L by ~+0.022 (~+12% relative) and BERTScore-F1 by ~+0.0067 (~+0.8% relative)
- The win is driven by **template substitution**, not factual improvement. The training set includes 71+ DrugBank entries sharing the skeleton "`{X}` pollen is the pollen of the `{X}` plant. `{X}` pollen is mainly used in allergenic testing." The fine-tune learns the template and slot-fills the entity at inference; ROUGE and BERTScore both reward this even when the substituted entity is wrong.
- Verified 0 / 450 literal Q+A pair overlap between train and test, so this is template generalization, not memorization.
- Under the LLM-as-judge Accuracy dimension, GPT-5.5 leads (judge results in the parent repo's `results/llm_judge_evaluation.csv`).

Detailed numbers and charts live in the parent repo:

- `results/llm_generation_evaluation.csv` + `llm_generation_eval_chart.png` + `llm_generation_bertscore_chart.png`
- `results/llm_judge_evaluation.csv` + `llm_judge_eval_chart.png`
- `results/model_comparison.csv` + `model_comparison_chart.png`
- `results/qlora_loss_curve.png` + `results/qlora_source_mix.png`

## How to use

### Option 1 — Ollama (recommended for local serving)

```bash
# Fetch the GGUF + Modelfile
huggingface-cli download Davis426/COMP8420-Healthcare-LLM-Assistant \
  --include "qwen-medqa-gguf/*" \
  --local-dir ./models

# Register with Ollama
cd ./models/qwen-medqa-gguf
ollama create medqa-qwen -f Modelfile

# Try it
ollama run medqa-qwen "What is amoxicillin used for?"
```

### Option 2 — transformers + peft (Python)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "Davis426/COMP8420-Healthcare-LLM-Assistant"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id, subfolder="qwen-medqa-adapter")

messages = [{"role": "user", "content": "What is amoxicillin used for?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### Option 3 — llama.cpp directly

```bash
huggingface-cli download Davis426/COMP8420-Healthcare-LLM-Assistant \
  --include "qwen-medqa-gguf/model.Q4_K_M.gguf" --local-dir .

./llama-cli -m model.Q4_K_M.gguf -p "What is amoxicillin used for?" -n 256
```

## Limitations

This model is a teaching / research artifact. **Do not use for real clinical decisions.** Specifically:

- **Catastrophic forgetting on out-of-distribution prompts.** Fine-tuning on a narrow Q&A distribution at 1.5B parameter scale shifts the base model hard. Casual / non-medical questions get answered in MedQA-style; the base model's general conversational ability is degraded.
- **Weakened in-context grounding.** Every training pair has shape `user_question -> answer`, with no retrieved-context block. As a result the fine-tuned model partly loses the ability to read RAG passages in the prompt and tends to answer from parametric memory even when correct evidence is supplied. The parent repo's MASS-RAG pipeline retains GPT-5.5 for cases where grounded answers matter; this local model is sidebar-selectable for the comparison experience.
- **No factual safety net.** Both training data and evaluation rely on existing biomedical corpora; the model has no live knowledge cutoff or up-to-date drug-interaction database. The parent repo applies a regex-based PII railguard on user input, but the model output itself is not safety-filtered beyond what the base model already does.
- **English only.**

## License

`cc-by-nc-4.0` — research and non-commercial use. The base model (Qwen2.5-1.5B-Instruct) is Apache-2.0. Downstream dataset licenses may impose additional restrictions; please consult each source (BioASQ, MedQuAD, DrugBank, MedRAG textbooks) before redistribution.

## Citation

If you use or build on this work, please reference:

```bibtex
@misc{comp8420-2026-medqa-qwen,
  title  = {Healthcare NLP Assistant: QLoRA-fine-tuned Qwen2.5-1.5B for medical Q&A},
  author = {Davis426},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/Davis426/COMP8420-Healthcare-LLM-Assistant}}
}
```

Built on top of:

- Qwen2.5 (Alibaba): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- QLoRA (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314
- MASS-RAG (Xiao, Huang, Liu, Xie, 2026): https://arxiv.org/abs/2604.18509 (used by the parent repo's retrieval pipeline that this model plugs into)
- Unsloth: https://github.com/unslothai/unsloth
- llama.cpp + Ollama for GGUF serving
