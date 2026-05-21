# QLoRA training dataset

Built by `scripts/prepare_qlora_dataset.py`. Schema is messages format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "..."}
```

## Counts

| Split | Pairs |
|---|---|
| train | 8100 |
| val | 450 |
| test | 450 |
| **total** | **9000** |

## Source breakdown (training split)

| Source | Pairs |
|---|---|
| bioasq | 1345 |
| drugbank-description | 1349 |
| drugbank-indication | 1339 |
| drugbank-mechanism_of_action | 1350 |
| drugbank-side_effects | 1364 |
| medquad | 1353 |

## Cleaning thresholds

- `MIN_ANSWER_CHARS = 30`
- `MAX_ANSWER_CHARS = 1500` (truncated on sentence boundary)
- `MAX_QUESTION_CHARS = 400`
- `--per-source-limit = 1500` (0 = no cap)

## Reproducibility

- Shuffle seed: 42
- Split ratios: 90 / 5 / 5
- Chat template (Qwen2.5) applied at train time, NOT here. Swap the base
  model without rebuilding this dataset.
