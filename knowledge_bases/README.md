# Knowledge bases

All raw datasets are gitignored. Download manually into the matching subfolder.

## RAG knowledge-base sources (indexed into ChromaDB)

`scripts/build_chromadb.py` iterates over these 6 folders only.

| Folder | Dataset | Source | Expected file |
|---|---|---|---|
| `medical_transcriptions/` | Medical Transcriptions | Kaggle | `mtsamples.csv` |
| `bioasq/` | BioASQ training set | bioasq.org (free registration) | `BioASQ-training12b.json` |
| `medquad/` | MedQuAD | github.com/abachaa/MedQuAD | `medquad.csv` |
| `drugbank/` | DrugBank (approved) | drugbank.com (academic license) | `drugbank_approved.xml` |
| `medrag_textbooks/` | MedRAG Textbooks | huggingface.co/datasets/MedRAG/textbooks | `textbooks.jsonl` |
| `medtext/` | MedText | huggingface.co/datasets/BI55/MedText | `medtext.jsonl` |

## Non-KB datasets (stored here for convenience, NOT indexed)

| Folder | Dataset | Consumer | Expected file |
|---|---|---|---|
| `bc5cdr/` | BC5CDR (Chemical Disease Relation) | `notebooks/01_ner_evaluation.ipynb` (NER benchmark vs `en_core_web_sm`) | huggingface.co/datasets/bigbio/bc5cdr |
| `drug_review/` | UCI Drug Reviews | `scripts/train_sentiment.py` (LogReg + TF-IDF sentiment model) | TSV from UCI ML repo |

> Source URLs and expected filenames are provisional — confirm against each provider before scripting downloads.

After downloading, run `python scripts/check_datasets.py` (TBD) to verify file presence and minimum row counts.
