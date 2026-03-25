# RAG-Based Intelligent Document QA System

> **Research paper question-answering powered by Mistral-7B (QLoRA fine-tuned) + FAISS dense retrieval + BM25 sparse retrieval with Reciprocal Rank Fusion.**

---

## Architecture

```
PDF Papers
    │
    ▼
┌─────────────────┐
│  PDF Loader     │  PyMuPDF — fast extraction, handles complex layouts
│  + Chunker      │  RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
└────────┬────────┘
         │ chunks
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────┐
│ FAISS  │  │ BM25 │   Dense (semantic) + Sparse (keyword) indexes
│ Index  │  │Index │
└────┬───┘  └──┬───┘
     │         │
     └────┬────┘
          │  Reciprocal Rank Fusion (RRF)
          ▼
   ┌─────────────┐
   │  Top-5 docs │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────────┐
   │  Mistral-7B-Instruct│  4-bit QLoRA, optionally fine-tuned on domain corpus
   │  + RAG Prompt       │
   └──────┬──────────────┘
          │
          ▼
   ┌─────────────┐
   │  FastAPI    │  REST API → Azure App Service
   └─────────────┘
```

## Key Results

| Metric | Dense Only | Hybrid (Dense + BM25) | Improvement |
|--------|-----------|----------------------|-------------|
| Recall@5 | 0.61 | 0.72 | **+18%** |
| Avg Latency | 210ms | 195ms | - |

| Model | Answer Accuracy |
|-------|----------------|
| Mistral-7B (zero-shot) | Baseline |
| Mistral-7B + QLoRA fine-tuned | **+22%** |

---

## Project Structure

```
rag-document-qa/
├── data/
│   └── papers/              ← Drop your PDFs here
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py    ← PyMuPDF PDF extraction
│   │   └── chunker.py       ← Recursive text splitter
│   ├── retrieval/
│   │   ├── embeddings.py    ← HuggingFace sentence-transformers
│   │   ├── vector_store.py  ← FAISS dense index
│   │   ├── bm25_retriever.py← BM25 sparse retrieval
│   │   └── hybrid_retriever.py ← RRF fusion
│   ├── pipeline/
│   │   ├── prompt_templates.py
│   │   └── rag_chain.py     ← Full end-to-end pipeline
│   ├── finetuning/
│   │   ├── dataset_prep.py  ← QA pair formatting for SFTTrainer
│   │   └── train_qlora.py   ← QLoRA fine-tuning script
│   └── api/
│       ├── main.py          ← FastAPI app
│       └── schemas.py       ← Pydantic models
├── notebooks/
│   ├── 01_RAG_Pipeline_Demo.ipynb
│   └── 02_QLoRA_Finetuning.ipynb
├── scripts/
│   ├── ingest_docs.py       ← Build index from CLI
│   └── evaluate.py          ← ROUGE-L + Recall@5 eval
├── deploy/
│   ├── Dockerfile
│   └── azure-deploy.sh
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
cp .env.example .env   # fill in your tokens
```

### 2. Add Research Papers
```bash
cp your_papers/*.pdf data/papers/
```

### 3. Build Index
```bash
python scripts/ingest_docs.py --pdf_dir data/papers --index_path indexes/faiss
```

### 4. Run API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is federated learning?", "top_k": 5}'
```

---

## Notebooks (Colab / Kaggle)

| Notebook | Description |
|----------|-------------|
| `01_RAG_Pipeline_Demo.ipynb` | Full pipeline walkthrough — ingestion → retrieval → QA |
| `02_QLoRA_Finetuning.ipynb` | Fine-tune Mistral-7B on your domain QA pairs |
| `RAG_Final_clean_file.ipynb` | Final file which you can run directly in your Notebook |

---

## Fine-tuning (Colab Pro / Kaggle GPU)

```bash
# 1. Prepare QA dataset
python -m src.finetuning.dataset_prep

# 2. Train QLoRA adapter (~45 min on A100)
python -m src.finetuning.train_qlora \
    --train_file data/train_finetune_dataset.jsonl \
    --val_file   data/val_finetune_dataset.jsonl \
    --output_dir models/mistral-qlora-adapter

# 3. Use adapter in RAG pipeline (set in .env)
FINETUNED_ADAPTER_PATH=./models/mistral-qlora-adapter
```

---

## Azure Deployment

```bash
az login
chmod +x deploy/azure-deploy.sh
./deploy/azure-deploy.sh
```

API will be live at: `https://rag-document-qa.azurewebsites.net`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Mistral-7B-Instruct-v0.2 (QLoRA fine-tuned) |
| Orchestration | LangChain |
| Dense Retrieval | FAISS + sentence-transformers/all-MiniLM-L6-v2 |
| Sparse Retrieval | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Fine-tuning | QLoRA (peft + trl + bitsandbytes) |
| API | FastAPI + Uvicorn |
| Storage | Azure Blob Storage |
| Deployment | Azure App Service |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------||
| GET | `/health` | Liveness check |
| POST | `/query` | Ask a question |
| POST | `/upload` | Upload & index a new PDF |
| GET | `/documents` | List indexed papers |

Interactive docs: `http://localhost:8000/docs`
