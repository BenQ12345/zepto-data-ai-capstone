# Zepto Data & AI Platform Capstone

One public repository containing three connected, independently gradable modules:

- `/data_pipeline` — scrape → clean → convert → relational SQLite → SQL/pandas validation.
- `/analytics` — one Titanic dataset load → EDA/cleaning → predictive modeling → regression → saved pipeline.
- `/support_assistant` — local document embeddings + ChromaDB → LangGraph routing/RAG → structured FastAPI service.

## Setup

This repository uses **one consolidated `requirements.txt`** at the root.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run each module

### 1. Data pipeline

```bash
cd data_pipeline
python pipeline.py
```

This scrapes the first five catalogue pages (at least 60 books), cleans the fields, applies the fixed `1 GBP = 105.50 INR` rate, creates `books.db`, runs the required SQL, and validates the SQL JOIN against `pandas.merge`.

### 2. Analytics

Run the EDA first, then modeling:

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` calls `sns.load_dataset('titanic')` exactly once, immediately writes `analytics/titanic.csv`, and then performs the EDA. `02_modeling.py` reads that same committed CSV and never calls the Seaborn loader.

### 3. Support assistant

Build the local vector index:

```bash
cd support_assistant
python ingest.py
```

Run the API in the required deterministic mock mode:

```bash
# MOCK_LLM is optional; unset means mock mode.
uvicorn main:app --host 0.0.0.0 --port 7860
```

Examples:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee under INR 149?\"}"
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of India?\"}"
```

The optional real-LLM extension is enabled only with `MOCK_LLM=0` and a separately supplied provider key. No API key is required for grading the default mock path.

## Git workflow evidence

Before submission, the repository history should show a feature branch with at least two commits followed by a merge into `main`. A reproducible example:

```bash
git checkout -b feature/capstone
# make changes
git add . && git commit -m "Add capstone modules"
# make another change
git add . && git commit -m "Add validation and documentation"
git checkout main
git merge --no-ff feature/capstone -m "Merge capstone feature branch"
git log --graph --all --oneline
```

## Design summary

### Data pipeline
Uses BeautifulSoup/requests for raw acquisition, explicit parsing functions for messy text, a project-defined constant conversion rate, and a normalized two-table SQLite schema (`categories` → `books`). SQL outputs and a pandas equivalent JOIN are produced by the same run.

### Analytics
The raw Titanic dataset is loaded once from Seaborn and immediately snapshotted to `titanic.csv` for offline grading. EDA and modeling use the same repository artifact; model preprocessing is encapsulated in scikit-learn pipelines so imputers, encoders, and scalers are fit only on training data.

### Support assistant
The eight supplied policy documents are chunked and embedded with `all-MiniLM-L6-v2`, stored in ChromaDB, retrieved by the LangGraph `retrieve_and_answer` node, and exposed through FastAPI. The required grading mode is deterministic `MOCK_LLM` logic; real generation is an optional extension.

## No screenshots/PDFs

All required reasoning is represented as Markdown in the repository. Generated chart PNGs are only supporting artifacts; every required interpretation is also written in Markdown/text.
