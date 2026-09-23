# Zepto Support Assistant

## Required baseline

The graded path uses deterministic mock LLM behavior. Leave `MOCK_LLM` unset or set it to `1`; this path needs no LLM account and makes no LLM network calls.

### Build the ChromaDB collection

```bash
python ingest.py
```

`ingest.py` loads exactly the eight supplied documents, uses one `all-MiniLM-L6-v2` SentenceTransformer instance to embed each document-sized chunk, and stores the normalized vectors in the persistent `zepto_policy` ChromaDB collection with cosine distance.

### Run FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Required mock-mode examples

Policy query:

```json
{"query":"What is the delivery fee for an order below INR 149?"}
```

A typical raw response shape is:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.","sources":["doc_01","doc_05","doc_06"],"confidence":1.0}
```

General query:

```json
{"query":"What is the capital of India?"}
```

Expected raw response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The exact policy `sources` order is determined by Chroma similarity ranking at runtime; the example above documents the required response schema and representative mock output.

## Prompt design

The prompt in `prompt_template.py` explicitly contains the five requested skeleton elements:

- **Role** — Zepto grounded support assistant.
- **Context** — retrieved Zepto policy text.
- **Task** — resolve the user's policy question.
- **Format** — exact JSON schema.
- **Length** — concise 2–4 sentences.

It also has the negative constraint **“Do not answer using information that is not present in the provided context”** and a few-shot delivery-fee example. The prompt is used by the optional real-LLM generation path.

## LangGraph flow

`graph.py` builds a `StateGraph` with a typed `GraphState` and three named nodes:

```text
Query
  ↓
classify_intent
  ├── policy_question ──→ retrieve_and_answer ──→ JSON response
  │                         ↑
  │                    ChromaDB top-3
  │                         ↑
  │                    MiniLM embedding
  │
  └── general_question ──→ direct_answer ──→ JSON response
```

### Mock toggle behavior

`MOCK_LLM` controls only generation/classification behavior:

- `MOCK_LLM` unset or `1`: `classify_intent` uses the required keyword heuristic; `retrieve_and_answer` performs real embedding + ChromaDB retrieval and then builds the deterministic `Based on the retrieved context: ...` response; `direct_answer` returns its deterministic fixed string. No LLM provider is called.
- `MOCK_LLM=0`: intent classification and answer generation can use the optional Groq-backed real LLM path. Retrieval itself still runs through ChromaDB. Real-LLM JSON validation retries up to two additional times after the first failed validation.

## RAG architecture

**Ingestion → Embedding → Retrieval → Generation**

1. **Ingestion:** `docs/doc_01.txt` … `doc_08.txt` are read by `ingest.py` as eight chunks, each retaining its document ID.
2. **Embedding:** `SentenceTransformer('all-MiniLM-L6-v2')` creates normalized embeddings locally. `ingest.py` stores those vectors in the persistent `zepto_policy` ChromaDB collection using cosine distance.
3. **Retrieval:** `AssistantGraph.retrieve_and_answer()` embeds the incoming policy query with the same model and asks ChromaDB for the top three similar chunks.
4. **Generation:** In mock mode, the top chunk snippet becomes the deterministic answer. In the optional real mode, `retrieve_and_answer()` builds the structured prompt from the retrieved chunks and calls the LLM. `direct_answer()` bypasses retrieval for general questions.

## Docker

The Dockerfile is the required local container baseline. Build and run from the **repository root** because the Docker build copies the consolidated requirements file:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

The image leaves `MOCK_LLM=1` by default and runs the FastAPI service at port 7860. The Docker build runs `ingest.py`, so the ChromaDB collection is prepared inside the image.

The optional real-LLM and Hugging Face deployment extensions are not required for grading.
