# Module 3 — Zepto Support Assistant

This folder talks about the Module 3 requirements. 


## Setup

```bash
cd support_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `all-MiniLM-L6-v2` embedding model is downloaded by Sentence Transformers on first use. ChromaDB stores the vectors locally in `chroma_db/`.

## Run locally

Leave `MOCK_LLM` unset

```bash
export MOCK_LLM=1
uvicorn main:app --reload --port 7860
```

Then test:

```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"What is the delivery fee below INR 149?"}'
```

Expected shape:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": ["doc_01", "doc_02", "doc_03"],
  "confidence": 1.0
}
```

General question:

```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"What is the capital of India?"}'
```

Expected:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

The exact retrieved `sources` ordering can depend on the embedding/index implementation; the top result for a delivery question should be `doc_01`.

```

### Each Stage details

**Ingestion:** `main.py` function `ingest()` reads all eight text files from `docs/`. Each document is treated as one chunk, which is acceptable because the supplied documents are short.

**Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")` creates local embeddings. The embeddings are stored in the ChromaDB collection named `zepto_policies`.

**Retrieval:** `retrieve_and_answer()` embeds the incoming policy question and calls ChromaDB with `n_results=3`. Retrieval uses cosine distance/similarity through the collection configuration.

**Generation:** In required mock mode, `retrieve_and_answer()` creates the deterministic `Based on the retrieved context: ...` response from the top chunk. `direct_answer()` returns the fixed general-question response.

**MOCK_LLM branching:** Retrieval and embeddings always run for policy questions. Only the generation/classification portions branch on `MOCK_LLM`. With the variable unset or `1`, keyword classification and deterministic canned responses run without an LLM network call. With `MOCK_LLM=0`, the code reaches the optional `real_llm_classify()` / `real_llm_answer()` extension points.

## Structured prompt

The `PROMPT_TEMPLATE` in `main.py` contains the required role → context → task → format → length structure, plus an explicit negative constraint and a few-shot example. It is intended for the optional real-LLM generation path.

## JSON validation

The final response is represented by the Pydantic `AnswerResponse` model:

- `answer`: string
- `sources`: list of document/chunk IDs
- `confidence`: float from 0 to 1

Mock mode constructs this model directly, so validation is deterministic.

## Docker

Build:

```bash
docker build -t zepto-support .
```

Run:

```bash
docker run --rm -p 7860:7860 zepto-support
```

Test:

```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"How long can I report a damaged item?"}'
```

The Docker image runs with `MOCK_LLM=1` and therefore does not require an LLM API key.

## Optional real LLM

The supplied requirements make `MOCK_LLM=0` optional and ungraded. The two real-LLM extension functions are deliberately isolated in `main.py` so a provider can be added without changing the required mock pipeline.
