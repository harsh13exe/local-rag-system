# Local RAG System with Automated Evaluation

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system that answers questions about a document using only locally-run open models — no external API calls, no data leaving the machine. Built and evaluated end-to-end, from ingestion through Dockerized deployment.

## What this project demonstrates

- Building a RAG pipeline from scratch: ingestion → chunking → embedding → retrieval → generation
- Running entirely on local/open-source models via Ollama (no OpenAI/Claude API dependency)
- Serving the pipeline as a real API (FastAPI), not just a notebook
- Writing an automated evaluation harness to measure accuracy and hallucination rate, rather than eyeballing outputs
- Empirically comparing two candidate models and picking one based on measured evidence, not assumption
- Containerizing a multi-service AI system (app + local LLM server) with Docker Compose
- Debugging real production issues: bad source data, retrieval tuning, and model reasoning inconsistencies

## Architecture

```
Documents (PDF/txt/md)
        │
        ▼
  ingest.py — chunks documents, embeds with nomic-embed-text, stores in ChromaDB
        │
        ▼
   ChromaDB (persistent local vector store)
        │
        ▼
  main.py (FastAPI) — POST /ask
        │
        ├─ retrieves top-k relevant chunks (similarity search)
        ├─ builds a grounded prompt (context + question)
        ├─ generates an answer via llama3.2:3b (Ollama)
        └─ returns { answer, sources[] } with page-level citations
```

In Docker, this runs as two services: an `ollama` container serving the models, and an `app` container serving the FastAPI application, networked together via Docker Compose.

## Tech stack

| Component | Choice |
|---|---|
| Orchestration | LangChain / langchain-ollama |
| Embedding model | `nomic-embed-text` (via Ollama) |
| Generation model | `llama3.2:3b` (via Ollama) — selected after A/B testing, see below |
| Vector store | ChromaDB (local, persistent) |
| API layer | FastAPI + Pydantic |
| Evaluation | Custom keyword/refusal-based harness (`eval.py`) |
| Deployment | Docker + Docker Compose (multi-container: app + Ollama) |

Runs fully on CPU, no GPU required — tested on an 8GB RAM MacBook Pro.

## Evaluation results

A test set of 10 questions (7 answerable from the document, 3 designed to have no answer — to test hallucination resistance) was run 3 times each through the deployed API.

| Model | Accuracy | Notes |
|---|---|---|
| `llama3.2:3b` (base) | 30/30 (100%) | Fully correct on all keyword checks across all 3 runs; occasional slightly hedgy phrasing on one date-comparison question, but never dropped a fact. |
| `qwen2.5:3b` | 27/30 (90%) | Cleaner phrasing on the same date-comparison question, but omitted a factual detail (a payment gateway) in a different answer across all 3 runs. |
| `harsh-resume-model` (fine-tuned) | **30/30 (100%)** | Matched the base model's factual completeness, and fixed its exact reasoning weakness — explicit, correct "ends with Present" logic and no hedging, across all 3 runs. |

**Decision:** kept `llama3.2:3b` as the initial production model based on this round; the fine-tuned model (see below) subsequently improved on its main weakness without regressing accuracy elsewhere. This progression — base model selection, then targeted fine-tuning — is a concrete example of empirical, evidence-based iteration rather than picking or tuning a model by assumption.

All 3 "should-refuse" questions (e.g. "What is Harsh's salary expectation?") were correctly answered with "I don't have enough information" across all three models and all runs — no hallucination observed on out-of-scope questions, including after fine-tuning.

## Fine-tuning

**Motivation:** while `llama3.2:3b` was factually accurate, it showed a specific, reproducible reasoning weakness — when a resume contained multiple date-ranged entries, it sometimes produced hedging or logically confused phrasing (e.g. misidentifying which entry was current) even though it usually still landed on the right keywords. `qwen2.5:3b` fixed the phrasing but introduced a worse problem (a dropped fact). Rather than accept either tradeoff, the base model was fine-tuned to directly address this weakness.

**Approach:**
- **Method:** LoRA (via [Unsloth](https://github.com/unslothai/unsloth)) on top of `unsloth/Llama-3.2-3B-Instruct`, run on a free Google Colab T4 GPU (local hardware has no GPU, making cloud fine-tuning the practical choice for this project).
- **Dataset:** 35 hand-written instruction/input/output examples in Alpaca format (`training_data.jsonl`), focused on the specific weakness identified above — resolving the "current" entry among multiple date ranges, staying grounded in context, and refusing appropriately when information is absent. Mixed synthetic examples (to teach the general pattern) with resume-specific examples (for this project's domain).
- **Training:** 3 epochs, LoRA rank 16, batch size 8 (effective, via gradient accumulation), ~15 total steps. Training loss dropped from ~2.6 to ~0.93 over the run.
- **Export:** merged LoRA weights into the base model, converted to GGUF (`q4_k_m` quantization) for compatibility with Ollama, and loaded locally via a generated `Modelfile`.

**Result:** the fine-tuned model matched the base model's 100% keyword accuracy while eliminating the hedging/misreasoning pattern — see the "current job title and company" answer specifically improving from ambiguous ("not explicitly mentioned... it can be inferred") to explicit and correct ("this entry ends with 'Present', indicating it is his current role").

**Honest limitation:** the training set (35 examples) and evaluation set (10 questions) are both small and narrowly scoped to this project. This is sufficient to demonstrate the fine-tuning *process* end-to-end (data prep → LoRA training → GGUF export → Ollama deployment → A/B evaluation) and to nudge a specific, identified behavior, but a rigorous claim of general improvement would require a larger and more diverse dataset and test suite.

## Debugging notes (real issues hit and fixed)

1. **Bad source data:** An unrelated 102-page document was accidentally sitting in the ingestion folder alongside the intended file, causing retrieval to surface irrelevant content. Fixed by auditing `documents/` and re-ingesting with only the correct file.
2. **Chunking too aggressive:** Initial 500-character chunks with 50-character overlap sometimes split a job title away from its company name. Increased to 800/150, which fixed retrieval of multi-part facts.
3. **Small-model reasoning on ambiguous dates:** With multiple resume entries with similar-looking date ranges, the 3B model initially failed to identify the "current" role. Fixed with an explicit prompt instruction to look for the entry ending in "Present."

## Project structure

```
rag-project/
├── ingest.py             # Document loading, chunking, embedding, vector store creation
├── main.py               # FastAPI app (POST /ask, GET /health)
├── eval.py               # Automated evaluation harness
├── training_data.jsonl   # LoRA fine-tuning dataset (Alpaca format)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── documents/            # Source documents (not committed — user-provided)
└── chroma_db/            # Persisted vector store (generated by ingest.py)

# Not committed (large/environment-specific):
# - the fine-tuned .gguf model file and Modelfile (regenerate via the Colab notebook
#   described in the Fine-tuning section, or load an equivalent model of your own)
```

## Running it

**Locally:**
```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python ingest.py
uvicorn main:app --reload
```

**With Docker:**
```bash
docker compose up --build
docker exec -it rag-ollama ollama pull llama3.2:3b
docker exec -it rag-ollama ollama pull nomic-embed-text
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current job title and company?"}'
```

## Known limitations

- 3B-scale local models occasionally vary in explanatory phrasing (though not in factual content) across otherwise identical queries.
- No re-ranking step after initial retrieval — a cross-encoder re-ranker could improve precision on ambiguous queries.
- Evaluation is keyword-based, not semantic similarity-based (e.g. RAGAS) — a good next iteration.

## Possible future work

- Expand the fine-tuning dataset (currently 35 examples) to a larger, more diverse set to validate the reasoning improvement generalizes beyond this specific test suite
- Add a re-ranking step (e.g. cross-encoder) between retrieval and generation
- Swap keyword-based eval for RAGAS-style semantic scoring (faithfulness, answer relevancy, context precision)
- Add CI integration (GitHub Actions) to run `eval.py` automatically on every change and block regressions

## Resume-ready summary

> Built and deployed a fully local RAG system (LangChain, Ollama, ChromaDB, FastAPI, Docker) achieving 100% accuracy on a 30-check automated evaluation suite. Diagnosed and fixed data-quality, chunking, and model-reasoning issues; empirically A/B tested candidate generation models and selected/fine-tuned based on measured accuracy rather than assumption. LoRA fine-tuned a 3B model (Unsloth, Google Colab GPU) on a custom dataset to resolve a specific reasoning weakness identified during evaluation, then exported to GGUF and deployed via Ollama — verified with the same automated eval suite. Fully containerized with Docker Compose for reproducible deployment.