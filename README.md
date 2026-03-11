# EventLens

EventLens is a local SEC 8-K intelligence system that combines semantic retrieval, heuristic reranking, structured event extraction, deterministic tool orchestration, evaluation, and confidence-based escalation.

## What it does

Given a question like:

- Which companies announced acquisitions?
- Which companies acquired ElectraMeccanica?
- Which filings mentioned stock repurchases?

EventLens can:

1. semantically search SEC 8-K chunks from Qdrant
2. rerank results with finance-specific heuristics
3. extract structured event candidates
4. generate grounded answers with citations
5. estimate confidence and flag weak outputs for review

---

## Architecture

```text
User Question
    |
    v
phase6_agent_answer.py
    |
    +--> classify_task()
    |
    +--> search_sec_filings()
    |       |
    |       +--> Ollama embeddings
    |       +--> Qdrant vector search
    |       +--> heuristic reranking
    |
    +--> extract_event_candidates()
    |
    +--> answer_from_evidence()
    |       |
    |       +--> grounded answer generation via Ollama
    |
    +--> compute_confidence()
            |
            +--> final_answer / needs_review