#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List
from confidence import compute_confidence

from eventlens_tools import (
    answer_from_evidence,
    candidates_to_json,
    evidence_to_json,
    extract_event_candidates,
    search_sec_filings,
)


def classify_task(question: str) -> str:
    q = question.lower()

    acquisition_terms = [
        "acquisition",
        "acquire",
        "acquired",
        "acquiring",
        "merger",
        "merge",
        "business combination",
        "buyout",
        "takeover",
        "purchase",
    ]

    if any(term in q for term in acquisition_terms):
        return "acquisition_search"

    return "general_sec_search"


def build_agent_plan(task_type: str) -> List[str]:
    if task_type == "acquisition_search":
        return [
            "search_sec_filings",
            "extract_event_candidates",
            "answer_from_evidence",
        ]

    return [
        "search_sec_filings",
        "answer_from_evidence",
    ]


def run_agent(
    question: str,
    *,
    collection: str,
    qdrant_url: str,
    ollama_host: str,
    embed_model: str,
    llm_model: str,
    top_k: int,
    candidate_k: int,
    max_chunks_per_accession: int,
    symbol: str | None,
    exchange: str | None,
    accession: str | None,
    date_from: str | None,
    date_to: str | None,
    evidence_chars: int,
) -> Dict[str, Any]:
    task_type = classify_task(question)
    plan = build_agent_plan(task_type)

    evidence = search_sec_filings(
        query=question,
        collection=collection,
        qdrant_url=qdrant_url,
        ollama_host=ollama_host,
        embed_model=embed_model,
        top_k=top_k,
        candidate_k=candidate_k,
        max_chunks_per_accession=max_chunks_per_accession,
        symbol=symbol,
        exchange=exchange,
        accession=accession,
        date_from=date_from,
        date_to=date_to,
    )

    candidates = []
    if "extract_event_candidates" in plan:
        candidates = extract_event_candidates(evidence, event_type="acquisition")

    answer = answer_from_evidence(
        question=question,
        evidence_chunks=evidence,
        llm_model=llm_model,
        ollama_host=ollama_host,
        evidence_chars=evidence_chars,
    )
    confidence = compute_confidence(
        evidence=evidence,
        candidates=candidates,
        answer=answer,
    )

    return {
        "question": question,
        "task_type": task_type,
        "plan": plan,
        "evidence": evidence,
        "candidates": candidates,
        "answer": answer,
        "confidence": confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", required=True)

    parser.add_argument("--collection", default="sec_8k_chunks")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--embed_model", default="nomic-embed-text:latest")
    parser.add_argument("--llm_model", default="qwen2.5:7b-instruct")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_chunks_per_accession", type=int, default=2)

    parser.add_argument("--symbol")
    parser.add_argument("--exchange")
    parser.add_argument("--accession")
    parser.add_argument("--date_from")
    parser.add_argument("--date_to")

    parser.add_argument("--evidence_chars", type=int, default=1200)

    parser.add_argument("--show_plan", action="store_true")
    parser.add_argument("--show_evidence", action="store_true")
    parser.add_argument("--show_candidates", action="store_true")

    args = parser.parse_args()

    result = run_agent(
        question=args.question,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_host=args.ollama_host,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_chunks_per_accession=args.max_chunks_per_accession,
        symbol=args.symbol,
        exchange=args.exchange,
        accession=args.accession,
        date_from=args.date_from,
        date_to=args.date_to,
        evidence_chars=args.evidence_chars,
    )

    if args.show_plan:
        print("\n=== AGENT PLAN ===\n")
        print(json.dumps({
            "task_type": result["task_type"],
            "plan": result["plan"],
        }, indent=2))

    if args.show_evidence:
        print("\n=== TOOL OUTPUT: search_sec_filings ===\n")
        print(json.dumps(evidence_to_json(result["evidence"]), indent=2))

    if args.show_candidates and result["candidates"]:
        print("\n=== TOOL OUTPUT: extract_event_candidates ===\n")
        print(json.dumps(candidates_to_json(result["candidates"]), indent=2))
    
    print("\n=== CONFIDENCE ===\n")
    print(json.dumps(result["confidence"], indent=2))

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])


if __name__ == "__main__":
    main()