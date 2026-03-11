#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List

from eventlens_tools import extract_event_candidates
from phase6_agent_answer import run_agent


EVAL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "Which companies announced acquisitions?",
        "expected_symbols": ["XOS", "PATK"],
        "expected_companies": ["Xos", "Patrick Industries"],
        "min_expected_count": 2,
    },
    {
        "question": "Which companies acquired Sportech?",
        "expected_symbols": ["PATK"],
        "expected_companies": ["Patrick Industries"],
        "min_expected_count": 1,
    },
    {
        "question": "Which companies acquired ElectraMeccanica?",
        "expected_symbols": ["XOS"],
        "expected_companies": ["Xos"],
        "min_expected_count": 1,
    },
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def contains_citation(answer: str) -> bool:
    return bool(re.search(r"\[Source\s+\d+\]", answer))


def score_answer(answer: str, expected_symbols: List[str], expected_companies: List[str]) -> Dict[str, Any]:
    answer_n = normalize_text(answer)

    matched_symbols = [s for s in expected_symbols if normalize_text(s) in answer_n]
    matched_companies = [c for c in expected_companies if normalize_text(c) in answer_n]

    matched_any = list(set(matched_symbols + matched_companies))

    return {
        "matched_symbols": matched_symbols,
        "matched_companies": matched_companies,
        "matched_any_count": len(matched_any),
        "has_citation": contains_citation(answer),
    }


def score_extraction(candidates: List[Dict[str, Any]], expected_symbols: List[str], expected_companies: List[str]) -> Dict[str, Any]:
    candidate_text = " ".join(
        [
            f"{c.get('company', '')} {c.get('symbol', '')} {c.get('target', '')} {c.get('title', '')}"
            for c in candidates
        ]
    )
    candidate_text_n = normalize_text(candidate_text)

    matched_symbols = [s for s in expected_symbols if normalize_text(s) in candidate_text_n]
    matched_companies = [c for c in expected_companies if normalize_text(c) in candidate_text_n]

    return {
        "matched_symbols": matched_symbols,
        "matched_companies": matched_companies,
        "candidate_count": len(candidates),
    }


def evaluate_one(
    item: Dict[str, Any],
    *,
    collection: str,
    qdrant_url: str,
    ollama_host: str,
    embed_model: str,
    llm_model: str,
    top_k: int,
    candidate_k: int,
    max_chunks_per_accession: int,
    evidence_chars: int,
) -> Dict[str, Any]:
    question = item["question"]

    t0 = time.perf_counter()
    result = run_agent(
        question=question,
        collection=collection,
        qdrant_url=qdrant_url,
        ollama_host=ollama_host,
        embed_model=embed_model,
        llm_model=llm_model,
        top_k=top_k,
        candidate_k=candidate_k,
        max_chunks_per_accession=max_chunks_per_accession,
        symbol=None,
        exchange=None,
        accession=None,
        date_from=None,
        date_to=None,
        evidence_chars=evidence_chars,
    )
    latency_sec = time.perf_counter() - t0

    evidence = result["evidence"]
    candidates = result["candidates"]
    answer = result["answer"]

    answer_score = score_answer(
        answer,
        expected_symbols=item["expected_symbols"],
        expected_companies=item["expected_companies"],
    )

    extraction_score = score_extraction(
        [c.to_dict() for c in candidates],
        expected_symbols=item["expected_symbols"],
        expected_companies=item["expected_companies"],
    )

    evidence_symbols = [e.symbol for e in evidence if e.symbol]
    retrieval_hit_count = sum(1 for s in item["expected_symbols"] if s in evidence_symbols)

    passed = (
        answer_score["matched_any_count"] >= item["min_expected_count"]
        and answer_score["has_citation"]
    )

    return {
        "question": question,
        "task_type": result["task_type"],
        "plan": result["plan"],
        "latency_sec": round(latency_sec, 3),
        "retrieval_symbols": evidence_symbols,
        "retrieval_hit_count": retrieval_hit_count,
        "extraction_score": extraction_score,
        "answer_score": answer_score,
        "passed": passed,
        "final_answer": answer,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_latency = sum(r["latency_sec"] for r in results) / total if total else 0.0
    citation_pass = sum(1 for r in results if r["answer_score"]["has_citation"])

    return {
        "total_questions": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "avg_latency_sec": round(avg_latency, 3),
        "citation_rate": round(citation_pass / total, 3) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--collection", default="sec_8k_chunks")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--embed_model", default="nomic-embed-text:latest")
    parser.add_argument("--llm_model", default="qwen2.5:7b-instruct")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--max_chunks_per_accession", type=int, default=2)
    parser.add_argument("--evidence_chars", type=int, default=1200)

    parser.add_argument("--out", default="runs/eval_eventlens_results.json")

    args = parser.parse_args()

    results = []
    for item in EVAL_QUESTIONS:
        r = evaluate_one(
            item,
            collection=args.collection,
            qdrant_url=args.qdrant_url,
            ollama_host=args.ollama_host,
            embed_model=args.embed_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_chunks_per_accession=args.max_chunks_per_accession,
            evidence_chars=args.evidence_chars,
        )
        results.append(r)

    summary = summarize(results)

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== EVAL SUMMARY ===\n")
    print(json.dumps(summary, indent=2))

    print("\n=== PER-QUESTION RESULTS ===\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()