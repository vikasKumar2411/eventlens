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
    # Broad acquisition discovery
    {
        "question": "Which companies announced acquisitions?",
        "expected_symbols": ["XOS", "PATK"],
        "expected_companies": ["Xos", "Patrick Industries"],
        "min_expected_count": 2,
        "category": "broad_acquisition",
    },
    {
        "question": "Which companies announced merger deals?",
        "expected_symbols": ["XOS", "PATK", "EXE"],
        "expected_companies": ["Xos", "Patrick Industries", "Chesapeake Energy"],
        "min_expected_count": 2,
        "category": "broad_transaction",
    },
    {
        "question": "Which filings discussed definitive transaction agreements?",
        "expected_symbols": ["XOS", "PATK", "MAGE"],
        "expected_companies": ["Xos", "Patrick Industries", "Magellan Gold"],
        "min_expected_count": 2,
        "category": "broad_transaction",
    },

    # Target-specific positives
    {
        "question": "Which companies acquired Sportech?",
        "expected_symbols": ["PATK"],
        "expected_companies": ["Patrick Industries"],
        "min_expected_count": 1,
        "category": "target_specific",
    },
    {
        "question": "Which companies acquired ElectraMeccanica?",
        "expected_symbols": ["XOS"],
        "expected_companies": ["Xos"],
        "min_expected_count": 1,
        "category": "target_specific",
    },
    {
        "question": "Which company acquired Southwestern Energy?",
        "expected_symbols": ["EXE"],
        "expected_companies": ["Chesapeake Energy"],
        "min_expected_count": 1,
        "category": "target_specific",
    },
    {
        "question": "Which company entered into a purchase agreement with Gold Express Mines?",
        "expected_symbols": ["MAGE"],
        "expected_companies": ["Magellan Gold"],
        "min_expected_count": 1,
        "category": "target_specific",
    },

    # Transaction wording variants
    {
        "question": "Which companies entered merger agreements?",
        "expected_symbols": ["EXE"],
        "expected_companies": ["Chesapeake Energy"],
        "min_expected_count": 1,
        "category": "merger_variant",
    },
    {
        "question": "Which companies announced all-stock deals?",
        "expected_symbols": ["XOS"],
        "expected_companies": ["Xos"],
        "min_expected_count": 1,
        "category": "deal_variant",
    },

    # False-positive / ambiguity stress tests
    {
        "question": "Which companies amended merger agreements?",
        "expected_symbols": ["TCPC"],
        "expected_companies": ["BlackRock TCP Capital"],
        "min_expected_count": 1,
        "category": "false_positive_stress",
    },
    {
        "question": "Which companies entered cooperation agreements?",
        "expected_symbols": ["SUP"],
        "expected_companies": ["Superior Industries"],
        "min_expected_count": 1,
        "category": "false_positive_stress",
    },

    # Harder / likely weak cases
    {
        "question": "Which filings mentioned stock repurchases?",
        "expected_symbols": [],
        "expected_companies": [],
        "min_expected_count": 0,
        "category": "hard_negative",
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

    entity_hits = set()
    for s in matched_symbols:
        entity_hits.add(normalize_text(s))
    for c in matched_companies:
        entity_hits.add(normalize_text(c))

    return {
        "matched_symbols": matched_symbols,
        "matched_companies": matched_companies,
        "matched_entity_count": len(entity_hits),
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

    entity_hits = set()
    for s in matched_symbols:
        entity_hits.add(normalize_text(s))
    for c in matched_companies:
        entity_hits.add(normalize_text(c))

    return {
        "matched_symbols": matched_symbols,
        "matched_companies": matched_companies,
        "matched_entity_count": len(entity_hits),
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
    confidence = result.get("confidence", {})

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

    if item["min_expected_count"] == 0:
        passed = True
    else:
        passed = (
            answer_score["matched_entity_count"] >= item["min_expected_count"]
            and answer_score["has_citation"]
        )

    return {
        "question": question,
        "category": item.get("category"),
        "task_type": result["task_type"],
        "plan": result["plan"],
        "latency_sec": round(latency_sec, 3),
        "retrieval_symbols": evidence_symbols,
        "retrieval_hit_count": retrieval_hit_count,
        "extraction_score": extraction_score,
        "answer_score": answer_score,
        "confidence": confidence,
        "passed": passed,
        "final_answer": answer,
    }

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_latency = sum(r["latency_sec"] for r in results) / total if total else 0.0
    citation_pass = sum(1 for r in results if r["answer_score"]["has_citation"])
    final_answer_count = sum(1 for r in results if r.get("confidence", {}).get("status") == "final_answer")
    needs_review_count = sum(1 for r in results if r.get("confidence", {}).get("status") == "needs_review")

    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1

    return {
        "total_questions": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "avg_latency_sec": round(avg_latency, 3),
        "citation_rate": round(citation_pass / total, 3) if total else 0.0,
        "final_answer_rate": round(final_answer_count / total, 3) if total else 0.0,
        "needs_review_rate": round(needs_review_count / total, 3) if total else 0.0,
        "by_category": by_category,
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