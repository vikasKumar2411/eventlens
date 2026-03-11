from __future__ import annotations

import re
from typing import Any, Dict, List

from eventlens_tools import EvidenceChunk, EventCandidate


def has_valid_source_citations(answer: str) -> bool:
    citations = re.findall(r"\[Source\s+(\d+)\]", answer)
    if not citations:
        return False
    return all(c.isdigit() for c in citations)


def cited_source_ids(answer: str) -> List[int]:
    return [int(x) for x in re.findall(r"\[Source\s+(\d+)\]", answer)]


def compute_confidence(
    *,
    evidence: List[EvidenceChunk],
    candidates: List[EventCandidate],
    answer: str,
) -> Dict[str, Any]:
    if not evidence:
        return {
            "confidence_score": 0.0,
            "status": "needs_review",
            "reason": "no_evidence",
        }

    top_score = evidence[0].score if evidence else 0.0
    strong_evidence_count = sum(1 for e in evidence if e.score >= 1.0)
    candidate_count = len(candidates)
    valid_citations = has_valid_source_citations(answer)
    cited_ids = cited_source_ids(answer)

    max_source_id = max((e.source_id for e in evidence), default=0)
    citation_ids_in_range = all(1 <= cid <= max_source_id for cid in cited_ids) if cited_ids else False

    score = 0.0

    if top_score >= 1.20:
        score += 0.35
    elif top_score >= 1.00:
        score += 0.25
    elif top_score >= 0.80:
        score += 0.15

    if strong_evidence_count >= 2:
        score += 0.25
    elif strong_evidence_count == 1:
        score += 0.15

    if candidate_count >= 2:
        score += 0.20
    elif candidate_count == 1:
        score += 0.10

    if valid_citations and citation_ids_in_range:
        score += 0.20

    score = round(min(score, 1.0), 3)

    if not valid_citations:
        return {
            "confidence_score": score,
            "status": "needs_review",
            "reason": "invalid_or_missing_citations",
        }

    if not citation_ids_in_range:
        return {
            "confidence_score": score,
            "status": "needs_review",
            "reason": "citation_source_out_of_range",
        }

    if top_score < 0.80:
        return {
            "confidence_score": score,
            "status": "needs_review",
            "reason": "weak_top_evidence",
        }

    if candidate_count == 0:
        return {
            "confidence_score": score,
            "status": "needs_review",
            "reason": "no_structured_candidates",
        }

    return {
        "confidence_score": score,
        "status": "final_answer",
        "reason": "sufficient_evidence",
    }