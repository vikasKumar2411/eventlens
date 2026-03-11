#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# -----------------------------
# Embeddings + generation
# -----------------------------
def ollama_embed(texts: List[str], model: str, host: str) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        r = requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=180,
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


def ollama_generate(prompt: str, model: str, host: str) -> str:
    r = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# -----------------------------
# Retrieval helpers
# -----------------------------
def build_filter(
    symbol: Optional[str],
    exchange: Optional[str],
    accession: Optional[str],
) -> Optional[qm.Filter]:
    must: List[qm.FieldCondition] = []

    if symbol:
        must.append(qm.FieldCondition(key="symbol", match=qm.MatchValue(value=symbol)))
    if exchange:
        must.append(qm.FieldCondition(key="exchange", match=qm.MatchValue(value=exchange)))
    if accession:
        must.append(
            qm.FieldCondition(
                key="sec_accession_number",
                match=qm.MatchValue(value=accession),
            )
        )

    if not must:
        return None

    return qm.Filter(must=must)


def post_filter_hits_by_date(
    hits: List[qm.ScoredPoint],
    date_from: Optional[str],
    date_to: Optional[str],
) -> List[qm.ScoredPoint]:
    if not date_from and not date_to:
        return hits

    out: List[qm.ScoredPoint] = []
    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        dt = payload.get("release_dt_utc")
        if not isinstance(dt, str) or len(dt) < 10:
            continue
        d = dt[:10]
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        out.append(h)
    return out


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def acquisition_keyword_bonus(query: str, payload: Dict[str, Any]) -> float:
    q = normalize_text(query)
    title = normalize_text(str(payload.get("title") or ""))
    text = normalize_text(str(payload.get("text") or ""))

    acquisition_query_triggers = [
        "acquisition",
        "acquire",
        "acquired",
        "acquiring",
        "merger",
        "merge",
        "business combination",
        "purchase",
        "buyout",
        "takeover",
    ]

    if not contains_any(q, acquisition_query_triggers):
        return 0.0

    strong_title_phrases = [
        "to acquire",
        "announces acquisition",
        "announced acquisition",
        "acquisition of",
        "enters definitive agreement to acquire",
        "definitive agreement to acquire",
        "merger agreement",
        "major merger deal",
        "business combination agreement",
        "to merge with",
        "acquired",
        "all-stock deal",
    ]

    strong_text_phrases = [
        "entered into a definitive agreement",
        "definitive agreement to acquire",
        "agreed to acquire",
        "announced that it will acquire",
        "announced the acquisition of",
        "acquisition of",
        "will acquire",
        "acquired",
        "merger agreement",
        "business combination agreement",
        "merge with",
    ]

    weak_negative_title_phrases = [
        "post-merger",
        "extension",
        "prepayment",
        "strategic flexibility",
        "share repurchase",
        "repurchase program",
        "buyback",
        "amendment",
        "supplement",
    ]

    weak_negative_text_phrases = [
        "post-merger",
        "extension",
        "share repurchase",
        "repurchase program",
        "buyback",
        "amendment",
        "supplement",
    ]

    boilerplate_phrases = [
        "forward-looking statements",
        "safe harbor",
        "special note regarding forward-looking statements",
        "cautionary statement",
        "certain statements herein",
        "exhibit 99.1",
        "signature pursuant to the requirements",
        "/s/",
    ]

    bonus = 0.0

    for p in strong_title_phrases:
        if p in title:
            bonus += 0.16

    for p in strong_text_phrases:
        if p in text:
            bonus += 0.06

    for p in weak_negative_title_phrases:
        if p in title:
            bonus -= 0.14

    for p in weak_negative_text_phrases:
        if p in text:
            bonus -= 0.05

    for p in boilerplate_phrases:
        if p in text:
            bonus -= 0.08

    if contains_any(title, strong_title_phrases) and contains_any(text, strong_text_phrases):
        bonus += 0.08

    return bonus


def chunk_evidence_quality_bonus(payload: Dict[str, Any]) -> float:
    title = normalize_text(str(payload.get("title") or ""))
    text = normalize_text(str(payload.get("text") or ""))

    strong_text_phrases = [
        "entered into a definitive agreement",
        "definitive agreement",
        "definitive merger agreement",
        "agreed to acquire",
        "will acquire",
        "announced the acquisition",
        "announced acquisition",
        "acquisition of",
        "merger agreement",
        "business combination agreement",
        "to acquire",
        "acquired",
        "merge with",
    ]

    moderately_good_phrases = [
        "following the close of the transaction",
        "subject to shareholder approval",
        "subject to customary closing conditions",
        "all-stock deal",
        "cash and stock",
        "purchase price",
        "transaction value",
    ]

    bad_boilerplate_phrases = [
        "forward-looking statements",
        "safe harbor",
        "special note regarding forward-looking statements",
        "cautionary statement",
        "certain statements herein",
        "signature pursuant to the requirements",
        "exhibit 99.1",
        "ex-99.1",
        "/s/",
        "title: president and ceo",
        "title: chief executive officer",
    ]

    very_bad_section_markers = [
        "anticipated, believes, could, estimates, expects, forecasts",
        "statements may be identified by words such as",
        "pursuant to the requirements of the securities exchange act",
    ]

    score = 0.0

    for p in strong_text_phrases:
        if p in text:
            score += 0.12

    for p in moderately_good_phrases:
        if p in text:
            score += 0.04

    for p in bad_boilerplate_phrases:
        if p in text:
            score -= 0.14

    for p in very_bad_section_markers:
        if p in text:
            score -= 0.18

    # If the title is strong but text is weak, still allow a small rescue bonus.
    strong_title_phrases = [
        "to acquire",
        "acquisition of",
        "announced acquisition",
        "definitive agreement to acquire",
        "merger deal",
        "merger agreement",
        "all-stock deal",
    ]
    if contains_any(title, strong_title_phrases):
        score += 0.05

    return score


def rerank_hits(query: str, hits: List[qm.ScoredPoint]) -> List[qm.ScoredPoint]:
    rescored_hits: List[qm.ScoredPoint] = []

    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        base = float(h.score or 0.0)
        bonus = acquisition_keyword_bonus(query, payload)
        quality = chunk_evidence_quality_bonus(payload)
        final_score = base + bonus + quality

        payload["_base_score"] = round(base, 6)
        payload["_bonus_score"] = round(bonus, 6)
        payload["_quality_score"] = round(quality, 6)
        payload["_final_score"] = round(final_score, 6)
        h.payload = payload
        h.score = final_score

        rescored_hits.append(h)

    rescored_hits.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return rescored_hits


def select_best_chunk_per_symbol(
    hits: List[qm.ScoredPoint],
    max_symbols: Optional[int] = None,
) -> List[qm.ScoredPoint]:
    best_by_symbol: Dict[str, qm.ScoredPoint] = {}
    anonymous_hits: List[qm.ScoredPoint] = []

    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        sym = payload.get("symbol")

        if not sym:
            anonymous_hits.append(h)
            continue

        prev = best_by_symbol.get(sym)
        if prev is None or float(h.score or 0.0) > float(prev.score or 0.0):
            best_by_symbol[sym] = h

    selected = list(best_by_symbol.values()) + anonymous_hits
    selected.sort(key=lambda x: float(x.score or 0.0), reverse=True)

    if max_symbols is not None:
        selected = selected[:max_symbols]

    return selected


def cap_chunks_per_accession(
    hits: List[qm.ScoredPoint],
    max_chunks_per_accession: int = 2,
) -> List[qm.ScoredPoint]:
    counts = defaultdict(int)
    out: List[qm.ScoredPoint] = []

    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        acc = payload.get("sec_accession_number")
        if not acc:
            out.append(h)
            continue

        if counts[acc] >= max_chunks_per_accession:
            continue

        counts[acc] += 1
        out.append(h)

    return out


# -----------------------------
# Evidence formatting
# -----------------------------
def build_evidence_context(hits: List[qm.ScoredPoint], text_chars: int = 1200) -> str:
    blocks: List[str] = []

    for i, h in enumerate(hits, 1):
        p: Dict[str, Any] = h.payload or {}
        snippet = str(p.get("text") or "").replace("\n", " ").strip()[:text_chars]

        block = (
            f"[Source {i}]\n"
            f"symbol: {p.get('symbol')}\n"
            f"company_name: {p.get('company_name')}\n"
            f"accession: {p.get('sec_accession_number')}\n"
            f"release_dt_utc: {p.get('release_dt_utc')}\n"
            f"title: {p.get('title')}\n"
            f"chunk_index: {p.get('chunk_index')}\n"
            f"text: {snippet}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(question: str, evidence: str) -> str:
    return f"""
You are a financial filings assistant working only from retrieved SEC 8-K evidence.

Use ONLY the evidence below.
Do not invent facts.
If the evidence is insufficient or ambiguous, say so clearly.

Task:
Identify EVERY company in the evidence that directly announced an acquisition, merger, business combination, or definitive transaction agreement.

Rules:
- Extract all supported companies, not just the single strongest one.
- Prefer direct transaction announcements over commentary, exhibits, amendments, forward-looking boilerplate, extensions, signatures, or post-merger discussion.
- A strong acquisition-related title is valid evidence when consistent with the filing metadata and snippet.
- If multiple sources support different companies, include all of them.
- Do not exclude a company only because another source appears stronger.
- Every listed company must have at least one citation.
- Do not use companies that are only loosely related.

Return format exactly:

Answer:
- <company_name or symbol>: <one short reason> [Source X]
- <company_name or symbol>: <one short reason> [Source Y]

Sources:
- Source X, symbol: ..., accession: ..., chunk_index: ...
- Source Y, symbol: ..., accession: ..., chunk_index: ...

User question:
{question}

Evidence:
{evidence}
""".strip()


def print_retrieved_hits(hits: List[qm.ScoredPoint], show_text_chars: int = 220) -> None:
    print("\n=== RETRIEVED EVIDENCE ===\n")
    for i, h in enumerate(hits, 1):
        p: Dict[str, Any] = h.payload or {}
        txt = str(p.get("text") or "").replace("\n", " ")[:show_text_chars]
        base = p.get("_base_score")
        bonus = p.get("_bonus_score")
        quality = p.get("_quality_score")
        final = p.get("_final_score")

        print(
            f"[Source {i}] score={float(h.score or 0.0):.4f} "
            f"(base={base}, bonus={bonus}, quality={quality}, final={final}) | "
            f"symbol={p.get('symbol')} | accession={p.get('sec_accession_number')} | "
            f"chunk={p.get('chunk_index')}"
        )
        print(f"title: {p.get('title')}")
        print(f"text: {txt}")
        print("-" * 100)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="sec_8k_chunks")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")

    ap.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text:latest")
    ap.add_argument("--llm_model", default="llama3:latest")

    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--candidate_k", type=int, default=30)
    ap.add_argument("--max_chunks_per_accession", type=int, default=2)

    ap.add_argument("--symbol")
    ap.add_argument("--exchange")
    ap.add_argument("--accession")
    ap.add_argument("--date_from")
    ap.add_argument("--date_to")

    ap.add_argument("--evidence_chars", type=int, default=1200)
    ap.add_argument("--show_retrieved", action="store_true")
    args = ap.parse_args()

    qdrant = QdrantClient(url=args.qdrant_url)

    cols = [c.name for c in qdrant.get_collections().collections]
    if args.collection not in cols:
        raise SystemExit(
            f"Collection '{args.collection}' not found in Qdrant. Available: {cols}"
        )

    q_vec = ollama_embed(
        [args.question],
        model=args.embed_model,
        host=args.ollama_host,
    )[0]

    q_filter = build_filter(
        symbol=args.symbol,
        exchange=args.exchange,
        accession=args.accession,
    )

    res = qdrant.query_points(
        collection_name=args.collection,
        query=q_vec,
        limit=max(args.candidate_k, args.top_k),
        query_filter=q_filter,
        with_payload=True,
        with_vectors=False,
    )

    hits = res.points
    hits = post_filter_hits_by_date(hits, args.date_from, args.date_to)
    hits = rerank_hits(args.question, hits)
    hits = select_best_chunk_per_symbol(hits, max_symbols=args.top_k)
    hits = cap_chunks_per_accession(
        hits,
        max_chunks_per_accession=args.max_chunks_per_accession,
    )
    hits = hits[: args.top_k]

    if not hits:
        print("No evidence retrieved.")
        return

    if args.show_retrieved:
        print_retrieved_hits(hits)

    evidence = build_evidence_context(hits, text_chars=args.evidence_chars)
    prompt = build_prompt(args.question, evidence)
    answer = ollama_generate(prompt, model=args.llm_model, host=args.ollama_host)

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()