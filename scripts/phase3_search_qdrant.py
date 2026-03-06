
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


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


def build_filter(
    symbol: Optional[str],
    exchange: Optional[str],
    accession: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Optional[qm.Filter]:
    must: List[qm.FieldCondition] = []

    if symbol:
        must.append(qm.FieldCondition(key="symbol", match=qm.MatchValue(value=symbol)))
    if exchange:
        must.append(qm.FieldCondition(key="exchange", match=qm.MatchValue(value=exchange)))
    if accession:
        must.append(
            qm.FieldCondition(
                key="sec_accession_number", match=qm.MatchValue(value=accession)
            )
        )

    # date filtering is still handled post-retrieval for now
    if date_from or date_to:
        return None

    if not must:
        return None

    return qm.Filter(must=must)


def post_filter_hits_by_date(
    hits: List[qm.ScoredPoint], date_from: Optional[str], date_to: Optional[str]
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


def acquisition_keyword_bonus(query: str, payload: Dict[str, Any]) -> float:
    """
    Lightweight reranking bonus for acquisition-style queries.
    We only apply this if the query itself looks acquisition-related.
    """
    q = normalize_text(query)

    acquisition_triggers = [
        "acquisition",
        "acquire",
        "acquired",
        "acquiring",
        "merger",
        "merge",
        "business combination",
        "definitive agreement",
        "purchase",
        "buyout",
    ]

    if not any(t in q for t in acquisition_triggers):
        return 0.0

    title = normalize_text(str(payload.get("title") or ""))
    text = normalize_text(str(payload.get("text") or ""))

    strong_terms = [
        "acquire",
        "acquisition",
        "merger",
        "business combination",
        "definitive agreement",
        "merge",
        "acquired",
        "all-stock deal",
    ]

    weak_negative_terms = [
        "share repurchase",
        "repurchase program",
        "buyback",
        "extend the date",
        "extension",
    ]

    bonus = 0.0

    for t in strong_terms:
        if t in title:
            bonus += 0.08
        if t in text:
            bonus += 0.03

    for t in weak_negative_terms:
        if t in title:
            bonus -= 0.08
        if t in text:
            bonus -= 0.03

    return bonus


def rerank_hits(query: str, hits: List[qm.ScoredPoint]) -> List[qm.ScoredPoint]:
    """
    Reorder hits using:
    base score + lightweight domain bonus/penalty
    """
    scored: List[Tuple[float, qm.ScoredPoint]] = []

    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        base = float(h.score or 0.0)
        bonus = acquisition_keyword_bonus(query, payload)
        final_score = base + bonus
        scored.append((final_score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


def dedupe_by_accession(hits: List[qm.ScoredPoint]) -> List[qm.ScoredPoint]:
    """
    Keep only the best chunk per SEC filing accession number.
    Assumes hits are already sorted in desired order.
    """
    seen = set()
    out: List[qm.ScoredPoint] = []

    for h in hits:
        payload: Dict[str, Any] = h.payload or {}
        acc = payload.get("sec_accession_number")
        if not acc:
            continue
        if acc in seen:
            continue
        seen.add(acc)
        out.append(h)

    return out


def pretty_print_hit(h: qm.ScoredPoint, show_text_chars: int) -> None:
    p: Dict[str, Any] = h.payload or {}
    text = p.get("text") or ""
    if not isinstance(text, str):
        text = str(text)

    header = (
        f"score={h.score:.4f} | symbol={p.get('symbol')} | "
        f"date={str(p.get('release_dt_utc'))[:10]} | "
        f"accession={p.get('sec_accession_number')} | "
        f"chunk={p.get('chunk_index')} [{p.get('char_start')},{p.get('char_end')}]"
    )
    title = p.get("title")
    if title:
        header += f"\n  title: {title}"

    snippet = text[:show_text_chars].replace("\n", " ").strip()
    print(header)
    print(f"  text: {snippet}")
    print("-" * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="sec_8k_chunks")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text:latest")

    ap.add_argument("--query", help="Your natural language query")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--show_text_chars", type=int, default=450)

    ap.add_argument("--symbol", help="Filter by ticker symbol, e.g. TSLA")
    ap.add_argument("--exchange", help="Filter by exchange, e.g. NASDAQ")
    ap.add_argument("--accession", help="Filter by SEC accession number")
    ap.add_argument("--date_from", help="Filter (post) by date YYYY-MM-DD")
    ap.add_argument("--date_to", help="Filter (post) by date YYYY-MM-DD")

    ap.add_argument("--interactive", action="store_true", help="Prompt for queries in a loop")
    args = ap.parse_args()

    qdrant = QdrantClient(url=args.qdrant_url)

    cols = [c.name for c in qdrant.get_collections().collections]
    if args.collection not in cols:
        raise SystemExit(
            f"Collection '{args.collection}' not found in Qdrant. Available: {cols}"
        )

    def run_query(q: str) -> None:
        q_vec = ollama_embed([q], model=args.embed_model, host=args.ollama_host)[0]

        q_filter = build_filter(
            symbol=args.symbol,
            exchange=args.exchange,
            accession=args.accession,
            date_from=args.date_from,
            date_to=args.date_to,
        )

        # pull a larger candidate pool so reranking + dedupe have room to work
        candidate_limit = max(args.top_k * 6, 24)

        res = qdrant.query_points(
            collection_name=args.collection,
            query=q_vec,
            limit=candidate_limit,
            query_filter=q_filter,
            with_payload=True,
            with_vectors=False,
        )

        hits = res.points
        hits = post_filter_hits_by_date(hits, args.date_from, args.date_to)
        hits = rerank_hits(q, hits)
        hits = dedupe_by_accession(hits)
        hits = hits[: args.top_k]

        print(f"\nQuery: {q}")
        print(f"Returned: {len(hits)} unique filings\n")

        for h in hits:
            pretty_print_hit(h, show_text_chars=args.show_text_chars)

    if args.interactive:
        print("Interactive mode. Type a query and press Enter. Type 'exit' to quit.")
        while True:
            q = input("\n> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            run_query(q)
    else:
        if not args.query:
            raise SystemExit("Provide --query '...' or use --interactive")
        run_query(args.query)


if __name__ == "__main__":
    main()