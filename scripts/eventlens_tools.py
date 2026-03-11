from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class EvidenceChunk:
    source_id: int
    symbol: Optional[str]
    company_name: Optional[str]
    accession: Optional[str]
    release_dt_utc: Optional[str]
    chunk_index: Optional[int]
    title: str
    text: str
    score: float
    base_score: float
    bonus_score: float
    quality_score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventCandidate:
    event_type: str
    company: Optional[str]
    target: Optional[str]
    source_id: int
    symbol: Optional[str]
    accession: Optional[str]
    title: str
    evidence_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Ollama helpers
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

def extract_query_focus_terms(query: str) -> List[str]:
    """
    Extract simple high-signal focus terms from the user query.
    This is intentionally lightweight for now.
    """
    q = normalize_text(query)

    stop_terms = {
        "which", "companies", "company", "announced", "acquisition", "acquisitions",
        "acquire", "acquired", "acquiring", "merger", "merge", "merged",
        "business", "combination", "reported", "discussed", "mentioned",
        "filings", "filing", "stock", "repurchases", "repurchase", "did",
        "does", "do", "what", "who", "when", "where", "why", "how", "and",
        "or", "the", "a", "an", "of", "in", "for", "to"
    }

    raw_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9\.\-&]", query)
    cleaned = []
    for term in raw_terms:
        t = normalize_text(term)
        if len(t) < 4:
            continue
        if t in stop_terms:
            continue
        cleaned.append(t)

    # preserve order, dedupe
    seen = set()
    out = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def question_target_bonus(query: str, payload: Dict[str, Any]) -> float:
    """
    Boost chunks that mention specific entities/targets from the question.
    Example: if the question mentions ElectraMeccanica, chunks mentioning it
    in title/text should get promoted.
    """
    focus_terms = extract_query_focus_terms(query)
    if not focus_terms:
        return 0.0

    title = normalize_text(str(payload.get("title") or ""))
    text = normalize_text(str(payload.get("text") or ""))
    company_name = normalize_text(str(payload.get("company_name") or ""))
    symbol = normalize_text(str(payload.get("symbol") or ""))

    bonus = 0.0

    for term in focus_terms:
        if term in title:
            bonus = 0.18
        if term in text:
            bonus = 0.08
        if term in company_name:
            bonus = 0.06
        if term == symbol:
            bonus = 0.05

    return bonus

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
        target_bonus = question_target_bonus(query, payload)
        quality = chunk_evidence_quality_bonus(payload)
        final_score = base + bonus + target_bonus + quality

        payload["_base_score"] = round(base, 6)
        payload["_bonus_score"] = round(bonus, 6)
        payload["_target_bonus_score"] = round(target_bonus, 6)
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
# Tool 1: search_sec_filings
# -----------------------------
def search_sec_filings(
    query: str,
    *,
    collection: str = "sec_8k_chunks",
    qdrant_url: str = "http://localhost:6333",
    ollama_host: str = "http://127.0.0.1:11434",
    embed_model: str = "nomic-embed-text:latest",
    top_k: int = 5,
    candidate_k: int = 30,
    max_chunks_per_accession: int = 2,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    accession: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[EvidenceChunk]:
    qdrant = QdrantClient(url=qdrant_url)

    cols = [c.name for c in qdrant.get_collections().collections]
    if collection not in cols:
        raise ValueError(f"Collection '{collection}' not found in Qdrant. Available: {cols}")

    q_vec = ollama_embed(
        [query],
        model=embed_model,
        host=ollama_host,
    )[0]

    q_filter = build_filter(
        symbol=symbol,
        exchange=exchange,
        accession=accession,
    )

    res = qdrant.query_points(
        collection_name=collection,
        query=q_vec,
        limit=max(candidate_k, top_k),
        query_filter=q_filter,
        with_payload=True,
        with_vectors=False,
    )

    hits = res.points
    hits = post_filter_hits_by_date(hits, date_from, date_to)
    hits = rerank_hits(query, hits)
    hits = select_best_chunk_per_symbol(hits, max_symbols=top_k)
    hits = cap_chunks_per_accession(
        hits,
        max_chunks_per_accession=max_chunks_per_accession,
    )
    hits = hits[:top_k]

    evidence: List[EvidenceChunk] = []
    for i, h in enumerate(hits, 1):
        p: Dict[str, Any] = h.payload or {}
        evidence.append(
            EvidenceChunk(
                source_id=i,
                symbol=p.get("symbol"),
                company_name=p.get("company_name"),
                accession=p.get("sec_accession_number"),
                release_dt_utc=p.get("release_dt_utc"),
                chunk_index=p.get("chunk_index"),
                title=str(p.get("title") or ""),
                text=str(p.get("text") or ""),
                score=float(h.score or 0.0),
                base_score=float(p.get("_base_score") or 0.0),
                bonus_score=float(p.get("_bonus_score") or 0.0),
                quality_score=float(p.get("_quality_score") or 0.0),
                metadata=p,
            )
        )

    return evidence


# -----------------------------
# Tool 2: extract_event_candidates
# -----------------------------
ACQUISITION_PATTERNS = [
    r"\bacquire\b",
    r"\bacquires\b",
    r"\bacquired\b",
    r"\bacquisition\b",
    r"\bmerger\b",
    r"\bmerge\b",
    r"\ball-stock deal\b",
    r"\bbusiness combination\b",
]


def looks_like_acquisition(text: str) -> bool:
    text_l = text.lower()
    return any(re.search(pat, text_l) for pat in ACQUISITION_PATTERNS)

def heuristic_extract_company_and_target(title: str) -> tuple[Optional[str], Optional[str]]:
    title_clean = re.sub(r"\s+", " ", title).strip()

    patterns = [
        r"^(?P<company>.+?)\s+to\s+Acquire\s+(?P<target>.+?)(?:\s+for\b|\s+in\b|\s+through\b|$)",
        r"^(?P<company>.+?)\s+to\s+acquire\s+(?P<target>.+?)(?:\s+for\b|\s+in\b|\s+through\b|$)",
        r"^(?P<company>.+?)\s+Acquires\s+(?P<target>.+?)(?:\s+for\b|\s+in\b|\s+through\b|$)",
        r"^(?P<company>.+?)\s+acquires\s+(?P<target>.+?)(?:\s+for\b|\s+in\b|\s+through\b|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, title_clean, flags=re.IGNORECASE)
        if m:
            return m.group("company").strip(), m.group("target").strip()

    return None, None


def is_strong_acquisition_candidate(chunk: EvidenceChunk) -> bool:
    title = normalize_text(chunk.title or "")
    text = normalize_text(chunk.text or "")

    strong_title_phrases = [
        "to acquire",
        "acquires",
        "acquired",
        "acquisition of",
        "merger deal",
        "merger agreement",
        "business combination",
        "all-stock deal",
    ]

    strong_text_phrases = [
        "will acquire",
        "agreed to acquire",
        "announced the acquisition",
        "entered into a definitive agreement",
        "definitive merger agreement",
        "acquisition of",
        "purchase agreement",
    ]

    weak_negative_title_phrases = [
        "appoints",
        "board",
        "cooperation agreement",
        "repurchase",
        "buyback",
        "amendment",
    ]

    weak_negative_text_phrases = [
        "extraordinary transaction involving the company",
        "sale or acquisition of material assets",
        "standstill period",
        "vote its shares",
        "board recommendations",
    ]

    if any(p in title for p in weak_negative_title_phrases):
        return False

    if any(p in text for p in weak_negative_text_phrases):
        return False

    strong_title = any(p in title for p in strong_title_phrases)
    strong_text = any(p in text for p in strong_text_phrases)

    return strong_title or strong_text


def extract_event_candidates(
    evidence_chunks: List[EvidenceChunk],
    *,
    event_type: str = "acquisition",
) -> List[EventCandidate]:
    candidates: List[EventCandidate] = []

    for chunk in evidence_chunks:
        joined = f"{chunk.title}\n{chunk.text}"

        if event_type == "acquisition":
            if not looks_like_acquisition(joined):
                continue
            if not is_strong_acquisition_candidate(chunk):
                continue

            company, target = heuristic_extract_company_and_target(chunk.title)

            candidates.append(
                EventCandidate(
                    event_type="acquisition",
                    company=company or chunk.company_name or chunk.symbol,
                    target=target,
                    source_id=chunk.source_id,
                    symbol=chunk.symbol,
                    accession=chunk.accession,
                    title=chunk.title,
                    evidence_snippet=chunk.text[:300].replace("\n", " "),
                )
            )

    return candidates


# -----------------------------
# Tool 3: answer_from_evidence
# -----------------------------
def build_evidence_context(evidence_chunks: List[EvidenceChunk], text_chars: int = 1200) -> str:
    blocks: List[str] = []

    for chunk in evidence_chunks:
        snippet = chunk.text.replace("\n", " ").strip()[:text_chars]

        block = (
            f"[Source {chunk.source_id}]\n"
            f"symbol: {chunk.symbol}\n"
            f"company_name: {chunk.company_name}\n"
            f"accession: {chunk.accession}\n"
            f"release_dt_utc: {chunk.release_dt_utc}\n"
            f"title: {chunk.title}\n"
            f"chunk_index: {chunk.chunk_index}\n"
            f"text: {snippet}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_answer_prompt(question: str, evidence: str) -> str:
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


def answer_from_evidence(
    question: str,
    evidence_chunks: List[EvidenceChunk],
    *,
    llm_model: str = "qwen2.5:7b-instruct",
    ollama_host: str = "http://127.0.0.1:11434",
    evidence_chars: int = 1200,
) -> str:
    evidence = build_evidence_context(evidence_chunks, text_chars=evidence_chars)
    prompt = build_answer_prompt(question, evidence)
    return ollama_generate(prompt, model=llm_model, host=ollama_host)


# -----------------------------
# Serializers
# -----------------------------
def evidence_to_json(evidence_chunks: List[EvidenceChunk]) -> List[Dict[str, Any]]:
    return [chunk.to_dict() for chunk in evidence_chunks]


def candidates_to_json(candidates: List[EventCandidate]) -> List[Dict[str, Any]]:
    return [candidate.to_dict() for candidate in candidates]