TOOL_SCHEMAS = [
    {
        "name": "search_sec_filings",
        "description": "Search SEC 8-K filings semantically and return ranked evidence chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "collection": {"type": "string"},
                "top_k": {"type": "integer"},
                "candidate_k": {"type": "integer"},
                "max_chunks_per_accession": {"type": "integer"},
                "symbol": {"type": ["string", "null"]},
                "exchange": {"type": ["string", "null"]},
                "accession": {"type": ["string", "null"]},
                "date_from": {"type": ["string", "null"]},
                "date_to": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "symbol": {"type": ["string", "null"]},
                    "company_name": {"type": ["string", "null"]},
                    "accession": {"type": ["string", "null"]},
                    "release_dt_utc": {"type": ["string", "null"]},
                    "chunk_index": {"type": ["integer", "null"]},
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["source_id", "title", "text", "score"],
            },
        },
    },
    {
        "name": "extract_event_candidates",
        "description": "Extract structured acquisition or transaction candidates from evidence chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_chunks": {"type": "array"},
                "event_type": {"type": "string"},
            },
            "required": ["evidence_chunks"],
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "company": {"type": ["string", "null"]},
                    "target": {"type": ["string", "null"]},
                    "source_id": {"type": "integer"},
                    "symbol": {"type": ["string", "null"]},
                    "accession": {"type": ["string", "null"]},
                    "title": {"type": "string"},
                    "evidence_snippet": {"type": "string"},
                },
                "required": ["event_type", "source_id", "title", "evidence_snippet"],
            },
        },
    },
    {
        "name": "answer_from_evidence",
        "description": "Generate a grounded answer using only the supplied SEC filing evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "evidence_chunks": {"type": "array"},
                "evidence_chars": {"type": "integer"},
            },
            "required": ["question", "evidence_chunks"],
        },
        "output_schema": {
            "type": "string"
        },
    },
]