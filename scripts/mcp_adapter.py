#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any, Callable, Dict

from eventlens_tools import (
    answer_from_evidence,
    candidates_to_json,
    evidence_to_json,
    extract_event_candidates,
    search_sec_filings,
)
from tool_schemas import TOOL_SCHEMAS


def _tool_schema_map() -> Dict[str, Dict[str, Any]]:
    return {tool["name"]: tool for tool in TOOL_SCHEMAS}


def _tool_fn_map() -> Dict[str, Callable[..., Any]]:
    return {
        "search_sec_filings": search_sec_filings,
        "extract_event_candidates": extract_event_candidates,
        "answer_from_evidence": answer_from_evidence,
    }


def list_tools() -> Dict[str, Any]:
    return {
        "tools": TOOL_SCHEMAS
    }


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fns = _tool_fn_map()

    if name not in fns:
        return {
            "ok": False,
            "error": f"Unknown tool: {name}",
        }

    fn = fns[name]

    try:
        result = fn(**arguments)

        if name == "search_sec_filings":
            result = evidence_to_json(result)
        elif name == "extract_event_candidates":
            result = candidates_to_json(result)

        return {
            "ok": True,
            "tool": name,
            "result": result,
        }
    except Exception as e:
        return {
            "ok": False,
            "tool": name,
            "error": str(e),
        }


def main() -> None:
    """
    Minimal MCP-style local adapter over stdin/stdout.

    Request format:
    {"action": "list_tools"}
    {"action": "call_tool", "name": "...", "arguments": {...}}
    """
    import sys

    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"ok": False, "error": "No input provided"}))
        return

    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"Invalid JSON: {e}"}))
        return

    action = req.get("action")

    if action == "list_tools":
        print(json.dumps({"ok": True, **list_tools()}, indent=2))
        return

    if action == "call_tool":
        name = req.get("name")
        arguments = req.get("arguments", {}) or {}
        print(json.dumps(call_tool(name, arguments), indent=2))
        return

    print(json.dumps({"ok": False, "error": f"Unknown action: {action}"}))


if __name__ == "__main__":
    main()