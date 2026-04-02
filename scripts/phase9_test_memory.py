#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eventlens_v25.memory import (
    load_all_runs,
    summarize_memory,
    get_memory_hint_for_intent,
)


def main():
    runs = load_all_runs()

    print("\n=== RUN COUNT ===")
    print(len(runs))

    print("\n=== GLOBAL MEMORY SUMMARY ===")
    pprint(summarize_memory(runs))

    print("\n=== ACQUISITION MEMORY HINT ===")
    pprint(get_memory_hint_for_intent(runs, intent="acquisition"))


if __name__ == "__main__":
    main()