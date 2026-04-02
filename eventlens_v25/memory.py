# eventlens_v25/memory.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from eventlens_v25.run_store import load_all_run_summaries


def load_all_runs(runs_dir: Path | None = None) -> List[Dict[str, Any]]:
    """
    Load all persisted run summaries from disk.
    """
    return load_all_run_summaries(runs_dir=runs_dir)


def filter_runs(
    runs: List[Dict[str, Any]],
    *,
    intent: Optional[str] = None,
    plan: Optional[str] = None,
    final_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter runs by intent / plan / final_status.
    """
    out: List[Dict[str, Any]] = []

    for run in runs:
        if intent is not None and run.get("intent") != intent:
            continue
        if plan is not None and run.get("plan") != plan:
            continue
        if final_status is not None and run.get("final_status") != final_status:
            continue
        out.append(run)

    return out


def filter_runs_by_intent(
    runs: List[Dict[str, Any]],
    intent: str,
) -> List[Dict[str, Any]]:
    return filter_runs(runs, intent=intent)


def _is_successful_run(run: Dict[str, Any]) -> bool:
    """
    For now, treat final_status='answered' as success.
    """
    return run.get("final_status") == "answered"


def _get_last_retry_strategy(run: Dict[str, Any]) -> Optional[str]:
    retry_history = run.get("retry_history", []) or []
    if not retry_history:
        return None

    last = retry_history[-1]
    return last.get("final_retry_strategy")


def compute_retry_strategy_stats(
    runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate performance stats by final retry strategy.

    Example output:
    {
      "narrow_search": {
        "runs": 10,
        "answered": 6,
        "escalated": 4,
        "answer_rate": 0.6,
        "avg_retrieval_attempt": 1.8,
      },
      ...
    }
    """
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "answered": 0,
            "escalated": 0,
            "total_retrieval_attempt": 0,
        }
    )

    for run in runs:
        strategy = _get_last_retry_strategy(run)
        if not strategy:
            continue

        grouped[strategy]["runs"] += 1
        grouped[strategy]["total_retrieval_attempt"] += int(run.get("retrieval_attempt", 0) or 0)

        if run.get("final_status") == "answered":
            grouped[strategy]["answered"] += 1
        elif run.get("final_status") == "escalated":
            grouped[strategy]["escalated"] += 1

    stats: Dict[str, Dict[str, Any]] = {}
    for strategy, vals in grouped.items():
        runs_n = vals["runs"]
        answered = vals["answered"]

        stats[strategy] = {
            "runs": runs_n,
            "answered": answered,
            "escalated": vals["escalated"],
            "answer_rate": round(answered / runs_n, 3) if runs_n else 0.0,
            "avg_retrieval_attempt": round(vals["total_retrieval_attempt"] / runs_n, 3) if runs_n else 0.0,
        }

    return dict(sorted(stats.items(), key=lambda kv: (-kv[1]["answer_rate"], -kv[1]["runs"], kv[0])))


def _rewrite_texts_from_run(run: Dict[str, Any]) -> List[str]:
    rewrite_history = run.get("rewrite_history", []) or []
    texts: List[str] = []

    for item in rewrite_history:
        text = (item.get("rewritten_query") or "").strip().lower()
        if text:
            texts.append(text)

    return texts


def _evidence_top_scores_from_run(run: Dict[str, Any]) -> List[float]:
    trace = run.get("trace", []) or []
    scores: List[float] = []

    for entry in trace:
        if entry.get("node") == "evaluate_evidence_node":
            value = entry.get("top_score")
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue

    return scores


def detect_run_stagnation(
    run: Dict[str, Any],
    *,
    min_score_delta: float = 0.02,
) -> Dict[str, Any]:
    """
    Detect whether a run's retries were stagnant.

    Signals:
    - repeated identical final retry strategy
    - rewritten queries are highly repetitive
    - top_score did not materially improve
    """
    retry_history = run.get("retry_history", []) or []
    strategies = [r.get("final_retry_strategy") for r in retry_history if r.get("final_retry_strategy")]
    unique_strategies = sorted(set(strategies))

    rewrite_texts = _rewrite_texts_from_run(run)
    unique_rewrites = sorted(set(rewrite_texts))

    scores = _evidence_top_scores_from_run(run)
    score_improvement = 0.0
    if len(scores) >= 2:
        score_improvement = scores[-1] - scores[0]

    repeated_same_strategy = len(strategies) >= 2 and len(unique_strategies) == 1
    repeated_same_rewrite = len(rewrite_texts) >= 2 and len(unique_rewrites) < len(rewrite_texts)
    weak_score_improvement = score_improvement < min_score_delta

    stagnation = (
        len(retry_history) >= 1
        and repeated_same_strategy
        and weak_score_improvement
    ) or (
        len(retry_history) >= 1
        and repeated_same_rewrite
        and weak_score_improvement
    )

    return {
        "stagnation": stagnation,
        "retry_count": len(retry_history),
        "unique_strategies": unique_strategies,
        "unique_strategy_count": len(unique_strategies),
        "rewrite_count": len(rewrite_texts),
        "unique_rewrite_count": len(unique_rewrites),
        "score_path": scores,
        "score_improvement": round(score_improvement, 6),
        "repeated_same_strategy": repeated_same_strategy,
        "repeated_same_rewrite": repeated_same_rewrite,
        "weak_score_improvement": weak_score_improvement,
    }


def compute_stagnation_stats(
    runs: List[Dict[str, Any]],
    *,
    min_score_delta: float = 0.02,
) -> Dict[str, Any]:
    """
    Aggregate stagnation patterns across runs.
    """
    total_runs = len(runs)
    retry_runs = 0
    stagnant_runs = 0
    by_strategy: Dict[str, Dict[str, int]] = defaultdict(lambda: {"runs": 0, "stagnant": 0})

    for run in runs:
        retry_history = run.get("retry_history", []) or []
        if retry_history:
            retry_runs += 1

        stagnation_info = detect_run_stagnation(run, min_score_delta=min_score_delta)
        last_strategy = _get_last_retry_strategy(run)

        if stagnation_info["stagnation"]:
            stagnant_runs += 1

        if last_strategy:
            by_strategy[last_strategy]["runs"] += 1
            if stagnation_info["stagnation"]:
                by_strategy[last_strategy]["stagnant"] += 1

    by_strategy_out: Dict[str, Dict[str, Any]] = {}
    for strategy, vals in by_strategy.items():
        runs_n = vals["runs"]
        stagnant = vals["stagnant"]
        by_strategy_out[strategy] = {
            "runs": runs_n,
            "stagnant": stagnant,
            "stagnation_rate": round(stagnant / runs_n, 3) if runs_n else 0.0,
        }

    return {
        "total_runs": total_runs,
        "retry_runs": retry_runs,
        "stagnant_runs": stagnant_runs,
        "stagnation_rate_overall": round(stagnant_runs / retry_runs, 3) if retry_runs else 0.0,
        "by_strategy": dict(sorted(by_strategy_out.items(), key=lambda kv: (-kv[1]["stagnation_rate"], -kv[1]["runs"], kv[0]))),
    }


def get_memory_hint_for_intent(
    runs: List[Dict[str, Any]],
    *,
    intent: str,
    min_runs_per_strategy: int = 2,
) -> Dict[str, Any]:
    """
    Return lightweight memory priors for a given intent.

    This is the object you can later inject into select_retry_strategy_node.
    """
    scoped_runs = filter_runs_by_intent(runs, intent)
    strategy_stats = compute_retry_strategy_stats(scoped_runs)
    stagnation_stats = compute_stagnation_stats(scoped_runs)

    eligible = {
        strategy: stats
        for strategy, stats in strategy_stats.items()
        if stats["runs"] >= min_runs_per_strategy
    }

    best_strategy = None
    if eligible:
        best_strategy = sorted(
            eligible.items(),
            key=lambda kv: (-kv[1]["answer_rate"], kv[1]["avg_retrieval_attempt"], kv[0]),
        )[0][0]

    return {
        "intent": intent,
        "num_runs_for_intent": len(scoped_runs),
        "best_strategy": best_strategy,
        "strategy_stats": strategy_stats,
        "stagnation_stats": stagnation_stats,
    }


def summarize_memory(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Broad summary across all runs.
    """
    intents = sorted({run.get("intent") for run in runs if run.get("intent")})
    by_intent = {
        intent: get_memory_hint_for_intent(runs, intent=intent)
        for intent in intents
    }

    return {
        "total_runs": len(runs),
        "strategy_stats_global": compute_retry_strategy_stats(runs),
        "stagnation_stats_global": compute_stagnation_stats(runs),
        "by_intent": by_intent,
    }