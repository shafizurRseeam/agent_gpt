"""
evaluation/timing_eval.py
═══════════════════════════════════════════════════════════════════════════
Per-stage timing breakdown for the PrivScope sanitization pipeline,
across multiple local LLM backbones.

Measures wall-clock time for:
  Stage 1  — Span extraction         (regex + spaCy, no LLM)
  Stage 2  — Carryover control       (1 LLM call: keep/drop decision)
  Stage 3b — Span abstraction        (1 batched LLM call for all kept spans)
  Total    — Stage 1 + Stage 2 + Stage 3b  (sanitization latency only)

Note: LC payload construction is excluded — not part of the sanitization
pipeline and excluded from latency in run_evaluation.py.

Usage (from project root):
    uv run python evaluation/timing_eval.py
    uv run python evaluation/timing_eval.py --models llama3.2 qwen2.5:7b
    uv run python evaluation/timing_eval.py --n-tasks 10
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agents.hybrid_agent import HybridAgent
from privacy.privscope   import PrivScope

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0
    _TOK_SOURCE = "tiktoken cl100k_base"
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text.split()) if text else 0
    _TOK_SOURCE = "word count"

_TASKS_FILE = _ROOT / "task_generated" / "task_prompts.json"
_OUT_FILE   = _ROOT / "evaluation" / "timing_eval.json"

_DEFAULT_MODELS = ["llama3.2", "phi3", "mistral", "qwen2.5:7b", "llama3.1:8b"]
_MODEL_LABELS = {
    "llama3.2":    "llama3.2    (3B)",
    "phi3":        "phi3        (3.8B)",
    "mistral":     "mistral     (7B)",
    "qwen2.5:7b":  "qwen2.5:7b  (7B)",
    "llama3.1:8b": "llama3.1:8b (8B)",
}

W = 78


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: list) -> float:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _fmt(s: float) -> str:
    return f"{s * 1000:7.1f} ms"


# ── Per-model timing run ──────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    tasks:      List[dict],
) -> Optional[Dict]:
    """
    Run PrivScope sanitization for every task with one local model.
    Returns per-task timing records and summary averages, or None on load failure.
    """
    try:
        agent = HybridAgent(local_model=model_name)
        print(f"  Loaded: {agent.local.model}")
    except Exception as e:
        print(f"  [SKIP] Could not load {model_name}: {e}")
        return None

    ps      = PrivScope(local_llm=agent.local)
    profile = agent.state.get("user_profile", {})
    traces  = agent.state.get("memory_traces", [])

    print(f"\n  {'#':>3}  {'Stage 1':>9}  {'Stage 2':>9}  {'Stage 3b':>10}  {'Total':>9}  {'Spans':>5}  {'Tokens':>6}")
    print(f"  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*6}")

    per_task = []
    for i, task in enumerate(tasks):
        prompt = task.get("prompt", "").strip()

        inferred_prefs   = agent._lc_infer_preferences(prompt)
        _, naive_payload = agent._lc_reason_cloud_query(prompt, inferred_prefs)

        cloud_payload, trace = ps.sanitize_with_trace(naive_payload, profile, prompt, traces)

        timings    = trace.get("stage_timings", {})
        t_s1       = timings.get("stage1_s",  0.0)
        t_s2       = timings.get("stage2_s",  0.0)
        t_s3b      = timings.get("stage3b_s", 0.0)
        spans_kept = len(trace.get("stage2", {}).get("kept", []))
        total      = t_s1 + t_s2 + t_s3b
        tokens     = _count_tokens(cloud_payload)

        print(
            f"  {i+1:>3}  "
            f"{_fmt(t_s1)}  "
            f"{_fmt(t_s2)}  "
            f"{_fmt(t_s3b)}  "
            f"{_fmt(total)}  "
            f"{spans_kept:>5}  "
            f"{tokens:>6}"
        )

        per_task.append({
            "task_idx":   i,
            "stage1_s":   t_s1,
            "stage2_s":   t_s2,
            "stage3b_s":  t_s3b,
            "total_s":    round(total, 4),
            "spans_kept": spans_kept,
            "tokens":     tokens,
        })

    keys = ["stage1_s", "stage2_s", "stage3b_s", "total_s"]
    avgs = {k: _mean([r[k] for r in per_task]) for k in keys}
    avgs["avg_spans_kept"] = _mean([r["spans_kept"] for r in per_task])
    avgs["avg_tokens"]     = _mean([r["tokens"]     for r in per_task])

    return {"model": model_name, "per_task": per_task, "averages": avgs}


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: List[Dict]) -> None:
    print(f"\n{'═' * W}")
    print(f"  TIMING SUMMARY  —  across {len(all_results)} model(s)")
    print(f"{'═' * W}")
    print(
        f"\n  {'Model':<20}  {'Stage1 ms':>10}  {'Stage2 ms':>10}  "
        f"{'Stage3b ms':>11}  {'Total ms':>9}  {'Spans':>5}  {'Tokens':>6}  {'S2 %':>6}  {'S3b %':>6}"
    )
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")

    for r in all_results:
        label = _MODEL_LABELS.get(r["model"], r["model"])
        a     = r["averages"]
        tot   = a["total_s"]
        s2_pct  = (a["stage2_s"]  / tot * 100) if tot > 0 else 0
        s3b_pct = (a["stage3b_s"] / tot * 100) if tot > 0 else 0
        first = r.get("first_run", False)
        print(
            f"  {label:<20}  "
            f"{a['stage1_s']*1000:>9.1f}{'*' if first else ' '}  "
            f"{a['stage2_s']*1000:>10.1f}  "
            f"{a['stage3b_s']*1000:>11.1f}  "
            f"{tot*1000:>9.1f}  "
            f"{a['avg_spans_kept']:>5.1f}  "
            f"{int(round(a['avg_tokens'])):>6}  "
            f"{s2_pct:>5.1f}%  "
            f"{s3b_pct:>5.1f}%"
        )

    print(f"\n  Stage 1  = span extraction (regex+spaCy, no LLM)")
    print(f"  Stage 2  = carryover control (1 LLM keep/drop call)")
    print(f"  Stage 3b = span abstraction  (1 batched LLM call)")
    print(f"  Total    = sanitization latency only (excludes LC payload construction)")
    print(f"  Tokens   = avg token count of final cloud payload  ({_TOK_SOURCE})")
    print(f"  *        = Stage 1 includes spaCy cold-load (~500ms one-time cost)")
    print(f"{'═' * W}\n")


# ── JSON save ─────────────────────────────────────────────────────────────────

def save_results(all_results: List[Dict], n_tasks: int, models: List[str]) -> None:
    payload = {
        "metadata": {
            "n_tasks":   n_tasks,
            "models":    models,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "results": {r["model"]: {"averages": r["averages"], "per_task": r["per_task"]} for r in all_results},
    }
    _OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Results saved → {_OUT_FILE.relative_to(_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Per-stage timing breakdown for PrivScope across local models.")
    ap.add_argument("--models", nargs="+", default=_DEFAULT_MODELS, metavar="MODEL",
                    help="Local Ollama models to test. Default: all five.")
    ap.add_argument("--n-tasks", type=int, default=None)
    args = ap.parse_args()

    data   = _load_json(_TASKS_FILE)
    tasks  = data["task_prompts"]
    n_eval = min(args.n_tasks, len(tasks)) if args.n_tasks else len(tasks)
    tasks  = tasks[:n_eval]

    print(f"\nPrivScope sanitization timing — {n_eval} tasks x {len(args.models)} model(s)")

    all_results = []
    for i, model_name in enumerate(args.models):
        print(f"\n{'─' * W}")
        print(f"  Model: {model_name}" + (" (first run — Stage 1 includes spaCy cold-load)" if i == 0 else ""))
        print(f"{'─' * W}")
        result = evaluate_model(model_name, tasks)
        if result is not None:
            result["first_run"] = (i == 0)
            all_results.append(result)

    if all_results:
        print_summary_table(all_results)
        save_results(all_results, n_eval, args.models)


if __name__ == "__main__":
    main()
