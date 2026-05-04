"""
evaluation/run_leakage_ablation.py
════════════════════════════════════════════════════════════════════════════════
Leakage ablation experiment — measures RLR and RLRatio across increasing
amounts of pre-existing history (S^hist).

Experiment design:
  For each preloaded trace size in [0, 5, 10, 20, 30, 40, 50]:
    1. Set S^hist baseline = working_trace_preloaded_{size}.json
    2. Reset working_trace.json to empty
    3. Run all tasks; for each task and method compute RLR + RLRatio
       (no cloud calls — leakage is payload-only)
    4. Save per-task results (including naive_payload and sanitized payload)
    5. Reset working_trace.json before next size

Goal: show whether more pre-existing history → higher residual leakage.

Usage (from project root agent_gpt/):
    uv run python evaluation/run_leakage_ablation.py
    uv run python evaluation/run_leakage_ablation.py --n-tasks 20
    uv run python evaluation/run_leakage_ablation.py --n-tasks 10 --out evaluation/my_ablation.json
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agents.hybrid_agent import HybridAgent
from privacy.privscope   import PrivScope
import privacy.presidio  as presidio_baseline
import privacy.pep       as pep_baseline

# ── Canonical file paths ──────────────────────────────────────────────────────
_TASKS_FILE      = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_FILE    = _ROOT / "state" / "profile_state.json"
_TRACE_FILE      = _ROOT / "state" / "working_trace.json"
_RESULTS_FILE    = Path(__file__).resolve().parent / "run_leakage_ablation.json"

_METHODS             = ("naive", "privscope", "presidio", "pep")
_PRELOADED_SIZES     = (0, 5, 10, 20, 30, 40, 50)
_PRELOADED_TEMPLATE  = "working_trace_preloaded_{size}.json"


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            sys.exit(f"[ERROR] Required file not found: {path}")
        return None
    raw = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, OSError):
            continue
    if raw is None:
        sys.exit(f"[ERROR] Could not read {path} with any supported encoding.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] Cannot parse JSON in {path}: {exc}")


def load_tasks(path: Path) -> List[dict]:
    data = _load_json(path)
    if not isinstance(data, dict) or "task_prompts" not in data:
        sys.exit(f"[ERROR] {path} must contain a top-level 'task_prompts' list.")
    tasks = data["task_prompts"]
    if not tasks:
        sys.exit(f"[ERROR] 'task_prompts' is empty in {path}")
    return tasks


def load_profile_facts(path: Path) -> List[str]:
    data    = _load_json(path)
    profile = data.get("user_profile", {}) if isinstance(data, dict) else {}
    facts   = []
    for v in profile.values():
        s = str(v).strip() if v is not None else ""
        if len(s) >= 3:
            facts.append(s)
    return facts


def _extract_strings(obj, min_len: int = 3) -> List[str]:
    out = []
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= min_len:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_strings(v, min_len))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_extract_strings(item, min_len))
    return out


def load_trace_facts(runtime_path: Path, preloaded_path: Optional[Path] = None) -> List[str]:
    def _facts_from_file(p: Path) -> List[str]:
        data    = _load_json(p, required=False)
        if data is None:
            return []
        entries = data.get("memory_traces", []) if isinstance(data, dict) else []
        out     = []
        for entry in entries:
            if isinstance(entry, dict):
                out.extend(_extract_strings(entry.get("data")))
        return out

    all_facts = _facts_from_file(preloaded_path) if preloaded_path else []
    all_facts += _facts_from_file(runtime_path)

    seen:   set        = set()
    deduped: List[str] = []
    for f in all_facts:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ════════════════════════════════════════════════════════════════════════════════
# TRACE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _reset_trace() -> None:
    """Clear working_trace.json to empty."""
    _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_FILE.write_text(json.dumps({"memory_traces": []}, indent=2), encoding="utf-8")


def _append_trace(entry: dict) -> None:
    data: dict = {"memory_traces": []}
    if _TRACE_FILE.exists():
        try:
            data = json.loads(_TRACE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data.setdefault("memory_traces", []).append(entry)
    _TRACE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _reload_agent_traces(agent: HybridAgent, preloaded_path: Path) -> None:
    """Refresh agent.state['memory_traces'] from preloaded + runtime trace."""
    traces: list = []
    pre     = _load_json(preloaded_path, required=False)
    if pre:
        traces.extend(pre.get("memory_traces", []))
    runtime = _load_json(_TRACE_FILE, required=False)
    if runtime:
        traces.extend(runtime.get("memory_traces", []))
    agent.state["memory_traces"] = traces


# ════════════════════════════════════════════════════════════════════════════════
# LEAKAGE DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def leaking_facts(facts: List[str], payload: str) -> List[str]:
    if not payload or not facts:
        return []
    pl = payload.lower()
    return [f for f in facts if len(f) >= 3 and f.lower() in pl]


# ════════════════════════════════════════════════════════════════════════════════
# PAYLOAD CONSTRUCTION & METHOD APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

def build_naive_payload(agent: HybridAgent, task_prompt: str) -> Tuple[str, str]:
    inferred_prefs          = agent._lc_infer_preferences(task_prompt)
    reasoning, naive_payload = agent._lc_reason_cloud_query(task_prompt, inferred_prefs)
    return naive_payload, reasoning


def apply_method(
    method:        str,
    naive_payload: str,
    task_prompt:   str,
    profile:       dict,
    traces:        list,
    ps:            PrivScope,
    local_llm,
) -> Tuple[str, float, dict]:
    if method == "naive":
        return naive_payload, 0.0, {"method": "naive"}

    t0 = time.perf_counter()

    if method == "privscope":
        payload, trace = ps.sanitize_with_trace(naive_payload, profile, task_prompt, traces)
    elif method == "presidio":
        payload, trace = presidio_baseline.sanitize_with_trace(naive_payload, profile, task_prompt, traces)
    elif method == "pep":
        payload, trace = pep_baseline.sanitize_with_trace(naive_payload, profile, task_prompt, traces, local_llm)
    else:
        raise ValueError(f"Unknown method: {method}")

    return payload, time.perf_counter() - t0, trace


# ════════════════════════════════════════════════════════════════════════════════
# PER-TASK EVALUATION (HistLS only)
# HistLS = |Reveal(S^hist, P^m)| / |S^hist|
# Denominator is S^hist only (not S^state = S^prof ∪ S^hist) so the metric
# isolates carryover leakage as history size grows without S^prof diluting it.
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_task(
    task_idx:       int,
    task:           dict,
    agent:          HybridAgent,
    ps:             PrivScope,
    profile_facts:  List[str],
    history_facts:  List[str],   # S^hist accumulated from prior tasks this run
    trace_facts:    List[str],   # S^hist from preloaded file (static for this run)
) -> dict:
    task_prompt = task.get("prompt", "").strip()
    domain      = task.get("domain", "")
    seed_id     = task.get("seed_id",    f"seed_{task_idx:04d}")
    variant_id  = task.get("variant_id", 0)

    # S^hist only — profile excluded from both numerator and denominator
    hist_facts = list(dict.fromkeys(history_facts + trace_facts))
    n_hist     = len(hist_facts)

    print(f"  [{task_idx+1:>4}] {seed_id} v{variant_id}  {domain}", end="  ", flush=True)

    naive_payload, _ = build_naive_payload(agent, task_prompt)
    profile_dict     = agent.state.get("user_profile", {})
    memory_traces    = agent.state.get("memory_traces", [])

    method_results: Dict[str, dict] = {}

    for method in _METHODS:
        payload, latency, trace = apply_method(
            method, naive_payload, task_prompt,
            profile_dict, memory_traces, ps, agent.local,
        )

        leaked_hist = leaking_facts(hist_facts, payload)
        HistLS      = round(len(leaked_hist) / n_hist, 4) if n_hist else 0.0

        method_results[method] = {
            "payload":     payload,
            "HistLS":      HistLS,
            "leaked_hist": leaked_hist,
            "latency_s":   round(latency, 6),
        }
        print(".", end="", flush=True)

    print()

    return {
        "task_idx":    task_idx,
        "seed_id":     seed_id,
        "variant_id":  variant_id,
        "domain":      domain,
        "prompt":      task_prompt,
        "naive_payload": naive_payload,
        "n_hist_facts":  n_hist,
        "methods":     method_results,
    }


# ════════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ════════════════════════════════════════════════════════════════════════════════

def compute_aggregates(task_results: List[dict]) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    for method in _METHODS:
        histls_vals: List[float] = []
        for t in task_results:
            m = t["methods"].get(method, {})
            if m.get("HistLS") is not None:
                histls_vals.append(m["HistLS"])
        n = len(histls_vals)
        agg[method] = {
            "n_tasks": n,
            "HistLS":  round(sum(histls_vals) / n, 4) if n else None,
        }
    return agg


# ════════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: Dict[str, dict]) -> None:
    W = 60
    print(f"\n{'═' * W}")
    print(f"  LEAKAGE ABLATION RESULTS  —  HistLS by history size")
    print(f"{'═' * W}")
    pct = lambda v: f"{v*100:.1f}%" if v is not None else "  n/a "

    col_w = 10
    hdr = f"  {'Size':>5}  {'Method':<12}  {'HistLS':>{col_w}}"
    print(hdr)
    print(f"  {'─'*5}  {'─'*12}  {'─'*col_w}")

    for size_key, run in all_results.items():
        agg = run["aggregates"]
        for i, method in enumerate(_METHODS):
            a     = agg.get(method, {})
            label = str(size_key) if i == 0 else ""
            print(f"  {label:>5}  {method:<12}  {pct(a.get('HistLS')):>{col_w}}")
        print()

    print(f"{'═' * W}")
    print(f"  HistLS = |Reveal(S^hist, P^m)| / |S^hist| per task (mean)")
    print(f"  Size   = number of pre-existing memory entries in preloaded trace")
    print(f"{'═' * W}\n")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Leakage ablation: HistLS vs preloaded history size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-tasks", type=int, default=None,
        help="Number of tasks to evaluate per history size (default: all)",
    )
    ap.add_argument(
        "--out", default=str(_RESULTS_FILE),
        help="Output JSON path (default: evaluation/run_leakage_ablation.json)",
    )
    ap.add_argument(
        "--tasks-file", default=str(_TASKS_FILE),
        help="Path to task_prompts.json",
    )
    ap.add_argument(
        "--local-model", default=None,
        help="Ollama local model (default: from config.py)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading evaluation data …")
    all_tasks     = load_tasks(Path(args.tasks_file))
    profile_facts = load_profile_facts(_PROFILE_FILE)

    n_total = len(all_tasks)
    n_eval  = min(args.n_tasks, n_total) if args.n_tasks else n_total
    tasks   = all_tasks[:n_eval]

    print(f"  Tasks         : {n_total} in file  →  evaluating {n_eval}")
    print(f"  Profile facts : {len(profile_facts)} values (S^prof, constant)")
    print(f"  History sizes : {list(_PRELOADED_SIZES)}")
    print(f"  Methods       : {', '.join(_METHODS)}")
    print(f"  Output        : {out_path}")

    print("\nInitialising models …")
    agent = HybridAgent(local_model=args.local_model)
    ps    = PrivScope(local_llm=agent.local)
    print(f"  Local LLM : {agent.local.model}")
    print(f"  No cloud calls — leakage is payload-only")

    all_results: Dict[str, dict] = {}

    for size in _PRELOADED_SIZES:
        preloaded_name = _PRELOADED_TEMPLATE.format(size=size)
        preloaded_path = _ROOT / "state" / preloaded_name

        if not preloaded_path.exists():
            print(f"\n[SKIP] {preloaded_name} not found — skipping size {size}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  History size = {size}  ({preloaded_name})")
        print(f"{'─' * 60}")

        # Reset working_trace so each size starts from a clean runtime slate
        _reset_trace()

        # Reload trace facts for this preloaded size
        trace_facts = load_trace_facts(_TRACE_FILE, preloaded_path)
        print(f"  S^hist facts at start : {len(trace_facts)}")

        # Reload agent state with this preloaded trace
        _reload_agent_traces(agent, preloaded_path)

        task_results:   List[dict] = []
        history_facts:  List[str]  = []   # accumulates S^cur across tasks this run

        for i, task in enumerate(tasks):
            _reload_agent_traces(agent, preloaded_path)

            result = evaluate_task(
                task_idx      = i,
                task          = task,
                agent         = agent,
                ps            = ps,
                profile_facts = profile_facts,
                history_facts = list(history_facts),
                trace_facts   = trace_facts,
            )
            task_results.append(result)

            # Grow S^hist with this task's current-task facts
            cur_facts = [str(f).strip() for f in task.get("sensitive_info", []) if str(f).strip()]
            history_facts.extend(f for f in cur_facts if f not in history_facts)

            # Append naive CLM response placeholder to trace (keeps carryover realistic)
            _append_trace({
                "task_id": result["seed_id"],
                "domain":  result["domain"],
                "data":    {"prompt": result["prompt"]},
            })

        aggregates = compute_aggregates(task_results)
        all_results[str(size)] = {
            "preloaded_trace":  preloaded_name,
            "n_trace_facts":    len(trace_facts),
            "n_tasks":          len(task_results),
            "aggregates":       aggregates,
            "tasks":            task_results,
        }

        print(f"\n  Aggregates for size {size}:")
        for method in _METHODS:
            a  = aggregates[method]
            hl = f"{a['HistLS']*100:.1f}%" if a["HistLS"] is not None else "n/a"
            print(f"    {method:<12}  HistLS={hl}")

    # Final reset so the workspace is clean
    _reset_trace()
    print("\nworking_trace.json reset after ablation run.")

    output = {
        "metadata": {
            "n_tasks":        n_eval,
            "history_sizes":  list(_PRELOADED_SIZES),
            "methods":        list(_METHODS),
            "local_model":    agent.local.model,
            "timestamp":      datetime.now().isoformat(),
        },
        "ablation_results": all_results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved → {out_path}\n")

    print_summary(all_results)


if __name__ == "__main__":
    main()
