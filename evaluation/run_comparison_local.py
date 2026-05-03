"""
evaluation/run_comparison_local.py
========================================================================
Local model sensitivity comparison for PEP and PrivScope.

No cloud calls — measures only sanitization metrics at the LC->CLM
boundary. Only PEP and PrivScope are evaluated because they are the only
methods that use an on-device model.

Metrics (per local model x method, aggregated across all tasks):
  LR       -- Leakage Rate (fraction of tasks leaking >=1 cur-task fact)
  LRatio   -- Mean fraction of S^cur leaked per task
  RLR      -- Residual Leakage Rate (S^prof u S^hist)
  RLRatio  -- Mean fraction of S^res leaked per task
  PR       -- Payload Reduction vs naive (negative = expansion)
  Lat (s)  -- Mean sanitization latency in seconds
  In tok   -- Mean input token count of sanitized payload

Local models tested (configurable via --models):
  llama3.2, phi3, mistral, qwen2.5:7b, llama3.1:8b

Usage (from project root agent_gpt/):
    uv run python evaluation/run_comparison_local.py
    uv run python evaluation/run_comparison_local.py --models llama3.2 qwen2.5:7b
    uv run python evaluation/run_comparison_local.py --tasks-file task_generated/task_prompts.json
========================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.hybrid_agent import HybridAgent
from privacy.privscope   import PrivScope
import privacy.pep       as pep_baseline

_ROOT         = Path(__file__).resolve().parent.parent
_TASKS_FILE   = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_FILE = _ROOT / "state" / "profile_state.json"
_OUT_FILE     = _ROOT / "evaluation" / "run_comparison_local.json"

_DEFAULT_MODELS = ["llama3.2", "phi3", "mistral", "qwen2.5:7b", "llama3.1:8b"]
_MODEL_LABELS = {
    "llama3.2":    "llama3.2    (3B)",
    "phi3":        "phi3        (3.8B)",
    "mistral":     "mistral     (7B)",
    "qwen2.5:7b":  "qwen2.5:7b  (7B)",
    "llama3.1:8b": "llama3.1:8b (8B)",
}
_METHODS = ("pep", "privscope")

W = 88

# ── Token counter ─────────────────────────────────────────────────────────────

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0
    _TOK_SOURCE = "tiktoken cl100k_base"
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) if text else 0
    _TOK_SOURCE = "word count (install tiktoken for accurate counts)"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, OSError):
            continue
    sys.exit(f"[ERROR] Could not read {path}")


def load_tasks(path: Path) -> List[dict]:
    data  = _load_json(path)
    tasks = data.get("task_prompts", [])
    if not tasks:
        sys.exit(f"[ERROR] No task_prompts found in {path}")
    return tasks


def load_profile_facts(path: Path) -> List[str]:
    data    = _load_json(path)
    profile = data.get("user_profile", {}) if isinstance(data, dict) else {}
    return [str(v).strip() for v in profile.values() if v and len(str(v).strip()) >= 3]


# ── Leakage detection ─────────────────────────────────────────────────────────

def leaking_facts(facts: List[str], payload: str) -> List[str]:
    if not payload or not facts:
        return []
    pl = payload.lower()
    return [f for f in facts if len(f) >= 3 and f.lower() in pl]


# ── Aggregation helper ────────────────────────────────────────────────────────

def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ── Display helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def _section(label: str) -> None:
    print(f"\n{'-' * W}")
    print(f"  {label}")
    print(f"{'-' * W}")


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(
    model_name:    str,
    tasks:         List[dict],
    profile_facts: List[str],
) -> Dict[str, dict]:
    print(f"\n  Initialising {model_name} ...", flush=True)
    try:
        agent = HybridAgent(local_model=model_name)
        print(f"  Loaded: {agent.local.model}")
    except Exception as e:
        print(f"  [SKIP] Could not load {model_name}: {e}")
        return {}

    ps = PrivScope(local_llm=agent.local)

    acc = {
        m: {k: [] for k in ("LR", "LRatio", "RLR", "RLRatio", "PR", "latency", "in_tok")}
        for m in _METHODS
    }

    history_facts: List[str] = []

    for i, task in enumerate(tasks):
        task_prompt  = task.get("prompt", "").strip()
        cur_facts    = [str(f).strip() for f in task.get("sensitive_info", []) if str(f).strip()]
        profile_dict = agent.state.get("user_profile", {})
        traces       = agent.state.get("memory_traces", [])

        residual_facts = list(dict.fromkeys(profile_facts + list(dict.fromkeys(history_facts))))

        inferred_prefs = agent._lc_infer_preferences(task_prompt)
        _, naive_payload = agent._lc_reason_cloud_query(task_prompt, inferred_prefs)
        naive_tokens = count_tokens(naive_payload)

        print(f"    [{i + 1:>3}/{len(tasks)}]", end=" ", flush=True)

        for method in _METHODS:
            t0 = time.perf_counter()
            try:
                if method == "pep":
                    payload, _ = pep_baseline.sanitize_with_trace(
                        naive_payload, profile_dict, task_prompt, traces, agent.local
                    )
                else:
                    payload, _ = ps.sanitize_with_trace(
                        naive_payload, profile_dict, task_prompt, traces
                    )
            except Exception as e:
                print(f"[{method} ERR: {e}]", end=" ", flush=True)
                payload = naive_payload
            latency = time.perf_counter() - t0

            leaked_cur = leaking_facts(cur_facts,      payload)
            leaked_res = leaking_facts(residual_facts, payload)
            n_cur      = len(cur_facts)
            n_res      = len(residual_facts)
            tok_m      = count_tokens(payload)

            acc[method]["LR"].append(1 if leaked_cur else 0)
            acc[method]["LRatio"].append(len(leaked_cur) / n_cur if n_cur else None)
            acc[method]["RLR"].append(1 if leaked_res else 0)
            acc[method]["RLRatio"].append(len(leaked_res) / n_res if n_res else 0.0)
            acc[method]["PR"].append((naive_tokens - tok_m) / naive_tokens if naive_tokens else 0.0)
            acc[method]["latency"].append(latency)
            acc[method]["in_tok"].append(tok_m)
            print(".", end="", flush=True)

        history_facts.extend(cur_facts)
        print()

    return {
        m: {
            "LR":      _mean(acc[m]["LR"]),
            "LRatio":  _mean(acc[m]["LRatio"]),
            "RLR":     _mean(acc[m]["RLR"]),
            "RLRatio": _mean(acc[m]["RLRatio"]),
            "PR":      _mean(acc[m]["PR"]),
            "Lat":     _mean(acc[m]["latency"]),
            "InTok":   _mean(acc[m]["in_tok"]),
            "n":       len(acc[m]["LR"]),
        }
        for m in _METHODS
    }


# ── Results table ─────────────────────────────────────────────────────────────

def _print_table(model_results: Dict[str, Dict[str, dict]]) -> None:
    cM  = 20
    cV  =  6

    metrics  = ["LR", "LRatio", "RLR", "RLRatio", "PR", "Lat", "InTok"]
    m_hdrs   = ["LR", "LRatio", "RLR", "RLRtio", "PR", "Lat(s)", "InTok"]

    def fv(v, key):
        if v is None:
            return "  --  "
        if key == "InTok":
            return f"{int(round(v))}"
        if key == "Lat":
            return f"{v:.2f}"
        return f"{v:.2f}"

    span = len(metrics) * (cV + 2) - 1

    print()
    print(
        f"  {'Local model':<{cM}}  "
        f"{'PEP':^{span}}  |  "
        f"{'PrivScope':^{span}}"
    )

    def mhdr_row():
        return "  ".join(f"{h:>{cV}}" for h in m_hdrs)

    print(f"  {'':<{cM}}  {mhdr_row()}  |  {mhdr_row()}")
    print(f"  {'-' * cM}  {'-' * span}  |  {'-' * span}")

    for model_name, results in model_results.items():
        label = _MODEL_LABELS.get(model_name, model_name)
        row   = f"  {label:<{cM}}"
        for idx, method in enumerate(_METHODS):
            r = results.get(method, {})
            vals = "  ".join(f"{fv(r.get(k), k):>{cV}}" for k in metrics)
            row += f"  {vals}"
            if idx == 0:
                row += "  |"
        print(row)


# ── JSON output ───────────────────────────────────────────────────────────────

def save_results(model_results: Dict[str, Dict[str, dict]], n_tasks: int, models: List[str]) -> None:
    payload = {
        "metadata": {
            "n_tasks":   n_tasks,
            "models":    models,
            "methods":   list(_METHODS),
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "results": {
            model: {
                method: metrics
                for method, metrics in method_results.items()
            }
            for model, method_results in model_results.items()
        },
    }
    _OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {_OUT_FILE.relative_to(_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare local model backbones for PEP and PrivScope (no cloud calls).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--models", nargs="+", default=_DEFAULT_MODELS,
        metavar="MODEL",
        help="Local Ollama models to test. Default: all five.",
    )
    ap.add_argument(
        "--tasks-file", default=str(_TASKS_FILE),
        help="Path to task_prompts.json.",
    )
    args = ap.parse_args()

    tasks         = load_tasks(Path(args.tasks_file))
    profile_facts = load_profile_facts(_PROFILE_FILE)

    _header("LOCAL MODEL SENSITIVITY  --  PEP vs PrivScope  (no cloud calls)")
    print(f"  Tasks          : {len(tasks)}")
    print(f"  Profile facts  : {len(profile_facts)}")
    print(f"  Models         : {', '.join(args.models)}")
    print(f"  Methods        : {', '.join(_METHODS)}  (only methods that use a local LLM)")
    print(f"  Token counter  : {_TOK_SOURCE}")

    model_results: Dict[str, Dict[str, dict]] = {}
    for model_name in args.models:
        _section(f"Model: {model_name}")
        model_results[model_name] = evaluate_model(model_name, tasks, profile_facts)

    _section("RESULTS  --  local model sensitivity")
    _print_table(model_results)

    print()
    print(f"  LR       = fraction of tasks leaking >=1 current-task sensitive fact")
    print(f"  LRatio   = mean fraction of S^cur leaked per task")
    print(f"  RLR      = fraction of tasks leaking >=1 residual (S^prof + S^hist) fact")
    print(f"  RLRatio  = mean fraction of S^res leaked per task")
    print(f"  PR       = mean fractional payload token reduction vs naive (negative = expansion)")
    print(f"  Lat(s)   = mean sanitization latency in seconds")
    print(f"  InTok    = mean input token count of sanitized payload sent to CLM")
    print(f"\n{'=' * W}")

    save_results(model_results, len(tasks), args.models)
    print()


if __name__ == "__main__":
    main()
