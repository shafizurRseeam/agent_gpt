"""
evaluation/model_sensitivity_eval.py
════════════════════════════════════════════════════════════════════════════════
Model sensitivity experiment.

Fixes the cloud LLM (gpt-4o-mini) and varies the local model (LC backbone).
Runs the same set of tasks through each local model and computes:

  Privacy  : LR, LRatio, RLR, RLRatio, PR
  Utility  : URR  (Utility Retention Rate vs naive)
  Efficiency: mean sanitization latency (s)

Results are stored per-model and saved to JSON.

Local models evaluated (all served via Ollama on localhost:11434):
  llama3.2:1b  — 1B,   smallest, tends toward over-disclosure
  llama3.2     — 3B,   default balanced model
  llama3.1:8b  — 8B,   larger / more instruction-following
  mistral      — 7B,   alternative architecture
  phi3         — 3.8B, Microsoft instruction-tuned

Cloud model (fixed): gpt-4o-mini

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

  # 10 tasks, all default models
  uv run python evaluation/model_sensitivity_eval.py --n-tasks 10

  # Custom model list (comma-separated Ollama model names)
  uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 \\
      --models llama3.2:1b,llama3.2,llama3.1:8b

  # Fixed seed
  uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 --seed 42

  # Custom output
  uv run python evaluation/model_sensitivity_eval.py --n-tasks 10 \\
      --output evaluation/sensitivity_results.json

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_eval import (
    PayloadGenerator,
    compute_metrics,
    _extract_profile_facts,
    _token_count,
    _payload_reduction,
)
from evaluation.utility_eval import (
    compute_urr,
    _cloud_search,
    _parse_names,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_TASKS_PATH  = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_PATH = _ROOT / "state" / "profile_state.json"
_DEFAULT_OUT = Path(__file__).resolve().parent / "sensitivity_results.json"

# Default local models to sweep (Ollama names)
_DEFAULT_MODELS = [
    "llama3.2:1b",
    "llama3.2",
    "llama3.1:8b",
    "mistral",
    "phi3",
]

_BASELINES    = ("naive", "privacyscope", "presidio")
_CLOUD_MODEL  = "gpt-4o-mini"


# ── Task selection ────────────────────────────────────────────────────────────

def select_tasks(tasks_path: Path, n_tasks: int, rng: random.Random) -> list[dict]:
    data = json.loads(tasks_path.read_text())
    pool = data["task_prompts"]
    return rng.sample(pool, k=min(n_tasks, len(pool)))


# ── Empty per-baseline accumulator ───────────────────────────────────────────

def _empty_accum() -> dict:
    return {
        b: {
            "lr_hits": 0, "lr_ratio_sum": 0.0, "lr_n": 0,
            "rlr_hits": 0, "rlr_ratio_sum": 0.0, "rlr_n": 0,
            "pr_sum": 0.0, "pr_n": 0,
            "urr_sum": 0.0, "urr_n": 0,
            "san_lat_sum": 0.0, "san_lat_n": 0,
        }
        for b in _BASELINES
    }


# ── Run one local model over the task list ────────────────────────────────────

def run_one_model(
    local_model:   str,
    tasks:         list[dict],
    profile_facts: list[str],
    cloud,
) -> tuple[dict, list[dict]]:
    """
    Returns (summary_per_baseline, per_task_records).
    summary_per_baseline: {baseline: {LR, LRatio, RLR, RLRatio, PR, URR, mean_san_lat_s}}
    """
    print(f"\n  {'─'*60}")
    print(f"  Local model: {local_model}")
    print(f"  {'─'*60}")

    gen = PayloadGenerator.__new__(PayloadGenerator)
    # Initialise manually so we can inject the model name
    from llm.local_llm import LocalLLM
    from privacy.privacyscope import PrivacyScope
    import privacy.presidio as presidio_mod
    from state.state_io import load_state

    gen.local      = LocalLLM(model=local_model)
    gen.ps         = PrivacyScope()
    gen.presidio = presidio_mod
    gen.state      = load_state()

    accum        = _empty_accum()
    trace_facts: list[str] = []
    records:     list[dict] = []

    for i, task_entry in enumerate(tasks, 1):
        prompt     = task_entry["prompt"]
        seed_facts = [str(x).strip() for x in task_entry.get("sensitive_info", []) if str(x).strip()]
        resid_facts = profile_facts + trace_facts

        print(f"\n  [{i}/{len(tasks)}] {task_entry.get('domain','?')}  {task_entry.get('seed_id','?')}")
        print(f"  {prompt[:70]}")

        # Generate payloads
        payloads = gen.generate(prompt)

        # Cloud search for URR
        provider_names: dict[str, list[str]] = {}
        for b in _BASELINES:
            raw   = _cloud_search(cloud, payloads[b])
            provider_names[b] = _parse_names(raw)

        naive_names = provider_names["naive"]

        task_metrics: dict[str, dict] = {}
        for b in _BASELINES:
            # Privacy metrics
            m  = compute_metrics(seed_facts, resid_facts, payloads[b])
            pr = _payload_reduction(payloads["naive"], payloads[b])
            m["PR"]             = pr
            m["tokens_naive"]   = _token_count(payloads["naive"])
            m["tokens_payload"] = _token_count(payloads[b])

            # URR
            urr_m = compute_urr(naive_names, provider_names[b])
            m["URR"]            = urr_m["URR"]
            m["matched"]        = urr_m["matched"]
            m["total_naive"]    = urr_m["total_naive"]
            m["matched_pairs"]  = urr_m["matched_pairs"]
            m["naive_names"]    = naive_names
            m["returned_names"] = provider_names[b]

            task_metrics[b] = m

            # Accumulate
            if seed_facts:
                accum[b]["lr_hits"]      += (m["LR"] or 0)
                accum[b]["lr_ratio_sum"] += (m["LRatio"] or 0.0)
                accum[b]["lr_n"]         += 1

            accum[b]["rlr_hits"]      += m["RLR"]
            accum[b]["rlr_ratio_sum"] += m["RLRatio"]
            accum[b]["rlr_n"]         += 1

            if pr is not None:
                accum[b]["pr_sum"] += pr
                accum[b]["pr_n"]   += 1

            if m["URR"] is not None:
                accum[b]["urr_sum"] += m["URR"]
                accum[b]["urr_n"]   += 1

            san_lat = {
                "naive":        0.0,
                "privacyscope": payloads["latency_ps_s"],
                "presidio":   payloads["latency_ner_s"],
            }.get(b, 0.0)
            accum[b]["san_lat_sum"] += san_lat
            accum[b]["san_lat_n"]   += 1

        # Print per-task row
        print(f"  {'Baseline':<14}  {'LR':>4}  {'RLR':>4}  {'URR':>7}  {'PR':>7}  Tok(N→S)")
        for b in _BASELINES:
            mm    = task_metrics[b]
            lr_s  = str(mm["LR"])   if mm["LR"]  is not None else " —"
            rlr_s = str(mm["RLR"])
            urr_s = f"{mm['URR']*100:5.1f}%" if mm["URR"] is not None else "  —  "
            pr_s  = f"{mm['PR']*100:5.1f}%"  if mm["PR"]  is not None else "  —  "
            print(
                f"  {b:<14}  {lr_s:>4}  {rlr_s:>4}  "
                f"{urr_s:>7}  {pr_s:>7}  "
                f"{mm['tokens_naive']}→{mm['tokens_payload']}"
            )

        records.append({
            "task_num":       i,
            "domain":         task_entry.get("domain", ""),
            "seed_id":        task_entry.get("seed_id", ""),
            "variant_id":     task_entry.get("variant_id", ""),
            "prompt":         prompt,
            "sensitive_info": seed_facts,
            "latency": {
                "lc_s":    payloads["latency_lc_s"],
                "ps_s":    payloads["latency_ps_s"],
                "ner_s":   payloads["latency_ner_s"],
                "total_s": payloads["latency_total_s"],
            },
            "payloads": {
                "naive":        payloads["naive"],
                "privacyscope": payloads["privacyscope"],
                "presidio":   payloads["presidio"],
                "lc_reasoning": payloads["lc_reasoning"],
            },
            "metrics": task_metrics,
        })

        trace_facts.extend(seed_facts)

    # Build summary
    summary: dict[str, dict] = {}
    for b in _BASELINES:
        a     = accum[b]
        lr_n  = a["lr_n"];   rlr_n = a["rlr_n"]
        pr_n  = a["pr_n"];   urr_n = a["urr_n"];  lat_n = a["san_lat_n"]
        summary[b] = {
            "LR":             a["lr_hits"]      / lr_n  if lr_n  else None,
            "LRatio":         a["lr_ratio_sum"] / lr_n  if lr_n  else None,
            "RLR":            a["rlr_hits"]     / rlr_n if rlr_n else None,
            "RLRatio":        a["rlr_ratio_sum"]/ rlr_n if rlr_n else None,
            "PR":             a["pr_sum"]       / pr_n  if pr_n  else None,
            "URR":            a["urr_sum"]      / urr_n if urr_n else None,
            "mean_san_lat_s": round(a["san_lat_sum"] / lat_n, 3) if lat_n else 0.0,
            "n_lr":  lr_n,
            "n_rlr": rlr_n,
        }

    return summary, records


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    tasks_path:  Path,
    n_tasks:     int,
    local_models: list[str],
    rng_seed:    int,
    output:      Path,
) -> None:

    print(f"\n{'═'*68}")
    print(f"  MODEL SENSITIVITY EVALUATION")
    print(f"  Cloud model (fixed) : {_CLOUD_MODEL}")
    print(f"  Local models        : {', '.join(local_models)}")
    print(f"  Tasks               : {n_tasks}  (seed={rng_seed})")
    print(f"{'═'*68}\n")

    rng   = random.Random(rng_seed)
    tasks = select_tasks(tasks_path, n_tasks, rng)

    profile_data  = json.loads(_PROFILE_PATH.read_text()) if _PROFILE_PATH.exists() else {}
    profile_facts = _extract_profile_facts(profile_data)

    print(f"  Selected {len(tasks)} tasks:")
    for t in tasks:
        print(f"    [{t.get('domain','?')}]  {t.get('seed_id','?')}  →  {t['prompt'][:60]}…")

    from llm.cloud_router import CloudLLM
    cloud = CloudLLM()

    all_model_results: dict[str, dict] = {}

    for local_model in local_models:
        try:
            summary, records = run_one_model(local_model, tasks, profile_facts, cloud)
            all_model_results[local_model] = {
                "summary": summary,
                "tasks":   records,
            }
        except Exception as exc:
            print(f"\n  [Warning] Model '{local_model}' failed: {exc}")
            all_model_results[local_model] = {"error": str(exc)}

    # ── Print comparison table (PrivacyScope results across models) ───────────
    def _f(v): return f"{v*100:5.1f}%" if v is not None else "  —  "
    def _s(v): return f"{v:.2f}s"      if v is not None else "  —  "

    print(f"\n{'═'*90}")
    print(f"  SENSITIVITY TABLE  —  PrivacyScope  |  cloud: {_CLOUD_MODEL}")
    print(f"{'═'*90}")
    print(
        f"  {'Local model':<16}  "
        f"{'LR':>7}  {'LRatio':>7}  {'RLR':>7}  {'RLRatio':>8}  "
        f"{'URR':>7}  {'PR':>7}  {'Lat(s)':>7}"
    )
    print(
        f"  {'─'*16}  "
        f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  "
        f"{'─'*7}  {'─'*7}  {'─'*7}"
    )

    for model in local_models:
        res = all_model_results.get(model, {})
        if "error" in res:
            print(f"  {model:<16}  [failed: {res['error'][:50]}]")
            continue
        s = res["summary"].get("privacyscope", {})
        print(
            f"  {model:<16}  "
            f"{_f(s.get('LR')):>7}  {_f(s.get('LRatio')):>7}  "
            f"{_f(s.get('RLR')):>7}  {_f(s.get('RLRatio')):>8}  "
            f"{_f(s.get('URR')):>7}  {_f(s.get('PR')):>7}  "
            f"{_s(s.get('mean_san_lat_s')):>7}"
        )

    print(f"\n  (Naive and PRESIDIO per-model results saved in JSON)\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    result_doc = {
        "generated_at": datetime.now().isoformat(),
        "experiment":   "model_sensitivity",
        "cloud_model":  _CLOUD_MODEL,
        "local_models": local_models,
        "n_tasks":      len(tasks),
        "rng_seed":     rng_seed,
        "baselines":    list(_BASELINES),
        "results":      all_model_results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Model sensitivity: sweep local models, fix cloud to gpt-4o-mini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-tasks", type=int, default=10,
        help="Number of tasks to run per model (default: 10)",
    )
    ap.add_argument(
        "--models", default=None,
        help=(
            "Comma-separated Ollama model names to sweep. "
            f"Default: {','.join(_DEFAULT_MODELS)}"
        ),
    )
    ap.add_argument(
        "--tasks",  default=str(_TASKS_PATH),
        help="Path to task_prompts.json",
    )
    ap.add_argument(
        "--output", default=str(_DEFAULT_OUT),
        help=f"Output JSON file (default: {_DEFAULT_OUT})",
    )
    ap.add_argument(
        "--seed",   type=int, default=42,
        help="Random seed for task selection (default: 42)",
    )
    args = ap.parse_args()

    local_models = (
        [m.strip() for m in args.models.split(",")]
        if args.models else _DEFAULT_MODELS
    )

    run(
        tasks_path   = Path(args.tasks),
        n_tasks      = args.n_tasks,
        local_models = local_models,
        rng_seed     = args.seed,
        output       = Path(args.output),
    )


if __name__ == "__main__":
    main()
