"""
evaluation/domain_utility_eval.py
════════════════════════════════════════════════════════════════════════════════
Domain-wise utility evaluation — Utility Retention Rate (URR).

Same as utility_eval.py but picks N tasks PER DOMAIN and produces a
domain-stratified results table.

  URR = mean over tasks of  |matched| / |naive providers|

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

  # 5 tasks per domain (default)
  uv run python evaluation/domain_utility_eval.py

  # 10 tasks per domain
  uv run python evaluation/domain_utility_eval.py --n-tasks-per-domain 10

  # Fixed seed
  uv run python evaluation/domain_utility_eval.py --n-tasks-per-domain 10 --seed 42

  # Custom output
  uv run python evaluation/domain_utility_eval.py --output evaluation/domain_utility_results.json

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
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
from evaluation.run_eval import PayloadGenerator
from evaluation.utility_eval import (
    compute_urr,
    count_matches,
    names_match,
    _cloud_search,
    _parse_names,
    _BASELINES,
    _STOP,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_TASKS_PATH  = _ROOT / "task_generated" / "task_prompts.json"
_DEFAULT_OUT = Path(__file__).resolve().parent / "domain_utility_results.json"


# ── Task selection ─────────────────────────────────────────────────────────────

def select_tasks_per_domain(
    tasks_path:   Path,
    n_per_domain: int,
    rng:          random.Random,
) -> dict[str, list[dict]]:
    data      = json.loads(tasks_path.read_text())
    by_domain: dict[str, list] = defaultdict(list)
    for t in data["task_prompts"]:
        by_domain[t["domain"]].append(t)
    return {
        domain: rng.sample(pool, k=min(n_per_domain, len(pool)))
        for domain, pool in sorted(by_domain.items())
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    tasks_path:   Path,
    n_per_domain: int,
    rng_seed:     int,
    output:       Path,
) -> None:

    print(f"\n{'═' * 68}")
    print(f"  DOMAIN-WISE UTILITY EVALUATION  —  URR  —  {n_per_domain} tasks per domain")
    print(f"  Baselines : naive (ground truth), privacyscope, presidio")
    print(f"{'═' * 68}\n")

    rng          = random.Random(rng_seed)
    domain_tasks = select_tasks_per_domain(tasks_path, n_per_domain, rng)
    domains      = list(domain_tasks.keys())
    total        = sum(len(v) for v in domain_tasks.values())

    print(f"  Domains : {', '.join(domains)}")
    print(f"  Total   : {total} tasks  ({n_per_domain} per domain)")
    print("\n  Initialising models…")
    gen = PayloadGenerator()
    from llm.cloud_router import CloudLLM
    cloud = CloudLLM()
    print(f"  Local model : {gen.local.model}\n")

    # {domain: {baseline: {urr_sum, urr_n}}}
    domain_accum: dict[str, dict] = {
        d: {b: {"urr_sum": 0.0, "urr_n": 0} for b in _BASELINES}
        for d in domains
    }
    all_records: list[dict] = []
    task_num = 0

    for domain in domains:
        tasks = domain_tasks[domain]
        print(f"{'─' * 68}")
        print(f"  Domain: {domain}  ({len(tasks)} tasks)")
        print(f"{'─' * 68}")

        for task_entry in tasks:
            task_num += 1
            prompt    = task_entry["prompt"]
            print(f"\n  [{task_num}] {task_entry.get('seed_id','?')}  →  {prompt[:70]}")

            payloads = gen.generate(prompt)

            # Query cloud for each baseline
            provider_names: dict[str, list[str]] = {}
            print("  Cloud queries…")
            for b in _BASELINES:
                raw   = _cloud_search(cloud, payloads[b])
                names = _parse_names(raw)
                provider_names[b] = names
                print(f"    {b:<14} → {names}")

            naive_names = provider_names["naive"]

            task_metrics: dict[str, dict] = {}
            for b in _BASELINES:
                m = compute_urr(naive_names, provider_names[b])
                task_metrics[b] = m
                if m["URR"] is not None:
                    domain_accum[domain][b]["urr_sum"] += m["URR"]
                    domain_accum[domain][b]["urr_n"]   += 1

            # Per-task table
            print(f"\n  {'Baseline':<14}  {'URR':>7}  matched/naive")
            for b in _BASELINES:
                m     = task_metrics[b]
                urr_s = f"{m['URR']*100:5.1f}%" if m["URR"] is not None else "  —  "
                print(f"  {b:<14}  {urr_s:>7}  {m['matched']}/{m['total_naive']}")

            all_records.append({
                "task_num":   task_num,
                "domain":     domain,
                "seed_id":    task_entry.get("seed_id", ""),
                "variant_id": task_entry.get("variant_id", ""),
                "prompt":     prompt,
                "metrics":    task_metrics,
            })

    # ── Build domain summary ──────────────────────────────────────────────────
    domain_summary: dict[str, dict] = {}
    for domain in domains:
        domain_summary[domain] = {
            b: {
                "URR": (domain_accum[domain][b]["urr_sum"] /
                        domain_accum[domain][b]["urr_n"])
                       if domain_accum[domain][b]["urr_n"] else None,
                "n":   domain_accum[domain][b]["urr_n"],
            }
            for b in _BASELINES
        }

    # ── Print domain table ────────────────────────────────────────────────────
    def _f(v): return f"{v*100:5.1f}%" if v is not None else "  —  "

    print(f"\n{'═' * 60}")
    print(f"  DOMAIN-WISE URR RESULTS")
    print(f"{'═' * 60}")
    header = f"  {'Domain':<22}  {'Method':<14}  {'URR':>7}  {'N':>4}"
    sep    = f"  {'─'*22}  {'─'*14}  {'─'*7}  {'─'*4}"
    print(header)
    print(sep)

    for i, domain in enumerate(domains):
        if i > 0:
            print(sep)
        first = True
        for b in _BASELINES:
            s       = domain_summary[domain][b]
            dom_col = domain if first else ""
            first   = False
            print(f"  {dom_col:<22}  {b:<14}  {_f(s['URR']):>7}  {s['n']:>4}")

    print(f"{'═' * 60}")
    print()
    print("  URR = fraction of naive providers recovered (1.0 = full utility)")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    result_doc = {
        "generated_at":       datetime.now().isoformat(),
        "n_tasks_per_domain": n_per_domain,
        "rng_seed":           rng_seed,
        "domains":            domains,
        "baselines":          list(_BASELINES),
        "matching": {
            "rule_1_2":   "all content words must match",
            "rule_3plus": ">=2 content words must overlap",
            "stop_words": sorted(_STOP),
        },
        "domain_summary": domain_summary,
        "tasks":          all_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Domain-wise utility evaluation: URR per domain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-tasks-per-domain", type=int, default=5,
                    help="Tasks per domain (default: 5)")
    ap.add_argument("--tasks",   default=str(_TASKS_PATH),
                    help="Path to task_prompts.json")
    ap.add_argument("--output",  default=str(_DEFAULT_OUT),
                    help=f"Output JSON file (default: {_DEFAULT_OUT})")
    ap.add_argument("--seed",    type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()

    run(
        tasks_path   = Path(args.tasks),
        n_per_domain = args.n_tasks_per_domain,
        rng_seed     = args.seed,
        output       = Path(args.output),
    )


if __name__ == "__main__":
    main()
