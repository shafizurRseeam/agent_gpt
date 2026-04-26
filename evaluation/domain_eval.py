"""
evaluation/domain_eval.py
════════════════════════════════════════════════════════════════════════════════
Domain-wise privacy evaluation.

Picks N tasks per domain, generates payloads (LC only — no cloud calls),
applies PrivacyScope and PRESIDIO, then prints and saves a domain-stratified
results table.

Metrics per domain × baseline:
  LR       Leakage Rate             (lower is better)
  LRatio   Leakage Ratio            (lower is better)
  RLR      Residual Leakage Rate    (lower is better)
  RLRatio  Residual Leakage Ratio   (lower is better)
  PR       Payload Reduction        (higher is better)
  Tok      Token count naive → sanitized

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

  # 5 tasks per domain (default)
  uv run python evaluation/domain_eval.py

  # 10 tasks per domain
  uv run python evaluation/domain_eval.py --n-tasks-per-domain 10

  # Fixed seed for reproducibility
  uv run python evaluation/domain_eval.py --n-tasks-per-domain 10 --seed 42

  # Custom output file
  uv run python evaluation/domain_eval.py --output evaluation/domain_results.json

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

# ── Shared utilities from run_eval ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.run_eval import (
    PayloadGenerator,
    compute_metrics,
    _extract_profile_facts,
    _token_count,
    _payload_reduction,
    _BASELINES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
_TASKS_PATH   = _ROOT / "task_generated" / "task_prompts.json"
_PROFILE_PATH = _ROOT / "state" / "profile_state.json"
_DEFAULT_OUT  = Path(__file__).resolve().parent / "domain_results.json"


# ── Task selection ─────────────────────────────────────────────────────────────

def select_tasks_per_domain(
    tasks_path:        Path,
    n_per_domain:      int,
    rng:               random.Random,
) -> dict[str, list[dict]]:
    """
    Returns {domain: [task_dicts]} with up to n_per_domain tasks per domain.
    """
    data = json.loads(tasks_path.read_text())
    by_domain: dict[str, list] = defaultdict(list)
    for t in data["task_prompts"]:
        by_domain[t["domain"]].append(t)

    result = {}
    for domain, pool in sorted(by_domain.items()):
        result[domain] = rng.sample(pool, k=min(n_per_domain, len(pool)))
    return result


# ── Empty accumulator ─────────────────────────────────────────────────────────

def _empty_accum() -> dict:
    return {
        b: {"lr_hits": 0, "lr_ratio_sum": 0.0, "lr_n": 0,
            "rlr_hits": 0, "rlr_ratio_sum": 0.0, "rlr_n": 0,
            "pr_sum": 0.0, "pr_n": 0,
            "tok_naive_sum": 0, "tok_out_sum": 0, "tok_n": 0,
            "san_lat_sum": 0.0, "san_lat_n": 0}
        for b in _BASELINES
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    tasks_path:   Path,
    n_per_domain: int,
    rng_seed:     int,
    output:       Path,
) -> None:

    print(f"\n{'═' * 68}")
    print(f"  DOMAIN-WISE PRIVACY EVALUATION  —  {n_per_domain} tasks per domain")
    print(f"  Baselines : naive, privacyscope, presidio")
    print(f"  Matching  : case-insensitive substring")
    print(f"{'═' * 68}\n")

    rng          = random.Random(rng_seed)
    domain_tasks = select_tasks_per_domain(tasks_path, n_per_domain, rng)
    domains      = list(domain_tasks.keys())

    profile_data  = json.loads(_PROFILE_PATH.read_text()) if _PROFILE_PATH.exists() else {}
    profile_facts = _extract_profile_facts(profile_data)

    total = sum(len(v) for v in domain_tasks.values())
    print(f"  Domains  : {', '.join(domains)}")
    print(f"  Total    : {total} tasks  ({n_per_domain} per domain)")
    print(f"  Profile  : {len(profile_facts)} facts from user_profile\n")

    print("  Initialising local model…")
    gen = PayloadGenerator()
    print(f"  Local model: {gen.local.model}\n")

    # accumulators indexed by domain
    domain_accum: dict[str, dict] = {d: _empty_accum() for d in domains}
    # global trace grows across ALL tasks (cross-domain, like a real session)
    trace_facts: list[str] = []
    # per-task records for JSON output
    all_records: list[dict] = []

    task_num = 0
    for domain in domains:
        tasks = domain_tasks[domain]
        print(f"{'─' * 68}")
        print(f"  Domain: {domain}  ({len(tasks)} tasks)")
        print(f"{'─' * 68}")

        for task_entry in tasks:
            task_num += 1
            task_prompt = task_entry["prompt"]
            seed_facts  = [str(x).strip() for x in task_entry.get("sensitive_info", []) if str(x).strip()]
            resid_facts = profile_facts + trace_facts

            print(f"\n  [{task_num}] {task_entry.get('seed_id', '?')}  →  {task_prompt[:70]}")
            print(f"       seed facts ({len(seed_facts)}): {seed_facts}")

            payloads = gen.generate(task_prompt)

            a = domain_accum[domain]

            metrics_per_baseline: dict[str, dict] = {}
            for b in _BASELINES:
                m = compute_metrics(seed_facts, resid_facts, payloads[b])

                pr  = _payload_reduction(payloads["naive"], payloads[b])
                tok_naive = _token_count(payloads["naive"])
                tok_out   = _token_count(payloads[b])
                m["PR"]             = pr
                m["tokens_naive"]   = tok_naive
                m["tokens_payload"] = tok_out
                metrics_per_baseline[b] = m

                if seed_facts:
                    a[b]["lr_hits"]      += (m["LR"] or 0)
                    a[b]["lr_ratio_sum"] += (m["LRatio"] or 0.0)
                    a[b]["lr_n"]         += 1

                a[b]["rlr_hits"]      += m["RLR"]
                a[b]["rlr_ratio_sum"] += m["RLRatio"]
                a[b]["rlr_n"]         += 1

                if pr is not None:
                    a[b]["pr_sum"] += pr
                    a[b]["pr_n"]   += 1

                a[b]["tok_naive_sum"] += tok_naive
                a[b]["tok_out_sum"]   += tok_out
                a[b]["tok_n"]         += 1

                san_lat = {
                    "naive":        0.0,
                    "privacyscope": payloads["latency_ps_s"],
                    "presidio":   payloads["latency_ner_s"],
                }.get(b, 0.0)
                a[b]["san_lat_sum"] += san_lat
                a[b]["san_lat_n"]   += 1

            all_records.append({
                "task_num":       task_num,
                "domain":         domain,
                "seed_id":        task_entry.get("seed_id", ""),
                "variant_id":     task_entry.get("variant_id", ""),
                "prompt":         task_prompt,
                "sensitive_info": seed_facts,
                "n_resid_facts":  len(resid_facts),
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
                "metrics": metrics_per_baseline,
            })

            # accumulate trace for subsequent tasks
            trace_facts.extend(seed_facts)

    # ── Build per-domain summary ───────────────────────────────────────────────
    domain_summary: dict[str, dict] = {}
    for domain in domains:
        a = domain_accum[domain]
        per_baseline: dict[str, dict] = {}
        for b in _BASELINES:
            lr_n  = a[b]["lr_n"]
            rlr_n = a[b]["rlr_n"]
            pr_n  = a[b]["pr_n"]
            tok_n = a[b]["tok_n"]
            lat_n = a[b]["san_lat_n"]
            per_baseline[b] = {
                "LR":              a[b]["lr_hits"]      / lr_n   if lr_n   else None,
                "LRatio":          a[b]["lr_ratio_sum"] / lr_n   if lr_n   else None,
                "RLR":             a[b]["rlr_hits"]     / rlr_n  if rlr_n  else None,
                "RLRatio":         a[b]["rlr_ratio_sum"]/ rlr_n  if rlr_n  else None,
                "PR":              a[b]["pr_sum"]       / pr_n   if pr_n   else None,
                "mean_tok_naive":  a[b]["tok_naive_sum"]/ tok_n  if tok_n  else None,
                "mean_tok_out":    a[b]["tok_out_sum"]  / tok_n  if tok_n  else None,
                "mean_san_lat_s":  round(a[b]["san_lat_sum"] / lat_n, 3) if lat_n else 0.0,
                "n_lr":  lr_n,
                "n_rlr": rlr_n,
            }
        domain_summary[domain] = per_baseline

    # ── Print domain table ────────────────────────────────────────────────────
    def _f(v):
        return f"{v*100:5.1f}%" if v is not None else "  —  "

    def _tok(s):
        return s if s is not None else "—"

    col_w = 14

    print(f"\n{'═' * 80}")
    print(f"  DOMAIN-WISE RESULTS")
    print(f"{'═' * 80}")
    header = (
        f"  {'Domain':<20}  {'Method':<14}  "
        f"{'LR':>7}  {'LRatio':>7}  {'RLR':>7}  {'RLRatio':>8}  "
        f"{'PR':>7}  {'Lat(s)':>7}  Tok(N→S)"
    )
    sep = (
        f"  {'─'*20}  {'─'*14}  "
        f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  "
        f"{'─'*7}  {'─'*7}  {'─'*9}"
    )
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

            tok_n   = domain_accum[domain][b]["tok_n"]
            tok_nav = domain_accum[domain][b]["tok_naive_sum"] / tok_n if tok_n else 0
            tok_out = domain_accum[domain][b]["tok_out_sum"]   / tok_n if tok_n else 0
            tok_str = f"{tok_nav:.0f}→{tok_out:.0f}"

            lat_s = f"{s['mean_san_lat_s']:.2f}s"

            print(
                f"  {dom_col:<20}  {b:<14}  "
                f"{_f(s['LR']):>7}  {_f(s['LRatio']):>7}  "
                f"{_f(s['RLR']):>7}  {_f(s['RLRatio']):>8}  "
                f"{_f(s['PR']):>7}  {lat_s:>7}  {tok_str}"
            )

    print(f"{'═' * 80}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    result_doc = {
        "generated_at":      datetime.now().isoformat(),
        "n_tasks_per_domain": n_per_domain,
        "rng_seed":          rng_seed,
        "domains":           domains,
        "baselines":         list(_BASELINES),
        "matching":          "case-insensitive substring",
        "token_counting":    "whitespace-split word count",
        "domain_summary":    domain_summary,
        "tasks":             all_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Domain-wise privacy evaluation: N tasks per domain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-tasks-per-domain", type=int, default=5,
        help="Tasks to evaluate per domain (default: 5)",
    )
    ap.add_argument(
        "--tasks", default=str(_TASKS_PATH),
        help=f"Path to task_prompts.json (default: {_TASKS_PATH})",
    )
    ap.add_argument(
        "--output", default=str(_DEFAULT_OUT),
        help=f"Output JSON file (default: {_DEFAULT_OUT})",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task selection (default: 42)",
    )
    args = ap.parse_args()

    run(
        tasks_path   = Path(args.tasks),
        n_per_domain = args.n_tasks_per_domain,
        rng_seed     = args.seed,
        output       = Path(args.output),
    )


if __name__ == "__main__":
    main()
