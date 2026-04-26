"""
evaluation/utility_eval.py
════════════════════════════════════════════════════════════════════════════════
Utility evaluation — Utility Retention Rate (URR).

For each task the naive payload is sent to the cloud first; the returned
provider list becomes the ground truth.  Sanitized payloads (PrivacyScope,
NER-REDACT) are then sent and their provider lists are matched against the
naive list.

  URR = mean over tasks of  |matched| / |naive providers|

By construction URR = 1.0 for the naive baseline.

Provider name matching (stop-words stripped):
  · 1–2 content words  →  all content words must match
  · 3+  content words  →  ≥2 content words must overlap

Outputs aggregate URR per baseline and saves full detail to JSON.

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

  # 10 tasks, all domains mixed
  uv run python evaluation/utility_eval.py --n-tasks 10

  # Fixed seed for reproducibility
  uv run python evaluation/utility_eval.py --n-tasks 10 --seed 42

  # Filter to one domain
  uv run python evaluation/utility_eval.py --n-tasks 10 --domain medical_booking

  # Custom output file
  uv run python evaluation/utility_eval.py --output evaluation/utility_results.json

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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
from evaluation.run_eval import PayloadGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_TASKS_PATH  = _ROOT / "task_generated" / "task_prompts.json"
_DEFAULT_OUT = Path(__file__).resolve().parent / "utility_results.json"

_BASELINES = ("naive", "privacyscope", "presidio")

# ── Stop words ────────────────────────────────────────────────────────────────
_STOP = {
    "the", "of", "and", "&", "at", "a", "an", "in", "for", "to",
    "dr", "st", "mr", "ms", "mrs", "llc", "inc", "co", "ltd",
}


# ── Name matching ─────────────────────────────────────────────────────────────

def _content_words(name: str) -> list[str]:
    tokens = re.split(r"[\s\-/]+", name.lower())
    return [
        re.sub(r"[^a-z0-9]", "", t) for t in tokens
        if re.sub(r"[^a-z0-9]", "", t) and
           re.sub(r"[^a-z0-9]", "", t) not in _STOP
    ]


def names_match(naive_name: str, candidate_name: str) -> bool:
    nw = set(_content_words(naive_name))
    cw = set(_content_words(candidate_name))
    if not nw or not cw:
        return False
    overlap = nw & cw
    return nw.issubset(cw) if len(nw) <= 2 else len(overlap) >= 2


def count_matches(naive_names: list[str], candidate_names: list[str]) -> tuple[int, list[tuple]]:
    """
    Greedily match each naive provider to at most one candidate.
    Returns (match_count, [(naive_name, matched_candidate), ...]).
    """
    matched = 0
    pairs: list[tuple] = []
    used: set[int] = set()
    for naive in naive_names:
        for j, cand in enumerate(candidate_names):
            if j not in used and names_match(naive, cand):
                matched += 1
                pairs.append((naive, cand))
                used.add(j)
                break
    return matched, pairs


# ── Cloud helpers ─────────────────────────────────────────────────────────────

def _cloud_search(cloud_llm, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Return a numbered list of up to 5 local service providers matching the request. "
                "For each provider include exactly: name, address, phone. Be concise. "
                "Format each entry as:\n"
                "1. <Name>\n   Address: <address>\n   Phone: <phone>\n"
            ),
        },
        {"role": "user", "content": query},
    ]
    return cloud_llm.chat(messages)


def _parse_names(text: str) -> list[str]:
    names = []
    for line in text.strip().splitlines():
        m = re.match(r"^\d+[.)]\s*\*{0,2}(.+?)\*{0,2}\s*$", line.strip())
        if m:
            name = m.group(1).strip(" -:")
            if name:
                names.append(name)
    return names[:5]


# ── Task selection ────────────────────────────────────────────────────────────

def select_tasks(
    tasks_path: Path,
    n_tasks:    int,
    domain:     str | None,
    rng:        random.Random,
) -> list[dict]:
    data = json.loads(tasks_path.read_text())
    pool = data["task_prompts"]
    if domain:
        pool = [t for t in pool if t.get("domain") == domain]
        if not pool:
            raise SystemExit(f"No tasks found for domain '{domain}'")
    return rng.sample(pool, k=min(n_tasks, len(pool)))


# ── URR computation ───────────────────────────────────────────────────────────

def compute_urr(naive_names: list[str], candidate_names: list[str]) -> dict:
    matched, pairs = count_matches(naive_names, candidate_names)
    total = len(naive_names)
    return {
        "URR":            matched / total if total else None,
        "matched":        matched,
        "total_naive":    total,
        "matched_pairs":  pairs,
        "naive_names":    naive_names,
        "returned_names": candidate_names,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    tasks_path: Path,
    n_tasks:    int,
    domain:     str | None,
    rng_seed:   int,
    output:     Path,
) -> None:

    print(f"\n{'═' * 65}")
    print(f"  UTILITY EVALUATION  —  URR  —  {n_tasks} tasks")
    print(f"  Baselines : naive (ground truth), privacyscope, presidio")
    print(f"{'═' * 65}\n")

    rng   = random.Random(rng_seed)
    tasks = select_tasks(tasks_path, n_tasks, domain, rng)

    print(f"  Initialising models…")
    gen = PayloadGenerator()
    from llm.cloud_router import CloudLLM
    cloud = CloudLLM()
    print(f"  Local model : {gen.local.model}\n")

    # accumulators: {baseline: {urr_sum, urr_n}}
    accum = {b: {"urr_sum": 0.0, "urr_n": 0} for b in _BASELINES}
    records: list[dict] = []

    for i, task_entry in enumerate(tasks, 1):
        prompt = task_entry["prompt"]
        print(f"{'─' * 65}")
        print(f"  Task {i}/{len(tasks)}  [{task_entry.get('domain','?')}]  {task_entry.get('seed_id','?')}")
        print(f"  {prompt[:75]}")

        payloads = gen.generate(prompt)

        # Query cloud for each baseline
        provider_names: dict[str, list[str]] = {}
        print("  Cloud queries…")
        for b in _BASELINES:
            raw  = _cloud_search(cloud, payloads[b])
            names = _parse_names(raw)
            provider_names[b] = names
            print(f"    {b:<14} → {names}")

        naive_names = provider_names["naive"]

        # Compute URR per baseline
        task_metrics: dict[str, dict] = {}
        for b in _BASELINES:
            m = compute_urr(naive_names, provider_names[b])
            task_metrics[b] = m
            if m["URR"] is not None:
                accum[b]["urr_sum"] += m["URR"]
                accum[b]["urr_n"]   += 1

        # Per-task table
        print(f"\n  {'Baseline':<14}  {'URR':>7}  matched/naive")
        print(f"  {'─'*14}  {'─'*7}  {'─'*12}")
        for b in _BASELINES:
            m = task_metrics[b]
            urr_s = f"{m['URR']*100:5.1f}%" if m["URR"] is not None else "  —  "
            print(f"  {b:<14}  {urr_s:>7}  {m['matched']}/{m['total_naive']}")

        records.append({
            "task_num":    i,
            "domain":      task_entry.get("domain", ""),
            "seed_id":     task_entry.get("seed_id", ""),
            "variant_id":  task_entry.get("variant_id", ""),
            "prompt":      prompt,
            "metrics":     task_metrics,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        b: {
            "URR": accum[b]["urr_sum"] / accum[b]["urr_n"] if accum[b]["urr_n"] else None,
            "n":   accum[b]["urr_n"],
        }
        for b in _BASELINES
    }

    def _f(v): return f"{v*100:5.1f}%" if v is not None else "  —  "

    print(f"\n{'═' * 65}")
    print(f"  SUMMARY  —  {len(tasks)} tasks")
    print(f"{'═' * 65}")
    print(f"\n  {'Baseline':<14}  {'URR':>7}  N")
    print(f"  {'─'*14}  {'─'*7}  {'─'*4}")
    for b in _BASELINES:
        print(f"  {b:<14}  {_f(summary[b]['URR']):>7}  {summary[b]['n']}")
    print()
    print("  URR = fraction of naive providers recovered (1.0 = full utility)")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    result_doc = {
        "generated_at":  datetime.now().isoformat(),
        "n_tasks":       len(tasks),
        "domain_filter": domain,
        "rng_seed":      rng_seed,
        "baselines":     list(_BASELINES),
        "matching": {
            "rule_1_2":   "all content words must match",
            "rule_3plus": ">=2 content words must overlap",
            "stop_words": sorted(_STOP),
        },
        "summary": summary,
        "tasks":   records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Utility evaluation: URR (Utility Retention Rate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-tasks",  type=int, default=10,
                    help="Number of tasks (default: 10)")
    ap.add_argument("--domain",   default=None,
                    help="Filter to a specific domain")
    ap.add_argument("--tasks",    default=str(_TASKS_PATH),
                    help=f"Path to task_prompts.json")
    ap.add_argument("--output",   default=str(_DEFAULT_OUT),
                    help=f"Output JSON file (default: {_DEFAULT_OUT})")
    ap.add_argument("--seed",     type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()

    run(
        tasks_path = Path(args.tasks),
        n_tasks    = args.n_tasks,
        domain     = args.domain,
        rng_seed   = args.seed,
        output     = Path(args.output),
    )


if __name__ == "__main__":
    main()
