"""
evaluation/test_lc_payload.py
════════════════════════════════════════════════════════════════════════
Quick inspection of what the LC injects into the naive payload for a
sample of tasks, before running full experiments.

Usage (from project root):
    uv run python evaluation/test_lc_payload.py
    uv run python evaluation/test_lc_payload.py --n 5
    uv run python evaluation/test_lc_payload.py --n 3 --seed 10
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from agents.hybrid_agent import HybridAgent

_TASKS_FILE = _ROOT / "task_generated" / "task_prompts.json"


def load_tasks(n: int, seed_offset: int) -> list:
    data = json.loads(_TASKS_FILE.read_text(encoding="utf-8"))
    tasks = data["task_prompts"]
    # Pick n tasks starting from seed_offset, one per seed
    seen_seeds = set()
    picked = []
    for t in tasks:
        if t["seed_id"] not in seen_seeds:
            seen_seeds.add(t["seed_id"])
            picked.append(t)
        if len(picked) == n + seed_offset:
            break
    return picked[seed_offset:]


def divider(char="─", width=70):
    print(char * width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=3, help="Number of tasks to inspect (default: 3)")
    ap.add_argument("--seed", type=int, default=0, help="Start from this seed index (default: 0)")
    args = ap.parse_args()

    print("\nInitialising agent (local LLM must be running)...")
    agent = HybridAgent()
    print(f"Local model: {agent.local.model}\n")

    tasks = load_tasks(args.n, args.seed)

    for i, task in enumerate(tasks, 1):
        prompt          = task["prompt"]
        domain_sensitive = task.get("domain_sensitive", [])
        general_sensitive = task.get("general_sensitive", [])

        divider("═")
        print(f"  TASK {i}/{len(tasks)}  —  {task['seed_id']}  variant {task['variant_id']}")
        divider("═")

        print(f"\n  USER REQUEST (r_t):")
        print(f"  {prompt}\n")

        print(f"  SENSITIVE INFO in this task:")
        print(f"    domain_sensitive : {domain_sensitive}")
        print(f"    general_sensitive: {general_sensitive}\n")

        # Step 1: LC infers preferences from memory
        inferred = agent._lc_infer_preferences(prompt)

        print(f"  LC INFERRED CONTEXT (from memory traces):")
        if inferred:
            for line in inferred.splitlines():
                print(f"    {line}")
        else:
            print(f"    (nothing relevant found in memory)")
        print()

        # Step 2: LC builds naive payload
        reasoning, payload = agent._lc_reason_cloud_query(prompt, inferred)

        print(f"  LC REASONING:")
        for line in reasoning.splitlines():
            print(f"    {line}")
        print()

        divider()
        print(f"  NAIVE PAYLOAD (what goes to cloud before any sanitization):")
        divider()
        print(payload)
        divider()

        # Check which sensitive items made it into the payload
        pl = payload.lower()
        print(f"\n  SENSITIVE ITEM COVERAGE IN PAYLOAD:")
        all_items = domain_sensitive + general_sensitive
        for item in all_items:
            found = item.lower() in pl
            mark  = "YES" if found else " NO"
            print(f"    [{mark}]  {item}")
        print()

    divider("═")
    print("  Done.\n")


if __name__ == "__main__":
    main()
