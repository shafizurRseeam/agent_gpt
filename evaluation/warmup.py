"""
evaluation/warmup.py
════════════════════════════════════════════════════════════════════════════════
Warm-up phase: run a set of pre-selected tasks through the naive agent (mode 1)
to populate working_trace.json with cross-workflow context before evaluation.

What it does:
  1. Loads task_generated/task_prompts.json
  2. For each domain, randomly picks --seeds-per-domain seed IDs
  3. For each chosen seed, picks one prompt variant at random
  4. Saves the selected tasks to evaluation/warmup_tasks.json
  5. Runs each task through naive-only (mode 1) so working_trace.json fills up

After running this, the LC will have real cross-workflow memory to draw from
when you start the actual evaluation.

════════════════════════════════════════════════════════════════════════════════
HOW TO RUN  (from the project root: agent_gpt/)
════════════════════════════════════════════════════════════════════════════════

# Default: 3 seeds per domain
#   uv run python evaluation/warmup.py

# Custom seeds per domain:
#   uv run python evaluation/warmup.py --seeds-per-domain 5

# Use a specific task prompts file:
#   uv run python evaluation/warmup.py --tasks task_generated/task_prompts.json

# Reset working trace first, then warm up:
#   uv run python -c "from state.state_io import reset_trace; reset_trace()"
#   uv run python evaluation/warmup.py

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ── Load .env from project root ───────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent
_DEFAULT_TASKS = _ROOT / "task_generated" / "task_prompts.json"
_WARMUP_OUT    = Path(__file__).resolve().parent / "warmup_tasks.json"


def select_warmup_tasks(
    tasks_file: Path,
    seeds_per_domain: int,
    rng: random.Random,
) -> list[dict]:
    """
    For each domain, randomly pick `seeds_per_domain` unique seed IDs,
    then pick one prompt variant per seed.
    Returns a flat list of selected task dicts.
    """
    data = json.loads(tasks_file.read_text())
    all_prompts = data["task_prompts"]

    # Group prompt variants by (domain, seed_id)
    by_domain_seed: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in all_prompts:
        by_domain_seed[p["domain"]][p["seed_id"]].append(p)

    selected = []
    for domain, seeds in sorted(by_domain_seed.items()):
        seed_ids      = list(seeds.keys())
        chosen_seeds  = rng.sample(seed_ids, k=min(seeds_per_domain, len(seed_ids)))
        for sid in chosen_seeds:
            variant = rng.choice(seeds[sid])
            selected.append(variant)

    return selected


def run_warmup(tasks_file: Path, seeds_per_domain: int) -> None:
    rng = random.Random(99)   # fixed seed for reproducibility

    print(f"\n{'═' * 60}")
    print(f"  WARM-UP PHASE")
    print(f"  tasks file      : {tasks_file}")
    print(f"  seeds per domain: {seeds_per_domain}")
    print(f"{'═' * 60}\n")

    # ── Select tasks ──────────────────────────────────────────────────────────
    selected = select_warmup_tasks(tasks_file, seeds_per_domain, rng)

    # Save warmup task list
    _WARMUP_OUT.parent.mkdir(parents=True, exist_ok=True)
    warmup_record = {
        "generated_at":    datetime.now().isoformat(),
        "tasks_file":      str(tasks_file),
        "seeds_per_domain": seeds_per_domain,
        "total_tasks":     len(selected),
        "tasks":           selected,
    }
    _WARMUP_OUT.write_text(json.dumps(warmup_record, indent=2, ensure_ascii=False))

    print(f"  Selected {len(selected)} warm-up tasks:")
    for t in selected:
        print(f"    [{t['domain']}]  {t['seed_id']}  →  {t['prompt'][:70]}…")
    print()

    # ── Run each task through naive-only agent ────────────────────────────────
    from agents.hybrid_agent import HybridAgent

    agent = HybridAgent()

    for i, task in enumerate(selected, 1):
        prompt = task["prompt"]
        print(f"\n{'─' * 60}")
        print(f"  Warm-up task {i}/{len(selected)}  [{task['domain']}]  {task['seed_id']}")
        print(f"{'─' * 60}")
        try:
            agent.run(prompt, mode=1)   # mode 1 = naive only, no interactive prompt
        except Exception as exc:
            print(f"  [Warning] Task failed: {exc}")

    print(f"\n{'═' * 60}")
    print(f"  Warm-up complete — {len(selected)} tasks run.")
    print(f"  working_trace.json now has cross-workflow memory.")
    print(f"  Warm-up task list saved to: {_WARMUP_OUT}")
    print(f"{'═' * 60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run warm-up tasks to populate working_trace.json before evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--seeds-per-domain", type=int, default=3,
        help="Number of seed IDs to pick per domain (default: 3)",
    )
    ap.add_argument(
        "--tasks", default=str(_DEFAULT_TASKS),
        help=f"Path to task_prompts.json (default: {_DEFAULT_TASKS})",
    )
    args = ap.parse_args()

    run_warmup(
        tasks_file       = Path(args.tasks),
        seeds_per_domain = args.seeds_per_domain,
    )


if __name__ == "__main__":
    main()
