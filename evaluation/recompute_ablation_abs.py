"""
evaluation/recompute_ablation_abs.py
════════════════════════════════════════════════════════════════════════
Recomputes the leakage ablation results using absolute leaked count
(|Reveal(S^hist, P^m)|) instead of HistLS ratio.

Reads run_leakage_ablation.json in-place, updates aggregates, and
reprints the summary table with AbsLeak (mean absolute leaked facts).

No experiment re-run needed — all leaked_hist lists are already stored.

Usage:
    uv run python evaluation/recompute_ablation_abs.py
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT       = Path(__file__).resolve().parent.parent
_JSON_FILE  = Path(__file__).resolve().parent / "run_leakage_ablation.json"
_METHODS    = ("naive", "privscope", "presidio", "pep")


def _mean(vals: list):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> None:
    if not _JSON_FILE.exists():
        sys.exit(f"[ERROR] {_JSON_FILE} not found — run run_leakage_ablation.py first.")

    data = json.loads(_JSON_FILE.read_text(encoding="utf-8"))

    ablation = data["ablation_results"]

    # ── Recompute AbsLeak per size ────────────────────────────────────────────
    for size_key, run in ablation.items():
        tasks = run["tasks"]
        acc = {m: [] for m in _METHODS}

        for task in tasks:
            for method in _METHODS:
                m     = task["methods"].get(method, {})
                count = len(m.get("leaked_hist", []))
                # add AbsLeak to per-task method record
                m["AbsLeak"] = count
                acc[method].append(count)

        # update aggregates
        for method in _METHODS:
            run["aggregates"][method]["AbsLeak"] = _mean(acc[method])

    # ── Save updated JSON ─────────────────────────────────────────────────────
    _JSON_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {_JSON_FILE.name} with AbsLeak values.")

    # ── Print summary table ───────────────────────────────────────────────────
    W = 60
    print(f"\n{'=' * W}")
    print(f"  LEAKAGE ABLATION  --  AbsLeak by history size")
    print(f"  AbsLeak = mean |Reveal(S^hist, P^m)| per task")
    print(f"{'=' * W}")

    col_w = 9
    print(f"  {'Size':>5}  {'Method':<12}  {'AbsLeak':>{col_w}}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*col_w}")

    for size_key, run in ablation.items():
        agg = run["aggregates"]
        for i, method in enumerate(_METHODS):
            a     = agg.get(method, {})
            label = str(size_key) if i == 0 else ""
            v     = a.get("AbsLeak")
            val   = f"{v:.3f}" if v is not None else "  n/a"
            print(f"  {label:>5}  {method:<12}  {val:>{col_w}}")
        print()

    print(f"{'=' * W}")
    print(f"  Size = number of pre-existing memory entries in preloaded trace")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
