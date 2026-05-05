"""
task_generated/patch_prompts_coverage.py
════════════════════════════════════════════════════════════════════════
Patch an existing task_prompts.json so that every prompt variant
contains all its sensitive_info items verbatim.

Uses the same _ensure_sensitive_coverage logic as task_generator.py.
No API calls needed — pure text injection.

Usage (from project root):
    uv run python task_generated/patch_prompts_coverage.py
    uv run python task_generated/patch_prompts_coverage.py --file task_generated/task_prompts.json
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from task_generated.task_generator import _ensure_sensitive_coverage


def patch(file_path: Path) -> None:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    tasks = data.get("task_prompts", [])

    patched = 0
    for task in tasks:
        original = task["prompt"]
        fixed = _ensure_sensitive_coverage(
            original,
            task.get("domain_sensitive", []),
            task.get("general_sensitive", []),
        )
        if fixed != original:
            task["prompt"] = fixed
            patched += 1

    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    prompts_only = file_path.with_stem(file_path.stem + "_prompts_only")
    if prompts_only.exists():
        po_data = [{"seed_id": t["seed_id"], "variant_id": t["variant_id"],
                    "domain": t["domain"], "prompt": t["prompt"],
                    "sensitive_info": t["sensitive_info"],
                    "domain_sensitive": t["domain_sensitive"],
                    "general_sensitive": t["general_sensitive"]}
                   for t in tasks]
        prompts_only.write_text(json.dumps(po_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Patched {patched}/{len(tasks)} prompts in {file_path.name}")
    if prompts_only.exists():
        print(f"Also updated {prompts_only.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parent / "task_prompts.json"),
        help="Path to task_prompts.json (default: task_generated/task_prompts.json)",
    )
    args = ap.parse_args()
    patch(Path(args.file))


if __name__ == "__main__":
    main()
