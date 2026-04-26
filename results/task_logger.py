"""
results/task_logger.py

Append-only task result logger for the AgentGPT privacy baseline evaluation.

Each task run is stored as one entry in results/task_results.json.

Schema per entry:
  task_id          int      — auto-incrementing, persists across runs
  timestamp        str      — ISO-8601
  prompt           str      — exact user input
  mode             int      — 1 = 2-path, 2 = 5-path
  payloads         dict     — cloud-bound payloads keyed by baseline name
  clm_responses    dict     — CLM responses keyed by baseline name

Baseline keys (null when mode=1):
  naive, privacyscope, pep, agentdam, presidio

──────────────────────────────────────────────────────────────────────────────
To RESET (clear all tasks and reset task counter), run in the terminal:

  python -c "from results.task_logger import reset; reset()"

──────────────────────────────────────────────────────────────────────────────
"""

import copy
import json
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path("results/task_results.json")

_BASELINE_KEYS = ("naive", "privacyscope", "pep", "agentdam", "presidio")

_TEMPLATE = {
    "_meta": {
        "description": (
            "AgentGPT privacy baseline evaluation results. "
            "Each entry captures the user prompt, all cloud-bound payloads "
            "(naive + privacy baselines), and the CLM response for each."
        ),
        "reset_command": (
            "python -c \"from results.task_logger import reset; reset()\""
        ),
        "baseline_keys": list(_BASELINE_KEYS),
        "task_count": 0,
    },
    "tasks": [],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return copy.deepcopy(_TEMPLATE)


def _save(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Public API ────────────────────────────────────────────────────────────────

def append_task(
    prompt: str,
    mode: int,
    payloads: dict,
    clm_responses: dict,
) -> int:
    """
    Append one task result and return the assigned task_id.

    Parameters
    ----------
    prompt        : user's raw input string
    mode          : 1 (2-path) or 2 (5-path)
    payloads      : dict with any subset of _BASELINE_KEYS
    clm_responses : dict with any subset of _BASELINE_KEYS
    """
    data    = _load()
    task_id = data["_meta"]["task_count"] + 1
    data["_meta"]["task_count"] = task_id

    entry = {
        "task_id":   task_id,
        "timestamp": datetime.now().isoformat(),
        "prompt":    prompt,
        "mode":      mode,
        "payloads": {
            k: payloads.get(k) for k in _BASELINE_KEYS
        },
        "clm_responses": {
            k: clm_responses.get(k) for k in _BASELINE_KEYS
        },
    }

    data["tasks"].append(entry)
    _save(data)
    return task_id


def reset() -> None:
    """Clear all tasks and reset the task counter to 0."""
    fresh = copy.deepcopy(_TEMPLATE)
    _save(fresh)
    print(f"Results cleared — {RESULTS_FILE} reset to empty.")
