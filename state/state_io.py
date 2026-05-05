"""
state/state_io.py

Central state I/O module.

Three files, two concerns:
  profile_state.json           — static user data (user_profile, contacts)
                                 Edit this file directly to change the user profile.
  working_trace_preloaded.json — static prior-life backstory for the user.
                                 Generated once; NEVER wiped between experiment runs.
  working_trace.json           — dynamic task data (memory_traces)
                                 Appended to automatically after every tool call.
                                 Reset this file between experiment runs.

Public API
----------
  load_state()         -> dict   merge all three files into one state dict
  save_state(state)    -> None   split and write to the correct file each
  append_trace(entry)  -> None   append one trace entry to working_trace.json only
                                 (does NOT touch profile_state.json or preloaded)
  reset_trace()        -> None   clear working_trace.json only (preloaded stays)
"""

#uv run python -c "from state.state_io import reset_trace; reset_trace()"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROFILE_PATH   = Path("state/profile_state.json")
BACKSTORY_PATH = Path("state/memory_backstory.json")   # fixed pre-history, never reset
TRACE_PATH     = Path("state/working_trace.json")       # grows during eval, reset between runs

_EMPTY_STATE: Dict[str, Any] = {
    "user_profile":    {},
    "contacts":        [],
    "memory_traces":   [],
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_state() -> Dict[str, Any]:
    """
    Load and merge all three state layers into one state dict:
      1. profile_state.json    — static demographics (never changes)
      2. memory_backstory.json — pre-existing history (fixed, never reset)
      3. working_trace.json    — runtime accumulation (reset between eval runs)
    """
    profile   = _read_json(PROFILE_PATH)
    backstory = _read_json(BACKSTORY_PATH)
    runtime   = _read_json(TRACE_PATH)

    traces = (
        backstory.get("memory_traces", [])
        + runtime.get("memory_traces", [])
    )

    return {
        "user_profile":  profile.get("user_profile", {}),
        "contacts":      profile.get("contacts", []),
        "memory_traces": traces,
    }


def save_state(state: Dict[str, Any]) -> None:
    """
    Persist state back to the source files.
    Profile fields → profile_state.json
    Trace fields   → working_trace.json  (preloaded is never touched by save_state)
    """
    profile_data = {
        "user_profile": state.get("user_profile", {}),
        "contacts":     state.get("contacts", []),
    }
    trace_data = {
        "memory_traces": state.get("memory_traces", []),
    }

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    PROFILE_PATH.write_text(json.dumps(profile_data, indent=2, ensure_ascii=False))
    TRACE_PATH.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False))


def reset_trace() -> None:
    """
    Clear working_trace.json only.
    memory_backstory.json is never touched — it is permanent pre-history.
    """
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps({"memory_traces": []}, indent=2))
    print("working_trace.json reset — runtime traces cleared (backstory unchanged).")


def append_trace(entry: Dict[str, Any]) -> None:
    """
    Append a single trace entry to working_trace.json only (run-time file).
    Never touches profile_state.json or working_trace_preloaded.json.
    """
    trace_data: Dict[str, Any] = {"memory_traces": []}
    if TRACE_PATH.exists():
        try:
            trace_data = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    trace_data.setdefault("memory_traces", []).append(entry)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False))
