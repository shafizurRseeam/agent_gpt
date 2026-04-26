"""
state/state_io.py

Central state I/O module.

Two files, two concerns:
  profile_state.json  — static user data (user_profile, contacts)
                        Edit this file directly to change the user profile.
  working_trace.json  — dynamic task data (memory_traces)
                        Appended to automatically after every tool call.

Public API
----------
  load_state()         -> dict   merge both files into one state dict
  save_state(state)    -> None   split and write to the correct file each
  append_trace(entry)  -> None   append one trace entry to working_trace.json
                                 (does NOT touch profile_state.json)
"""

#uv run python -c "from state.state_io import reset_trace; reset_trace()"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROFILE_PATH = Path("state/profile_state.json")
TRACE_PATH   = Path("state/working_trace.json")

_EMPTY_STATE: Dict[str, Any] = {
    "user_profile":    {},
    "contacts":        [],
    "memory_traces":   [],
}


def load_state() -> Dict[str, Any]:
    """Load and merge profile_state.json + working_trace.json into one dict."""
    profile = {}
    if PROFILE_PATH.exists():
        try:
            profile = json.loads(PROFILE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    trace = {}
    if TRACE_PATH.exists():
        try:
            trace = json.loads(TRACE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "user_profile":  profile.get("user_profile", {}),
        "contacts":      profile.get("contacts", []),
        "memory_traces": trace.get("memory_traces", []),
    }


def save_state(state: Dict[str, Any]) -> None:
    """
    Persist state back to the two source files.
    Profile fields → profile_state.json
    Trace fields   → working_trace.json
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
    """Clear all memory traces and reset working_trace.json to empty."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps({"memory_traces": []}, indent=2))
    print(f"working_trace.json reset — memory traces cleared.")


def append_trace(entry: Dict[str, Any]) -> None:
    """
    Append a single trace entry to working_trace.json only.
    Faster than load_state + save_state when only the trace changes.
    """
    trace_data: Dict[str, Any] = {"memory_traces": []}
    if TRACE_PATH.exists():
        try:
            trace_data = json.loads(TRACE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    trace_data.setdefault("memory_traces", []).append(entry)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False))
