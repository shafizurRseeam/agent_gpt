import json
import os

STATE_PATH = "state/working_state.json"


def get_contacts():
    """Returns the user's contacts from working state."""
    if not os.path.exists(STATE_PATH):
        return []
    with open(STATE_PATH) as f:
        state = json.load(f)
    return state.get("contacts", [])
