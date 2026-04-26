from state.state_io import load_state


def get_calendar():
    """Returns the user's calendar entries from profile/trace state."""
    return load_state().get("calendar", [])
