from state.state_io import load_state


def get_contacts():
    """Returns the user's contacts from profile state."""
    return load_state().get("contacts", [])
