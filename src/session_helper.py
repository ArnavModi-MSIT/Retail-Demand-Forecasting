"""
Per-browser-session identifier for the shared public demo.

This is NOT real authentication. It's a lightweight privacy boundary:
each browser session gets a random ID, stored in Streamlit's
session_state (lives in server memory, tied to that browser tab/session).
Runs are tagged with this ID so "Load Previous Run" only ever shows runs
created by the same session — preventing strangers on a shared public
deployment from seeing each other's uploaded data.

Limitations (acceptable for a demo, not for a real multi-tenant product):
    - Closing the tab / starting a new session loses access to old runs
      (they remain in the DB, just no longer visible to anyone).
    - No protection against someone deliberately sharing their session
      — but that's a self-inflicted choice, not a leak.
"""
import uuid

import streamlit as st


def get_session_id() -> str:
    """
    Return a stable ID for the current Streamlit session, generating one
    on first call. Stays the same across reruns within the same session,
    changes if the user starts a fresh session (new tab / cleared state).
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]
