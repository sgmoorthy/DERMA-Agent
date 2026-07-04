"""Helpers for reading secrets from environment variables or Streamlit secrets."""

from __future__ import annotations

import os
from typing import Optional


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a secret from environment variables or Streamlit secrets.

    Resolution order:
    1. Process environment variable
    2. Top-level Streamlit secret (e.g. `OPENAI_API_KEY`)
    3. Nested Streamlit section `[api_keys]` with lower-case key names
    4. Provided default
    """
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st  # type: ignore

        try:
            streamlit_value = st.secrets.get(name)
            if streamlit_value:
                return str(streamlit_value)
        except Exception:
            pass

        try:
            api_keys = st.secrets.get("api_keys")
            if api_keys:
                nested_value = api_keys.get(name.lower()) or api_keys.get(name)
                if nested_value:
                    return str(nested_value)
        except Exception:
            pass
    except Exception:
        pass

    return default


def has_secret(name: str) -> bool:
    """Return True if a secret is configured in env vars or Streamlit secrets."""
    return bool(get_secret(name))
