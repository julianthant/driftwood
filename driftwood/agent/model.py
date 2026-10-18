"""Chat model for proposals. Tests pass their own object and never call this."""

from __future__ import annotations

import os

DEFAULT_MODEL = "grok-4"


def build_chat_model(model: str | None = None, api_key: str | None = None):
    """Build a ChatXAI client from the environment.

    ``XAI_MODEL`` overrides the default ``grok-4``. ``XAI_API_KEY`` is
    required. An explicit argument wins over the environment. The SDK is
    imported only after the key is present.
    """

    key = api_key if api_key is not None else os.environ.get("XAI_API_KEY", "")
    name = model or os.environ.get("XAI_MODEL") or DEFAULT_MODEL
    if not key:
        raise RuntimeError("XAI_API_KEY is required to call the research model")
    from langchain_xai import ChatXAI

    return ChatXAI(model=name, api_key=key)
