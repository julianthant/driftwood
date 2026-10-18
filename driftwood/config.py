"""Paths and environment loading.

Credentials are read from the environment only. A local ``.env`` file is
optional and is never required for tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load a ``.env`` file from the working directory if one exists.

    Existing environment variables win, so a shell export is not overwritten
    by a stale file.
    """

    load_dotenv(override=False)


def database_path(explicit: str | Path | None = None) -> Path:
    """Resolve the DuckDB file path.

    Preference order: an explicit argument, ``DRIFTWOOD_DB``, then
    ``data/driftwood.duckdb`` under the current working directory.
    """

    if explicit:
        return Path(explicit)
    env = os.environ.get("DRIFTWOOD_DB")
    if env:
        return Path(env)
    return Path("data") / "driftwood.duckdb"
