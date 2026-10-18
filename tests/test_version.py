import tomllib
from pathlib import Path

from driftwood import __version__


def test_package_version_matches_pyproject() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == __version__
