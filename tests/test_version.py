"""The version is stated twice on purpose, and these keep the two in step.

`pyproject.toml` is canonical: it is what the release workflow compares the tag
against, and what installers record. `rust/Cargo.toml` has to restate it because
the compiled kernel reports `CARGO_PKG_VERSION` as `dfs_solver.native_version()`,
and nothing at build time forces the two to agree — a wheel whose kernel claims a
different version than its metadata is buildable, publishable, and confusing.

The Python source no longer states it at all; `__version__` is read from the
installed distribution.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import dfs_solver

REPO = Path(__file__).resolve().parent.parent


def _release(version: str) -> str:
    """Strip a PEP 440 pre/post/dev suffix, leaving `X.Y.Z`.

    Cargo versions are semver and cannot express `0.2.0.dev0`, so the two files
    can only be compared on the release part. That is the part that matters: it
    is what a published version is numbered.
    """
    return ".".join(version.split(".")[:3])


def _toml(relative: str) -> dict:
    with (REPO / relative).open("rb") as handle:
        return tomllib.load(handle)


def test_cargo_version_matches_pyproject():
    project = _toml("pyproject.toml")["project"]["version"]
    workspace = _toml("rust/Cargo.toml")["workspace"]["package"]["version"]
    assert workspace == _release(project)


def test_installed_version_matches_pyproject():
    """A stale install is the usual reason the other assertions here surprise you."""
    assert dfs_solver.__version__ == _toml("pyproject.toml")["project"]["version"]


def test_native_version_matches_package():
    """What `native_version` exists to catch, asserted rather than left to a user."""
    assert dfs_solver.native_version() == _release(dfs_solver.__version__)
