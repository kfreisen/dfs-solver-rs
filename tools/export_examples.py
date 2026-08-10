#!/usr/bin/env python3
"""Export the marimo examples into the built documentation site.

Two export modes, chosen in ``examples/manifest.toml``:

``wasm``
    ``marimo export html-wasm --mode run`` — a notebook that runs in the reader's
    browser through Pyodide. Genuinely interactive, and available only when every
    dependency has a Pyodide wheel. A package with a compiled extension cannot use
    this.

``static``
    ``marimo export html`` — pre-executed, outputs baked in. Works for anything.

Both modes execute the notebook, so a broken example fails the docs build rather
than publishing a page that errors when a reader opens it.

Usage::

    tools/export_examples.py --out site/examples
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODE_RE = re.compile(r'^\s*mode\s*=\s*"(?P<mode>wasm|static)"\s*$', re.MULTILINE)


def read_mode(manifest: Path) -> str:
    """Read the export mode from the examples manifest."""
    if not manifest.exists():
        return "static"
    match = MODE_RE.search(manifest.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f'{manifest}: no `mode = "wasm"|"static"` found in [export]')
    return match.group("mode")


def export_command(mode: str, notebook: Path, out_file: Path) -> list[str]:
    """Build the marimo command line for ``mode``."""
    base = ["marimo", "export"]
    if mode == "wasm":
        return [*base, "html-wasm", "--mode", "run", str(notebook), "-o", str(out_file)]
    return [*base, "html", str(notebook), "-o", str(out_file)]


def main() -> int:
    """Export every example notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    examples = sorted((REPO_ROOT / "examples").glob("*.py"))
    if not examples:
        print("no examples to export", file=sys.stderr)
        return 0

    mode = read_mode(REPO_ROOT / "examples" / "manifest.toml")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {len(examples)} example(s) as {mode}", file=sys.stderr)

    for notebook in examples:
        out_file = args.out / f"{notebook.stem}.html"
        # cwd is the repo root so a notebook can read examples/data/ by relative path.
        result = subprocess.run(
            export_command(mode, notebook.relative_to(REPO_ROOT), out_file.resolve()),
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"{notebook.relative_to(REPO_ROOT)}: export failed. The notebook is "
                f"executed during export, so this is usually the example itself "
                f"raising, not a marimo problem.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
