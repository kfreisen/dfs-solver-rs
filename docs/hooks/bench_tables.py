"""mkdocs hook: render committed benchmark results into documentation tables.

A benchmark number in prose drifts from the data behind it within about one release.
So no page in this site writes a speedup by hand — pages carry a placeholder::

    <!-- benchmarks -->

and this hook replaces it with tables built from
``benchmarks/results/<hardware-id>/<date>-<sha>.json`` at build time.
If the JSON is not committed, the page says so rather than showing a stale figure.

Results are grouped by hardware, because a speedup measured on one machine is a claim
about that machine. The most recent file per hardware wins; older files stay in the
repository as history.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLACEHOLDER = re.compile(r"^<!--\s*benchmarks\s*-->\s*$", re.MULTILINE)


def latest_results() -> list[dict[str, Any]]:
    """Return the newest committed result per hardware id, newest hardware first."""
    results_dir = REPO_ROOT / "benchmarks" / "results"
    if not results_dir.is_dir():
        return []

    newest: list[dict[str, Any]] = []
    for hw_dir in sorted(results_dir.iterdir()):
        if not hw_dir.is_dir():
            continue
        files = sorted(hw_dir.glob("*.json"))
        if not files:
            continue
        # Filenames lead with an ISO date, so lexical order is chronological.
        newest.append(json.loads(files[-1].read_text(encoding="utf-8")))
    return newest


def group_by_case(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket measurements by the case they measured, preserving first-seen order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["name"], []).append(case)
    return grouped


def render_result(result: dict[str, Any]) -> list[str]:
    """Render one machine's results as a heading plus a table per unit of measure."""
    hw = result["hardware"]
    ram = f", {hw['ram_gb']} GB RAM" if hw.get("ram_gb") else ""
    lines = [
        f"### `{hw['id']}`",
        "",
        f"{hw['cpu']} · {hw['cores']} cores{ram} · {hw['os']} · Python {hw['python']}",
        "",
        f"Measured {result['timestamp'][:10]} against `{result['git_sha']}` "
        f"of `{result['package']} {result['version']}`.",
        "",
    ]

    for case_name, measurements in group_by_case(result["cases"]).items():
        ranked = sorted(measurements, key=lambda m: m["median"])
        best = ranked[0]
        metric = best.get("metric", "seconds")
        unit = "s" if metric == "seconds" else ""

        lines += [
            f"**{case_name}**",
            "",
            f"| Implementation | Median ({metric}) | Relative |",
            "| --- | ---: | ---: |",
        ]
        for m in ranked:
            if m is best:
                relative = "fastest"
            elif best["median"]:
                # multiplication sign is the intended typography in the table.
                relative = f"{m['median'] / best['median']:.1f}× slower"  # noqa: RUF001
            else:
                relative = "—"
            lines.append(f"| `{m['impl']}` | {m['median']:.6f}{unit} | {relative} |")
        lines.append("")

        if len(measurements) == 1:
            lines += [
                "!!! warning",
                "    Only one implementation was measured for this case, so there is "
                "no baseline to compare against.",
                "",
            ]

    return lines


def render() -> str:
    """Render every committed result for ``package``, or an explanatory note."""
    results = latest_results()
    if not results:
        return (
            "!!! note\n"
            "    No benchmark results have been committed yet. "
            "They are recorded on a known machine and checked in — see "
            "[How benchmarks work](methodology.md).\n"
        )

    lines: list[str] = []
    for result in results:
        lines += render_result(result)
    lines += [
        "Reproduce with:",
        "",
        "```bash",
        "make bench",
        "```",
        "",
    ]
    return "\n".join(lines)


def on_page_markdown(markdown: str, **_kwargs: Any) -> str:
    """Substitute every benchmark placeholder on the page."""
    return PLACEHOLDER.sub(lambda _m: render(), markdown)
