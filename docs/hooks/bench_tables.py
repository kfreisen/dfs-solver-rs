"""mkdocs hook: render committed benchmark results into documentation tables.

A benchmark number in prose drifts from the data behind it within about one release.
So no page in this site writes a speedup by hand — pages carry a placeholder::

    <!-- benchmarks -->

and this hook replaces it with tables built from
``benchmarks/results/<hardware-id>/<date>-<sha>.json`` at build time.
If the JSON is not committed, the page says so rather than showing a stale figure.

A narrative page wants one comparison rather than all of them, and quoting it in
prose would be exactly the hand-typed number this exists to prevent. So there is a
second, smaller placeholder::

    <!-- headline -->

which renders only the end-to-end pipeline table.

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
# A second, smaller placeholder. A narrative page wants the one comparison the
# whole package rests on, not sixty rows of it — and quoting that comparison in
# prose would put typed numbers in a document, which is the thing this hook
# exists to prevent.
HEADLINE = re.compile(r"^<!--\s*headline\s*-->\s*$", re.MULTILINE)


def latest_results() -> list[dict[str, Any]]:
    """Return the newest committed result per hardware id, newest hardware first."""
    results_dir = REPO_ROOT / "benchmarks" / "results"
    if not results_dir.is_dir():
        return []

    newest: list[dict[str, Any]] = []
    for hw_dir in sorted(results_dir.iterdir()):
        if not hw_dir.is_dir():
            continue
        results = [json.loads(f.read_text(encoding="utf-8")) for f in hw_dir.glob("*.json")]
        if not results:
            continue
        # Ordered by the timestamp inside the file, not by filename. Filenames
        # lead with an ISO date and *end* with a commit sha, so two runs on the
        # same day sort by sha — which is arbitrary. That silently published a
        # superseded result: `2026-08-11-1244693` lost to the older
        # `2026-08-11-cd3076c` because "1" sorts before "c".
        newest.append(max(results, key=lambda r: r["timestamp"]))
    return newest


def group_by_case(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket measurements by the case they measured, preserving first-seen order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["name"], []).append(case)
    return grouped


def render_timing_case(case_name: str, measurements: list[dict[str, Any]]) -> list[str]:
    """Render one case as a table ranked fastest first."""
    ranked = sorted(measurements, key=lambda m: m["median"])
    best = ranked[0]
    metric = best.get("metric", "seconds")
    unit = "s" if metric == "seconds" else ""
    # Yield is only meaningful where a benchmark recorded it, and only worth a
    # column where something actually fell short: a table of 1.00 teaches nothing.
    show_yield = any(m.get("params", {}).get("yield", 1.0) < 1.0 for m in measurements)

    detail = next((m["detail"] for m in measurements if m.get("detail")), "")
    lines = [f"**{case_name}**" + (f" — {detail}" if detail else ""), ""]

    header = f"| Implementation | Median ({metric}) | Relative |"
    divider = "| --- | ---: | ---: |"
    if show_yield:
        header += " Lineups returned |"
        divider += " ---: |"
    lines += [header, divider]

    for m in ranked:
        if m is best:
            relative = "fastest"
        elif best["median"]:
            # multiplication sign is the intended typography in the table.
            relative = f"{m['median'] / best['median']:.1f}× slower"  # noqa: RUF001
        else:
            relative = "—"
        row = f"| `{m['impl']}` | {m['median']:.6f}{unit} | {relative} |"
        if show_yield:
            params = m.get("params", {})
            produced = params.get("produced")
            requested = params.get("requested")
            row += (
                f" {produced} of {requested} |"
                if produced is not None and requested is not None
                else " — |"
            )
        lines.append(row)
    lines.append("")

    if len(measurements) == 1:
        lines += [
            "!!! warning",
            "    Only one implementation was measured for this case, so there is "
            "no baseline to compare against.",
            "",
        ]
    return lines


def render_quality_case(case_name: str, measurements: list[dict[str, Any]]) -> list[str]:
    """Render a case that measured goodness rather than time.

    Kept apart from the timing tables and rendered as plain key/value rows,
    because these are not comparable to each other and ranking them by the
    stopwatch would be meaningless — the benchmark times a no-op.
    """
    lines: list[str] = []
    for m in measurements:
        detail = f" — {m['detail']}" if m.get("detail") else ""
        lines += [
            f"**{case_name}**{detail}",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
        ]
        lines += [f"| `{k}` | {v} |" for k, v in m.get("quality", {}).items()]
        lines.append("")
    return lines


def render_result(result: dict[str, Any]) -> list[str]:
    """Render one machine's results as a heading plus a table per case."""
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

    grouped = group_by_case(result["cases"])
    timing = {k: v for k, v in grouped.items() if v[0].get("metric", "seconds") == "seconds"}
    quality = {k: v for k, v in grouped.items() if v[0].get("metric", "seconds") != "seconds"}

    for case_name, measurements in timing.items():
        lines += render_timing_case(case_name, measurements)

    if quality:
        lines += [
            "#### Quality",
            "",
            "Speed on its own would be misleading: a generator of a hundred and fifty "
            "near-identical lineups is worthless for a large-field contest, and so is "
            "one that returns diverse rubbish. These are the numbers that say which "
            "this is.",
            "",
        ]
        for case_name, measurements in quality.items():
            lines += render_quality_case(case_name, measurements)

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
        "task bench",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_headline() -> str:
    """Render the end-to-end comparison alone, for a narrative page."""
    for result in latest_results():
        timed = [
            case
            for case in result["cases"]
            if case["name"].startswith("pipeline/")
            and case.get("quality")
            and case.get("metric", "seconds") == "seconds"
        ]
        by_impl = {case["impl"]: case for case in timed}
        ours = by_impl.get("slatekit_rust")
        solver = by_impl.get("milp_ortools_cpsat")
        if not (ours and solver):
            continue

        hw = result["hardware"]
        lines = [
            f"*{ours.get('detail', '')}. Measured on {hw['cpu']}, {result['timestamp'][:10]}.*",
            "",
            "| | Time | Median entry vs optimum | Overlap between entries | In the money |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, case in (("This package", ours), ("Solver + no-good cuts", solver)):
            q = case["quality"]
            lines.append(
                f"| {label} | {case['median']:.2f} s | {q['median_ratio']:.0%} | "
                f"{q['overlap']:.0%} | **{q['p_in_the_money']:.0%}** |"
            )
        lines.append("")
        return "\n".join(lines)

    return (
        "!!! note\n"
        "    No end-to-end benchmark has been committed yet — see "
        "[How benchmarks work](methodology.md).\n"
    )


def on_page_markdown(markdown: str, **_kwargs: Any) -> str:
    """Substitute every benchmark placeholder on the page."""
    markdown = PLACEHOLDER.sub(lambda _m: render(), markdown)
    return HEADLINE.sub(lambda _m: render_headline(), markdown)
