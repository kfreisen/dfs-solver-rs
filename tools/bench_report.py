"""Render a committed benchmark result into the README summary.

`benchmarks/run.py` produces the result file; this only reads one and renders it.
The split matters: the runner is the thing that must be careful about timing, and
this is the thing that must be careful about not putting a number in a document
by hand.

GitHub and PyPI run no mkdocs hooks, so the README summary is generated *into*
the file between markers rather than at build time. The docs site renders the
same data through `docs/hooks/bench_tables.py`.

Usage:

    python tools/bench_report.py                        # newest result, print
    python tools/bench_report.py --write                # ... and update README
    python tools/bench_report.py path/to/result.json --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
README = REPO_ROOT / "README.md"
README_BEGIN = "<!-- begin:benchmark-summary -->"
README_END = "<!-- end:benchmark-summary -->"

# The scenario the "what gets generated" table describes. The full mass-multi-entry
# configuration, because that is the one anybody actually runs.
HEADLINE_SCENARIO = "mme"
CONTEST_SCENARIO = "contest-scale"

OURS = "mlb_dfs_solver_rust"
PYTHON = "reference_python"
SOLVER = "milp_ortools_cpsat"


def duration(seconds: float) -> str:
    """Format a duration at a readable scale.

    These tables span six orders of magnitude — a sub-millisecond build next to a
    solver run measured in hours — and one unit across all of them makes either
    end unreadable.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 600.0:
        return f"{seconds:.1f} s"
    if seconds < 5400.0:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} hr"


def latest_result() -> Path | None:
    """The newest committed result across every machine."""
    files = sorted(RESULTS.glob("*/*.json"))
    if not files:
        return None
    return max(files, key=lambda f: json.loads(f.read_text(encoding="utf-8"))["timestamp"])


def by_scenario(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group cases as `{scenario: {impl: case}}`, preserving scenario order."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for case in report["cases"]:
        grouped.setdefault(case["name"], {})[case["impl"]] = case
    return grouped


def readme_summary(report: dict[str, Any]) -> str:
    """The block between the README markers."""
    hw = report["hardware"]
    grouped = by_scenario(report)
    lines = [
        README_BEGIN,
        "",
        f"{hw['cpu']}, {hw['cores']} cores · {hw['os']} · measured "
        f"{report['timestamp'][:10]} against `{report['git_sha']}`.",
        "",
        "**Every scenario is a way somebody plays.** Cumulative: each row adds one "
        "thing a player turns on. Both implementations are asked for the same "
        "number of entries on the same slate — 150 is a DraftKings MLB classic "
        "maximum, cash is played a few entries deep.",
        "",
        "| Scenario | Entries | This | Pure Python | CP-SAT | Speedup | Ours | CP-SAT's |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_scenario_rows(grouped),
    ]

    headline = grouped.get(HEADLINE_SCENARIO, {})
    ours, solver = headline.get(OURS), headline.get(SOLVER)
    if ours and solver:
        lines += [
            "",
            f"**What each one generates.** The `{HEADLINE_SCENARIO}` row above, "
            "described. The solver returns the top rosters by projection, so its "
            "entries cluster at the optimum and are built from a fraction of the "
            "slate. Which output you want depends on the contest.",
            "",
            "| | Players used | Mean overlap | In >50% of entries | Projection, min → max | Entries |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, case in (("This package", ours), ("Solver + no-good cuts", solver)):
            q = case["quality"]
            overlap = f"{q['mean_overlap']:.0%}" if "mean_overlap" in q else "—"
            lines.append(
                f"| {label} | **{q['distinct_players']} of {q['pool_size']}** | "
                f"{overlap} | "
                f"{q['players_over_50pct']} | "
                f"{q.get('worst_ratio', 0):.0%} → {q.get('best_ratio', 0):.0%} of optimum | "
                f"{q['lineups']:,} |"
            )

    contest = grouped.get(CONTEST_SCENARIO, {})
    ours, solver = contest.get(OURS), contest.get(SOLVER)
    if ours:
        lines += [
            "",
            "**At contest scale.** The same configuration at field size — the draw "
            "a field simulation or a candidate pool needs, and where the two stop "
            "being alternatives. Measured, not extrapolated.",
            "",
            "| | Lineups | Time | Per lineup |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label, case in (("This package", ours), ("Solver + no-good cuts", solver)):
            if case is None:
                continue
            produced = case["params"]["produced"]
            budget = case["params"].get("budget_s")
            if budget and produced < case["params"]["requested"]:
                # A truncated run presented as a completed one would understate
                # the solver; say what the budget bought instead.
                label = f"{label} ({duration(budget)} budget)"
            lines.append(
                f"| {label} | {produced:,} | {duration(case['median'])} | "
                f"{duration(case['median'] / max(produced, 1))} |"
            )

    lines += _workflow_section(report)
    lines += ["", README_END]
    return "\n".join(lines)


def _workflow_section(report: dict[str, Any]) -> list[str]:
    """The whole job, timed — one row per pipeline stage.

    Descriptive only: stage timings and sizes. The portfolio-quality measures in
    the `workflow` record stay on the docs site, where the caveats around them
    have room to be stated.
    """
    workflow = report.get("workflow")
    if not workflow:
        return []
    sizes = workflow["sizes"]
    lines = [
        "",
        "**The whole job, timed.** What a tournament player actually runs: build "
        f"{sizes['candidates']:,} candidates under a stacked, conflict-ruled spec, "
        f"build a {sizes['field']:,}-entry field under contest rules, simulate "
        f"{sizes['outcomes']:,} outcomes, score everything, draw the lines, and "
        "select a 150-entry tournament portfolio and a 20-entry cash portfolio.",
        "",
        "| Stage | Time |",
        "| --- | ---: |",
        *[f"| {s['name']} | {duration(s['seconds'])} |" for s in workflow["stages"]],
        f"| **total** | **{duration(workflow['total_seconds'])}** |",
    ]
    return lines


def _scenario_rows(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """One table row per scenario, both yields shown.

    Both, because a speedup dividing our time for 14 lineups by a solver's time
    for 25 is not a ratio, and a single "returned" column invites that reading.
    """
    kept: list[str] = []
    for name, impls in grouped.items():
        ours = impls.get(OURS)
        if ours is None:
            continue
        python, solver = impls.get(PYTHON), impls.get(SOLVER)
        speedup = (
            f"{solver['median'] / ours['median']:,.0f}×"  # noqa: RUF001
            if solver and ours["median"]
            else "—"
        )
        solver_returned = (
            f"{solver['params']['produced']:,} of {solver['params']['requested']:,}"
            if solver
            else "—"
        )
        kept.append(
            f"| {name} | {ours['params']['requested']:,} "
            f"| {duration(ours['median'])} "
            f"| {duration(python['median']) if python else '—'} "
            f"| {duration(solver['median']) if solver else '—'} "
            f"| {speedup} "
            f"| {ours['params']['produced']:,} of {ours['params']['requested']:,} "
            f"| {solver_returned} |"
        )
    return kept


def update_readme(report: dict[str, Any]) -> bool:
    """Rewrite the block between the markers. Returns whether anything changed."""
    text = README.read_text(encoding="utf-8")
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start == -1 or end == -1:
        msg = f"{README} is missing the benchmark summary markers"
        raise SystemExit(msg)
    updated = text[:start] + readme_summary(report) + text[end + len(README_END) :]
    if updated == text:
        return False
    README.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    """Render the newest (or named) result, optionally writing it into the README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", nargs="?", type=Path, default=None, help="result JSON; default is the newest."
    )
    parser.add_argument("--write", action="store_true", help="update the README summary.")
    args = parser.parse_args()

    path = args.result or latest_result()
    if path is None:
        print("no committed results found; run benchmarks/run.py first", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))

    print(readme_summary(report))
    if args.write and update_readme(report):
        print(f"\nupdated {README.name} benchmark summary", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
