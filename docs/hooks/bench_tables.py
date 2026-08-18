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

which renders only the full mass-multi-entry comparison.

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


def duration(seconds: float) -> str:
    """Format a duration at a readable scale.

    These tables span six orders of magnitude — a sub-millisecond build next to
    a solver run measured in hours — and one unit across all of them makes
    either end unreadable.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 600.0:
        return f"{seconds:.1f} s"
    if seconds < 5400.0:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} hr"


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
    """Render one scenario: how long each implementation took, and what it made."""
    ranked = sorted(measurements, key=lambda m: m["median"])
    best = ranked[0]

    detail = next((m["detail"] for m in measurements if m.get("detail")), "")
    params = best.get("params", {})
    requested = params.get("requested")
    lines = [
        f"**{case_name}**" + (f" — {detail}" if detail else ""),
        "",
        f"{requested:,} entries requested, {params.get('pool', '?')}-player slate."
        if requested
        else "",
        "",
        "| Implementation | Time | Relative | Returned | Players used | Overlap | Projection min → max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for m in ranked:
        same_yield = m.get("params", {}).get("produced") == best.get("params", {}).get("produced")
        if m is best:
            relative = "fastest"
        elif best["median"] and same_yield:
            # multiplication sign is the intended typography in the table.
            relative = f"{m['median'] / best['median']:.1f}× slower"  # noqa: RUF001
        else:
            # Unequal draws: dividing the times is not a ratio. The Returned
            # column carries the comparison.
            relative = "—"
        p = m.get("params", {})
        q = m.get("quality", {})
        returned = (
            f"{p['produced']:,} of {p['requested']:,}"
            if "produced" in p and "requested" in p
            else "—"
        )
        used = f"{q['distinct_players']} of {q['pool_size']}" if "distinct_players" in q else "—"
        overlap = f"{q['mean_overlap']:.0%}" if "mean_overlap" in q else "—"
        spread = (
            f"{q['worst_ratio']:.0%} → {q['best_ratio']:.0%}"
            if "worst_ratio" in q and "best_ratio" in q
            else "—"
        )
        lines.append(
            f"| `{m['impl']}` | {duration(m['median'])} | {relative} | "
            f"{returned} | {used} | {overlap} | {spread} |"
        )
    lines.append("")

    ours = next((m for m in measurements if m["impl"] == "mlb_dfs_solver_rust"), None)
    python = next((m for m in measurements if m["impl"] == "reference_python"), None)
    if case_name == "single-entry" and ours and python and python["median"] < ours["median"]:
        lines += [
            "!!! note",
            "    The pure-Python oracle outruns the kernel here: a microsecond "
            "job pays the kernel's fixed parallelism setup and gets nothing "
            "back. The row exists for the solver comparison — one roster is "
            "the case a solver wins.",
            "",
        ]
    if (
        case_name == "selection"
        and ours
        and ours["params"]["produced"] < ours["params"]["requested"]
    ):
        lines += [
            "!!! note",
            "    Tournament mode stops once no remaining candidate covers a new "
            "outcome, so a pool can be exhausted before the requested count is "
            "reached. That is the objective working, not a shortfall.",
            "",
        ]

    if len(measurements) == 1:
        lines += [
            "!!! warning",
            "    Only one implementation was measured for this case, so there is "
            "no baseline to compare against.",
            "",
        ]
    return lines


def render_detail_case(case_name: str, measurements: list[dict[str, Any]]) -> list[str]:
    """Every recorded measure for one scenario, per implementation."""
    lines: list[str] = []
    for m in measurements:
        quality = m.get("quality", {})
        if not quality:
            continue
        lines += [
            f'??? note "{case_name} — `{m["impl"]}`, every measure"',
            "",
            "    | Measure | Value |",
            "    | --- | ---: |",
            *[f"    | `{k}` | {v} |" for k, v in quality.items()],
            "",
        ]
    return lines


def render_result(result: dict[str, Any]) -> list[str]:
    """Render one machine's results as a heading plus a table per scenario."""
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
        "Each scenario is a configuration somebody plays, and both implementations "
        "are asked for the same number of entries on the same slate. Every figure "
        "describes the lineups or the inputs supplied — none of it scores a "
        "simulated contest.",
        "",
    ]

    for case_name, measurements in group_by_case(result["cases"]).items():
        lines += render_timing_case(case_name, measurements)
        lines += render_detail_case(case_name, measurements)

    lines += render_workflow(result)
    return lines


def render_workflow(result: dict[str, Any]) -> list[str]:
    """The whole-pipeline record: stage timings, then each portfolio described.

    Everything shown is a timing or a description of the lineups. No stage is
    scored against the contest — the field and the simulator are both harness
    fixtures, and a fixture cannot grade the thing it was written for.
    """
    workflow = result.get("workflow")
    if not workflow:
        return []
    sizes = workflow["sizes"]
    lines = [
        "### The whole job, timed",
        "",
        f"Build {sizes['candidates']:,} candidates, build a {sizes['field']:,}-entry "
        f"field under contest rules, simulate {sizes['outcomes']:,} outcomes, score "
        "everything, draw the lines, select a tournament portfolio and a cash "
        "portfolio.",
        "",
        "| Stage | Time |",
        "| --- | ---: |",
        *[f"| {s['name']} | {duration(s['seconds'])} |" for s in workflow["stages"]],
        f"| **total** | **{duration(workflow['total_seconds'])}** |",
        "",
        "The `draw lines` stage is `np.quantile` over the field's score matrix — "
        "caller-side NumPy that any pipeline pays regardless of what built the "
        "lineups.",
        "",
    ]
    for label in ("gpp", "cash"):
        described = workflow.get(label)
        if not described:
            continue
        lines += [
            f'??? note "workflow — the `{label}` portfolio, every measure"',
            "",
            "    | Measure | Value |",
            "    | --- | ---: |",
            *[f"    | `{k}` | {v} |" for k, v in described.items()],
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

    lines: list[str] = [
        "Every column is defined in "
        "[what the columns mean](methodology.md#what-the-columns-mean) — "
        "several read as more, or less, impressive than they are.",
        "",
    ]
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
    """Render the full-MME generated-lineups comparison, for a narrative page."""
    for result in latest_results():
        timed = [
            case
            for case in result["cases"]
            if case["name"] == "mme"
            and case.get("quality")
            and case.get("metric", "seconds") == "seconds"
        ]
        by_impl = {case["impl"]: case for case in timed}
        ours = by_impl.get("mlb_dfs_solver_rust")
        solver = by_impl.get("milp_ortools_cpsat")
        overlap_solver = by_impl.get("milp_ortools_cpsat_overlap")
        if not (ours and solver):
            continue

        rows = [("This package", ours), ("Solver + no-good cuts", solver)]
        if overlap_solver:
            shared = overlap_solver["params"].get("max_shared", "k")
            rows.append((f"Solver + max {shared}/10 shared", overlap_solver))
        hw = result["hardware"]
        lines = [
            f"*The `mme` scenario: {ours.get('detail', '')}. "
            f"{ours['params']['requested']} entries from each. "
            f"Measured on {hw['cpu']}, {result['timestamp'][:10]}.*",
            "",
            "| | Time | Players used | Projection, min → max | Top player's share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, case in rows:
            q = case["quality"]
            lines.append(
                f"| {label} | {duration(case['median'])} | "
                f"**{q['distinct_players']} of {q['pool_size']}** | "
                f"{q['worst_ratio']:.0%} → {q['best_ratio']:.0%} of optimum | "
                f"{q['max_exposure']:.0%} |"
            )
        lines.append("")
        return "\n".join(lines)

    return (
        "!!! note\n"
        "    No generated-lineup benchmark has been committed yet — see "
        "[How benchmarks work](methodology.md).\n"
    )


def on_page_markdown(markdown: str, **_kwargs: Any) -> str:
    """Substitute every benchmark placeholder on the page."""
    markdown = PLACEHOLDER.sub(lambda _m: render(), markdown)
    return HEADLINE.sub(lambda _m: render_headline(), markdown)
