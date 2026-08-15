#!/usr/bin/env python3
"""Normalize a pytest-benchmark JSON dump into this repo's committed result schema.

The package benchmarks against one or more baseline implementations kept under
``benchmarks/baselines/``, run over identical cases. pytest-benchmark's own JSON is
faithful but verbose and version-dependent, so it is not what gets committed. This
flattens it into a small stable schema the docs site reads at build time.

Benchmarks declare which case and implementation they are via ``extra_info``::

    def test_fast_path(benchmark):
        benchmark.extra_info["case"] = "cooldown/500sym"
        benchmark.extra_info["impl"] = "polars_asof"
        benchmark.extra_info["params"] = {"symbols": 500}
        benchmark(run, frame)

Cases are matched across implementations by their ``case`` string, which is what
makes a speedup table possible. A case measured for only one implementation is
reported, but flagged — it has no baseline to be faster than.

Usage::

    tools / bench_report.py.benchmark.json - -write
    tools / bench_report.py.benchmark.json  # stdout only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


def _run(cmd: list[str], default: str = "unknown") -> str:
    """Return the stripped stdout of ``cmd``, or ``default`` if it fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (subprocess.SubprocessError, OSError):
        return default
    return out.stdout.strip() or default


def cpu_model() -> str:
    """Return a human-readable CPU model name."""
    if sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    elif sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or platform.machine() or "unknown"


def ram_gb() -> float | None:
    """Return total system RAM in GiB, or None if it cannot be determined."""
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.virtual_memory().total / 1024**3, 1)


def hardware_id(cpu: str) -> str:
    """Derive a short filesystem-safe slug identifying this machine.

    Results are committed per machine, so this becomes a directory name and appears
    in published tables. It needs to be stable across runs on the same box and
    legible to someone reading the docs — hence CPU model plus OS, not a hash.
    """
    tokens = re.sub(r"\((R|TM|r|tm)\)|CPU|Processor|@.*", " ", cpu)
    tokens = re.sub(r"[^A-Za-z0-9]+", "-", tokens).strip("-").lower()
    tokens = re.sub(r"^(intel|amd|apple)-", "", tokens)
    return f"{tokens or 'unknown'}-{platform.system().lower()}"


def collect_hardware() -> dict[str, Any]:
    """Describe the machine this benchmark ran on."""
    cpu = cpu_model()
    return {
        "id": hardware_id(cpu),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def git_sha() -> str:
    """Return the current commit SHA, suffixed with ``-dirty`` if the tree is modified.

    Results files are excluded from the dirtiness check. They are this script's own
    output, so counting them would mark every run dirty the moment it wrote
    anything — and the flag is meant to warn that the *measured code* is
    uncommitted, which is a different question.
    """
    sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], default="")
    changed = [
        line for line in status.splitlines() if line.strip() and "benchmarks/results/" not in line
    ]
    return f"{sha}-dirty" if changed else sha


def project() -> tuple[str, str]:
    """Return this package's name and version from pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        info = tomllib.load(fh)["project"]
    return info["name"], info["version"]


def convert_cases(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten pytest-benchmark's ``benchmarks`` array into this repo's case schema."""
    cases: list[dict[str, Any]] = []
    for bench in raw.get("benchmarks", []):
        info = bench.get("extra_info") or {}
        stats = bench["stats"]
        missing = {"case", "impl"} - info.keys()
        if missing:
            raise SystemExit(
                f"benchmark {bench.get('fullname', bench.get('name'))!r} is missing "
                f"extra_info {sorted(missing)}. Every benchmark must declare which case "
                f"and which implementation it measures, or results cannot be compared."
            )
        case = {
            "name": info["case"],
            "impl": info["impl"],
            "params": info.get("params", {}),
            "metric": info.get("metric", "seconds"),
            "n": stats["rounds"],
            "mean": stats["mean"],
            "median": stats["median"],
            "stddev": stats["stddev"],
            "min": stats["min"],
        }
        # Optional, and carried through rather than dropped: `detail` is the
        # one-line description of a constraint rung, and `quality` is a bag of
        # score and diversity figures for benchmarks that measure goodness rather
        # than time. The docs site renders both.
        for key in ("detail", "quality"):
            if key in info:
                case[key] = info[key]
        cases.append(case)
    return cases


def speedup_table(cases: list[dict[str, Any]], fastest_impl: str | None) -> list[str]:
    """Render a human-readable speedup summary, grouped by case.

    Cases whose metric is not a time are left out. A quality benchmark runs a
    no-op through the harness so its row exists in the report, and ranking that
    stopwatch reading against a real measurement would put a meaningless number at
    the top of the table.
    """
    by_case: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if case.get("metric", "seconds") != "seconds":
            continue
        by_case.setdefault(case["name"], []).append(case)

    lines: list[str] = []
    for name, group in sorted(by_case.items()):
        lines.append(f"\n{name}")
        ranked = sorted(group, key=lambda c: c["median"])
        best = fastest_impl or ranked[0]["impl"]
        reference = next((c for c in group if c["impl"] == best), ranked[0])
        for case in ranked:
            ratio = case["median"] / reference["median"] if reference["median"] else float("nan")
            marker = "  (reference)" if case is reference else f"  {ratio:8.2f}x slower"
            lines.append(f"  {case['impl']:<22} {case['median']:12.6f}s{marker}")
        if len(group) == 1:
            lines.append("  ^ only one implementation measured — no baseline to compare against")
    return lines


README_BEGIN = "<!-- begin:benchmark-summary -->"
README_END = "<!-- end:benchmark-summary -->"


def _duration(seconds: float) -> str:
    """Format a duration at a readable scale.

    The tables here span six orders of magnitude — a sub-millisecond build next
    to a solver run measured in hours — and one unit across all of them makes
    either end unreadable.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 90.0:
        return f"{seconds:.1f} s"
    if seconds < 5400.0:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hr"


def readme_summary(report: dict[str, Any]) -> str:
    """Render the headline tables for the README.

    The docs site renders the full results from the committed JSON at build time,
    which the README cannot do — GitHub and PyPI run no hooks. So the summary is
    *generated into* the README between markers rather than typed, and `task bench`
    rewrites it. The rule the rest of this repository follows still holds: nobody
    types a number into a document, so no document can disagree with the data.

    Deliberately short. It is the first thing a reader sees, and the full ladder,
    every rung and every implementation belongs on the benchmarks page.
    """
    hw = report["hardware"]
    lines = [
        README_BEGIN,
        "",
        f"{hw['cpu']}, {hw['cores']} cores · {hw['os']} · measured "
        f"{report['timestamp'][:10]} against `{report['git_sha']}`.",
        "",
    ]

    # What each implementation generates, at portfolio size. Timed cases only:
    # the head-to-head case shares the prefix, carries a differently shaped
    # `quality` bag and is not a timing.
    generated = {
        c["impl"]: c
        for c in report["cases"]
        if c["name"] == "generated/stack"
        and c.get("quality")
        and c.get("metric", "seconds") == "seconds"
    }
    ours, solver = generated.get("slatekit_rust"), generated.get("milp_ortools_cpsat")
    if ours and solver:
        lines += [
            f"**What each one generates.** {ours.get('detail', '')}, "
            f"{ours['params']['requested']} lineups from each. The solver returns "
            "the top rosters by projection, so its entries cluster at the optimum "
            "and are built from a fraction of the slate. Which output you want "
            "depends on the contest; both halves are visible here.",
            "",
            "| | Time | Players used | Projection, min → max | Top player's share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, case in (("This package", ours), ("Solver + no-good cuts", solver)):
            q = case["quality"]
            lines.append(
                f"| {label} | {_duration(case['median'])} | "
                f"**{q['distinct_players']} of {q['pool_size']}** | "
                f"{q['worst_ratio']:.0%} → {q['best_ratio']:.0%} of optimum | "
                f"{q['max_exposure']:.0%} |"
            )
        lines.append("")

    contest = {
        c["impl"]: c
        for c in report["cases"]
        if c["name"].startswith("contest/") and c.get("metric", "seconds") == "seconds"
    }
    ours, solver = contest.get("slatekit_rust"), contest.get("milp_ortools_cpsat")
    if ours and solver:
        # The solver is measured on a prefix; its full cost is arithmetic on that
        # rate and is labelled as such rather than presented as a measurement.
        prefix = solver["params"]["requested"]
        target = solver["params"].get("extrapolated_to", 0)
        per_lineup = solver["median"] / max(prefix, 1)
        lines += [
            f"**At contest scale.** {ours.get('detail', '')} — the draw a field "
            "simulation or a candidate pool needs. Here the two are not "
            "alternatives.",
            "",
            "| | Lineups | Time |",
            "| --- | ---: | ---: |",
            f"| This package | {ours['params']['produced']:,} | {_duration(ours['median'])} |",
            f"| Solver + no-good cuts | {prefix} measured | {_duration(solver['median'])} |",
            f"| Solver, extrapolated | {target:,} | ~{_duration(per_lineup * target)} |",
            "",
        ]

    ladder = [c for c in report["cases"] if c["name"].startswith("constraints/")]
    if ladder:
        by_case: dict[str, dict[str, dict[str, Any]]] = {}
        for case in ladder:
            by_case.setdefault(case["name"], {})[case["impl"]] = case
        lines += [
            "**Cost of each constraint.** Every implementation asked for the same "
            "portfolio on the same slate, one constraint added per row.",
            "",
            # Both yields, not one. A speedup that divides our time for 14
            # lineups by the solver's time for 25 is not a ratio, and a single
            # "Returned" column silently invited exactly that reading.
            "| Constraint | This | Pure Python | CP-SAT | Speedup | Ours | CP-SAT's |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, impls in by_case.items():
            ours = impls.get("slatekit_rust")
            if ours is None:
                continue
            python = impls.get("reference_python")
            solver = impls.get("milp_ortools_cpsat")
            params = ours.get("params", {})

            detail = ours.get("detail", name.split("/", 1)[-1])
            ours_ms = f"{ours['median'] * 1e3:.2f} ms"
            python_ms = f"{python['median'] * 1e3:.1f} ms" if python else "—"
            solver_ms = f"{solver['median'] * 1e3:,.0f} ms" if solver else "—"
            speedup = (
                f"{solver['median'] / ours['median']:,.0f}×"  # noqa: RUF001
                if solver and ours["median"]
                else "—"
            )
            returned = f"{params.get('produced', '—')} of {params.get('requested', '—')}"
            solver_params = solver.get("params", {}) if solver else {}
            solver_returned = (
                f"{solver_params.get('produced', '—')} of {solver_params.get('requested', '—')}"
                if solver
                else "—"
            )
            lines.append(
                f"| {detail} | {ours_ms} | {python_ms} | {solver_ms} | {speedup} | "
                f"{returned} | {solver_returned} |"
            )
        lines.append("")

    quality = [c for c in report["cases"] if c["name"].startswith("quality/") and c.get("quality")]
    rows = [(c, c["quality"]) for c in quality if "median_ratio" in c["quality"]]
    if rows:
        lines += [
            "**Are they any good, and are they different?** Scored against the "
            "optimum CP-SAT proves, and compared with the solver's own portfolio at "
            "matched size.",
            "",
            "| Slate | Best vs optimum | Median vs optimum | Overlap within | "
            "Solver's overlap | Shared with solver | Identical |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for case, q in rows:
            lines.append(
                f"| {case.get('detail', case['name'])} | {q['best_ratio']:.0%} | "
                f"{q['median_ratio']:.0%} | {q['self_overlap_matched']:.0%} | "
                f"{q['solver_self_overlap']:.0%} | {q['overlap_with_solver']:.0%} | "
                f"{q['identical_to_solver']} |"
            )
        lines += ["", README_END, ""]
    else:
        lines += [README_END, ""]
    return "\n".join(lines)


def update_readme(report: dict[str, Any]) -> bool:
    """Splice the generated summary into README.md. Returns whether it changed."""
    path = REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        print(
            f"warning: README.md has no {README_BEGIN} / {README_END} markers, so the "
            f"summary was not written.",
            file=sys.stderr,
        )
        return False
    head, _, rest = text.partition(README_BEGIN)
    _, _, tail = rest.partition(README_END)
    # Exactly one blank line after the closing marker, however many the previous
    # version had: markdown needs the break or the next paragraph joins the marker.
    body = readme_summary(report).removesuffix("\n")
    updated = head + body + "\n\n" + tail.lstrip("\n")
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    """Convert a pytest-benchmark dump and optionally commit it under benchmarks/results/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="pytest-benchmark --benchmark-json output")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write into benchmarks/results/<hardware-id>/ instead of stdout",
    )
    parser.add_argument(
        "--reference-impl",
        default=None,
        help="implementation to treat as the reference in the printed table "
        "(default: the fastest measured)",
    )
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    hardware = collect_hardware()
    name, version = project()
    report = {
        "schema": SCHEMA_VERSION,
        "package": name,
        "version": version,
        "git_sha": git_sha(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware,
        "cases": convert_cases(raw),
    }

    if not report["cases"]:
        raise SystemExit(f"{args.input}: no benchmarks found")

    print("\n".join(speedup_table(report["cases"], args.reference_impl)), file=sys.stderr)

    if not args.write:
        print(json.dumps(report, indent=2))
        return 0

    date = report["timestamp"][:10]
    out_dir = REPO_ROOT / "benchmarks" / "results" / hardware["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}-{report['git_sha']}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    if update_readme(report):
        print("updated README.md benchmark summary", file=sys.stderr)

    if "-dirty" in report["git_sha"]:
        print(
            "warning: working tree is dirty, so this result is not reproducible from a "
            "commit. Re-run on a clean tree before committing the JSON.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
