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
        cases.append(
            {
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
        )
    return cases


def speedup_table(cases: list[dict[str, Any]], fastest_impl: str | None) -> list[str]:
    """Render a human-readable speedup summary, grouped by case."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
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

    if "-dirty" in report["git_sha"]:
        print(
            "warning: working tree is dirty, so this result is not reproducible from a "
            "commit. Re-run on a clean tree before committing the JSON.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
