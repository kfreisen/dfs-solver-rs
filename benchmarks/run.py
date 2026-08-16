"""Run the benchmarks and write a result file.

Deliberately not pytest. These are long — the contest-scale row alone is ten
thousand sequential CP-SAT solves — and pytest is a test runner, so putting them
there produced two bad outcomes. CI executed them on every push, because
`--benchmark-disable` stops the timing and not the work. And the portfolio size
was pinned at 25 lineups, a number nobody plays, chosen because it was the
largest a solver could finish inside a test session. A harness constraint had
become the published experiment.

Timing is a plain `perf_counter` around each call, repeated where a case is fast
enough for repetition to mean anything. There is no warmup subtlety to get wrong
here: the kernel does no JIT, allocates its own buffers, and every case is far
longer than a timer tick.

Usage:

    python benchmarks/run.py                  # every scenario, write a result
    python benchmarks/run.py --smoke          # one tiny pass, prove it executes
    python benchmarks/run.py -s mme -s stack  # named scenarios only
    python benchmarks/run.py --no-solver      # skip the MILP baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.milp import solve_milp_ortools, solve_portfolio_ortools
from baselines.reference import build_lineups_reference
from baselines.selection import select_portfolio_reference
from mlb_dfs_solver import __version__, build_lineups, score_lineups, select_portfolio
from scenarios import SCENARIO_NAMES, Scenario, build_scenarios, scenario_by_name
from simulate import simulate
from slates import describe_lineups
from workflow import run_workflow

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
# Schema 2 adds the top-level `workflow` record and the `selection` cases; every
# schema-1 field keeps its meaning, and the renderers read both.
SCHEMA = 2

# Per solve. Generous enough that nothing here hits it; present so one pathological
# instance cannot stall a run for hours.
SOLVER_LIMIT_S = 30.0
# The pure-Python oracle is ~15x slower than the kernel and proves the same
# output; past this many entries it adds wall clock and no information.
REFERENCE_MAX_ENTRIES = 150
# A case shorter than this is repeated until the total passes it, so a
# sub-millisecond measurement is not one timer tick wide.
TARGET_SECONDS = 2.0
MAX_ROUNDS = 20


def measure(fn, *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """Time `fn`, repeating while it is short. Returns (median seconds, result).

    The median rather than the minimum: the minimum reports the luckiest run,
    which is the right statistic for a microbenchmark on a quiet machine and the
    wrong one for a number somebody will use to predict how long their slate
    takes.
    """
    timings: list[float] = []
    result = None
    total = 0.0
    while len(timings) < MAX_ROUNDS:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        total += elapsed
        if total >= TARGET_SECONDS:
            break
    return float(np.median(timings)), result


def _run(cmd: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def cpu_model() -> str:
    """The CPU's marketing name, since a speedup is a claim about a machine."""
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"], platform.processor())
    return platform.processor() or platform.machine()


def hardware_id(cpu: str) -> str:
    """A short filesystem-safe slug identifying this machine.

    Results are committed per machine, so this becomes a directory name and
    appears in published tables. It needs to be stable across runs on the same
    box and legible to someone reading the docs — hence CPU model plus OS, not a
    hash. The `(R)`/`(TM)` strip is load bearing: without it the same machine
    lands in a second directory and the docs render two "latest" results.
    """
    tokens = re.sub(r"\((R|TM|r|tm)\)|CPU|Processor|@.*", " ", cpu)
    tokens = re.sub(r"[^A-Za-z0-9]+", "-", tokens).strip("-").lower()
    tokens = re.sub(r"^(intel|amd|apple)-", "", tokens)
    return f"{tokens or 'unknown'}-{platform.system().lower()}"


def hardware() -> dict[str, Any]:
    """Identify the machine, so results group by it."""
    cpu = cpu_model()
    ram = None
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    return {
        "id": hardware_id(cpu),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_gb": ram,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def git_sha() -> str:
    """The commit measured, with `-dirty` when the tree has uncommitted changes."""
    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"], "")
    return f"{sha}-dirty" if dirty else sha


def measure_scenario(scenario: Scenario, *, with_solver: bool, with_reference: bool) -> list[dict]:
    """Run every implementation on one scenario."""
    cases: list[dict[str, Any]] = []
    pool, spec = scenario.pool, scenario.spec

    # The yardstick both are described against. One lineup, no cuts, no time
    # pressure — this is arithmetic on the inputs, not a prediction.
    optimum = None
    best = solve_milp_ortools(pool, spec, locks=list(scenario.locks) or None, time_limit_s=120)
    if best is not None:
        optimum = float(pool.projection_of(np.array([best]), spec)[0])

    def record(impl: str, seconds: float, lineups: np.ndarray) -> None:
        cases.append(
            {
                "name": scenario.name,
                "impl": impl,
                "detail": scenario.detail,
                "median": seconds,
                "params": {
                    "requested": scenario.entries,
                    "produced": int(len(lineups)),
                    "pool": int(len(pool)),
                    "roster_size": spec.roster_size,
                },
                "quality": {
                    **({"optimum": round(optimum, 2)} if optimum else {}),
                    **describe_lineups(pool, spec, lineups, optimum),
                },
            }
        )

    print(f"  {scenario.name:14s} kernel ...", end="", flush=True)
    seconds, lineups = measure(build_lineups, pool, spec, **scenario.build_kwargs)
    record("mlb_dfs_solver_rust", seconds, lineups)
    print(f" {seconds * 1e3:9.2f} ms  ({len(lineups)}/{scenario.entries})")

    if with_reference and scenario.entries <= REFERENCE_MAX_ENTRIES:
        print(f"  {scenario.name:14s} python ...", end="", flush=True)
        seconds, lineups = measure(build_lineups_reference, pool, spec, **scenario.build_kwargs)
        record("reference_python", seconds, lineups)
        print(f" {seconds * 1e3:9.2f} ms  ({len(lineups)}/{scenario.entries})")

    if with_solver and scenario.solver:
        print(f"  {scenario.name:14s} cp-sat ...", end="", flush=True)
        seconds, found = measure(
            solve_portfolio_ortools,
            pool,
            spec,
            time_limit_s=SOLVER_LIMIT_S,
            **scenario.solver_kwargs,
        )
        lineups = np.array(found)
        if len(lineups):
            record("milp_ortools_cpsat", seconds, lineups)
            if scenario.solver_budget_s:
                # Recorded so the report can say "N lineups is what the budget
                # bought" instead of presenting a truncated run as a completed one.
                cases[-1]["params"]["budget_s"] = scenario.solver_budget_s
        print(f" {seconds:9.2f} s   ({len(lineups)}/{scenario.entries})")

    return cases


def measure_selection(*, with_reference: bool, smoke: bool = False) -> list[dict[str, Any]]:
    """Time the selection stage: the lazy-greedy kernel against the naive greedy.

    The setup — build, simulate, score — is shared workload, not the thing
    measured; the workflow record times the pipeline end to end. The line is a
    quantile of the candidates' own scores, which the docs rightly call circular
    as a *quality* measure — here it only shapes the workload, and no quality
    claim is made from it.
    """
    n_candidates, n_outcomes, n_select = (200, 100, 20) if smoke else (5_000, 2_000, 150)
    scenario = scenario_by_name("rules-only")
    pool, spec = scenario.pool, scenario.spec
    candidates = build_lineups(pool, spec, num_lineups=n_candidates, seed=11, attempts_per_lineup=5)
    universe = simulate(pool, n_outcomes)
    scores = score_lineups(pool, spec, candidates, universe)
    line = np.quantile(scores, 0.99, axis=0).astype(np.float32)
    detail = f"select {n_select} of {len(candidates):,} candidates, {n_outcomes:,} outcomes, gpp"

    cases: list[dict[str, Any]] = []

    def record(impl: str, seconds: float, chosen: Any) -> None:
        chosen = np.asarray(chosen, dtype=np.int64)
        cases.append(
            {
                "name": "selection",
                "impl": impl,
                "detail": detail,
                "median": seconds,
                "params": {
                    "requested": n_select,
                    "produced": int(len(chosen)),
                    "pool": int(len(pool)),
                    "roster_size": spec.roster_size,
                },
                "quality": describe_lineups(pool, spec, candidates[chosen]),
            }
        )

    print(f"  {'selection':14s} kernel ...", end="", flush=True)
    seconds, chosen = measure(select_portfolio, scores, mode="gpp", line=line, n_select=n_select)
    record("mlb_dfs_solver_rust", seconds, chosen)
    print(f" {seconds * 1e3:9.2f} ms  ({len(chosen)}/{n_select})")

    if with_reference:
        print(f"  {'selection':14s} python ...", end="", flush=True)
        seconds, chosen = measure(
            select_portfolio_reference, scores, mode="gpp", line=line, n_select=n_select
        )
        record("selection_reference_python", seconds, chosen)
        print(f" {seconds * 1e3:9.2f} ms  ({len(chosen)}/{n_select})")

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--scenario",
        action="append",
        choices=SCENARIO_NAMES,
        help="run only these (repeatable). Default: all.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="prove the harness executes: one scenario, tiny sizes, nothing written.",
    )
    parser.add_argument("--no-solver", action="store_true", help="skip the MILP baseline.")
    parser.add_argument("--no-reference", action="store_true", help="skip the Python oracle.")
    parser.add_argument("--out", type=Path, default=None, help="override the result path.")
    args = parser.parse_args()

    if args.smoke:
        # Deliberately tiny and deliberately not written anywhere. This exists so
        # CI can prove the code still runs without CI ever producing a number.
        scenario = scenario_by_name("stack")
        small = Scenario(
            name="smoke",
            detail="smoke",
            pool=scenario.pool,
            spec=scenario.spec,
            entries=5,
            attempts=10,
        )
        print("smoke:")
        measure_scenario(small, with_solver=not args.no_solver, with_reference=True)
        measure_selection(with_reference=True, smoke=True)
        run_workflow(smoke=True)
        print("ok")
        return 0

    wanted = set(args.scenario or SCENARIO_NAMES)
    print(f"building slates for {len(wanted)} scenario(s) ...")
    scenarios = [s for s in build_scenarios() if s.name in wanted]

    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scenario in scenarios:
        cases += measure_scenario(
            scenario,
            with_solver=not args.no_solver,
            with_reference=not args.no_reference,
        )

    # The selection stage and the whole-pipeline record belong to a full run;
    # a filtered run is somebody re-measuring one scenario.
    workflow_record = None
    if args.scenario is None:
        cases += measure_selection(with_reference=not args.no_reference)
        print("workflow:")
        workflow_record = run_workflow()
    print(f"\ntotal {time.perf_counter() - started:.1f}s")

    hw = hardware()
    report = {
        "schema": SCHEMA,
        "package": "mlb_dfs_solver",
        "version": __version__,
        "git_sha": git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": hw,
        "cases": cases,
        **({"workflow": workflow_record} if workflow_record else {}),
    }

    out = args.out or (RESULTS / hw["id"] / f"{report['timestamp'][:10]}-{report['git_sha']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # `--out` may point anywhere, so relativize only when it is inside the repo.
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"wrote {shown}")
    if report["git_sha"].endswith("-dirty"):
        print("warning: tree is dirty, so this result is not reproducible from a commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
