"""The per-slate loop, end to end through the library, and rescoring.

The end-to-end test uses the conftest MLB slate: build candidates, score them
over a synthetic outcome matrix, select a prefix-consistent portfolio, and enter
it into two contests with synthetic fields. Every number the loop writes is then
checked against the same arithmetic done directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from dfs_solver import build_lineups, field_line, score_lineups, select_portfolio
from dfs_solver.backtest.contest import Contest, PayoutTier, synthetic_gpp
from dfs_solver.backtest.entries import Entries, RunManifest
from dfs_solver.backtest.field import FieldCdf, expected_payouts
from dfs_solver.backtest.loop import SlateContest, rescore, run_slate
from dfs_solver.backtest.ranking import FieldScores, realized_ranks_and_payouts
from dfs_solver.pool import PlayerPool
from dfs_solver.spec import RosterSpec


@pytest.fixture
def slate(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Candidates, their simulated and realized scores, and a selection order."""
    n_sims = 300
    teams = mlb_pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, n_sims + 1)) * 0.5
    noise = rng.standard_normal((len(mlb_pool), n_sims + 1))
    universe = mlb_pool.projections[:, None] + mlb_pool.stddevs[:, None] * (noise + shock[teams])
    candidates = build_lineups(mlb_pool, mlb_spec, num_lineups=400, seed=11)
    scored = score_lineups(mlb_pool, mlb_spec, candidates, universe[:, :n_sims])
    # The last column of the universe is "what happened".
    realized = np.asarray(
        score_lineups(mlb_pool, mlb_spec, candidates, universe[:, n_sims:])[:, 0], dtype=np.float64
    )
    order = select_portfolio(scored, mode="gpp", line=field_line(scored, 0.9), n_select=150)
    return {"scored": scored, "realized": realized, "order": order}


@pytest.fixture
def contests(slate: dict[str, np.ndarray], rng: np.random.Generator) -> list[SlateContest]:
    centre = float(slate["scored"].mean())
    spread = float(slate["scored"].std())
    small = synthetic_gpp(20.0, 500, max_entries_per_user=20, contest_id="small")
    large = synthetic_gpp(3.0, 5_000, max_entries_per_user=150, contest_id="large")
    cdf = FieldCdf(np.sort(rng.normal(centre, spread, 20_000)), n_contests=10, window=30)
    return [
        SlateContest(small, FieldScores.of(rng.normal(centre, spread, 480))),
        SlateContest(large, rng.normal(centre, spread, 4_850), cdf=cdf),
    ]


def test_run_slate_end_to_end(slate: dict[str, np.ndarray], contests: list[SlateContest]) -> None:
    manifest = RunManifest(seed=11, n_sims=300, label="e2e")
    entries = run_slate(
        strategy="gpp",
        slate_key="day-1",
        sim_scores=slate["scored"],
        order=slate["order"],
        realized_scores=slate["realized"],
        contests=contests,
        manifest=manifest,
    )
    order = slate["order"]
    n_small = min(20, len(order))
    n_large = min(150, len(order))
    assert len(entries) == n_small + n_large
    assert entries.manifest == manifest
    assert set(entries.strategy.tolist()) == {"gpp"}
    assert set(entries.slate_key.tolist()) == {"day-1"}

    small = entries.where(contest_id="small")
    large = entries.where(contest_id="large")
    assert len(small) == n_small
    assert len(large) == n_large

    # Prefix consistency: the same lineup index is the same lineup in both.
    assert small.lineup_idx.tolist() == list(range(n_small))
    np.testing.assert_array_equal(small.lineup_score, large.lineup_score[:n_small])
    np.testing.assert_array_equal(small.lineup_score, slate["realized"][order[:n_small]])
    np.testing.assert_array_equal(small.sim_p50, large.sim_p50[:n_small])

    # Every rank and payout is what ranking directly says.
    for part, slate_contest in ((small, contests[0]), (large, contests[1])):
        ranks, payouts = realized_ranks_and_payouts(
            part.lineup_score, slate_contest.field, slate_contest.contest
        )
        np.testing.assert_array_equal(part.lineup_rank, ranks)
        np.testing.assert_allclose(part.lineup_payout, payouts)
        assert set(part.entry_fee.tolist()) == {slate_contest.contest.entry_fee}
        assert set(part.field_size.tolist()) == {slate_contest.contest.field_size}
        assert set(part.max_entries_per_user.tolist()) == {
            slate_contest.contest.max_entries_per_user
        }

    # The simulation summaries and PIT are per lineup.
    sims = slate["scored"][order[:n_large]]
    np.testing.assert_allclose(large.sim_p50, np.quantile(sims, 0.5, axis=1))
    np.testing.assert_allclose(large.sim_p99, np.quantile(sims, 0.99, axis=1))
    np.testing.assert_allclose(large.lineup_pit, (sims <= large.lineup_score[:, None]).mean(axis=1))
    assert np.all((large.lineup_pit >= 0) & (large.lineup_pit <= 1))

    # Expected payout only where a field model was given.
    assert np.all(np.isnan(small.ev_pred))
    assert contests[1].cdf is not None
    np.testing.assert_allclose(
        large.ev_pred,
        expected_payouts(sims, contests[1].cdf, contests[1].contest.payout_table, 5_000),
    )


def test_run_slate_enters_nothing_when_there_is_nothing_to_enter(
    slate: dict[str, np.ndarray], contests: list[SlateContest]
) -> None:
    empty = run_slate(
        strategy="s",
        slate_key="k",
        sim_scores=slate["scored"],
        order=np.array([], dtype=np.int64),
        realized_scores=slate["realized"],
        contests=contests,
    )
    assert len(empty) == 0
    none = run_slate(
        strategy="s",
        slate_key="k",
        sim_scores=slate["scored"],
        order=slate["order"],
        realized_scores=slate["realized"],
        contests=[],
    )
    assert len(none) == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sim_scores": np.zeros(4)}, "two-dimensional"),
        ({"realized_scores": np.zeros(3)}, "realized_scores has shape"),
        ({"order": [0, 4]}, "outside the 4 scored"),
        ({"order": [0, -1]}, "outside the 4 scored"),
        ({"order": [1, 1]}, "more than once"),
    ],
)
def test_run_slate_rejects_inconsistent_inputs(
    overrides: dict[str, object], match: str, contests: list[SlateContest]
) -> None:
    kwargs: dict[str, object] = {
        "strategy": "s",
        "slate_key": "k",
        "sim_scores": np.zeros((4, 5)),
        "order": [0, 1],
        "realized_scores": np.zeros(4),
        "contests": contests,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        run_slate(**kwargs)  # type: ignore[arg-type]


def test_slate_contest_refuses_a_void_or_oversized_field() -> None:
    contest = synthetic_gpp(1.0, 10, max_entries_per_user=1)
    with pytest.raises(ValueError, match="void"):
        SlateContest(contest, np.zeros(9))
    with pytest.raises(ValueError, match="holds 11 entries but field_size is 10"):
        SlateContest(contest, np.arange(1.0, 12.0))


# --- rescore -------------------------------------------------------------------


def two_contest_entries() -> tuple[Entries, Contest, Contest]:
    a = Contest(1, 1.0, 10, 3, (PayoutTier(1, 1, 10.0), PayoutTier(2, 3, 2.0)))
    b = Contest(2, 1.0, 10, 3, (PayoutTier(1, 2, 5.0),))
    columns = {
        "strategy": ["s"] * 4,
        "slate_key": ["k"] * 4,
        "contest_id": [1, 1, 2, 2],
        "lineup_idx": [0, 1, 0, 1],
        "entry_fee": [1.0] * 4,
        "field_size": [10] * 4,
        "max_entries_per_user": [3] * 4,
        "sim_p50": [0.0] * 4,
        "sim_p99": [0.0] * 4,
        "ev_pred": [float("nan")] * 4,
        "lineup_score": [50.0, 40.0, 50.0, 40.0],
        "lineup_rank": [1, 3, 2, 10],
        "lineup_payout": [10.0, 2.0, 5.0, 0.0],
        "lineup_pit": [0.5] * 4,
    }
    return Entries.from_columns(columns, RunManifest(seed=0, n_sims=1)), a, b


def test_rescore_re_reads_ranks_through_new_tables() -> None:
    entries, a, b = two_contest_entries()
    corrected_a = Contest(1, 1.0, 10, 3, (PayoutTier(1, 1, 100.0), PayoutTier(2, 3, 1.0)))
    # Contest 2 shrinks to a five-entry field: the rank-10 row now finishes
    # past the field and is paid nothing rather than raising.
    corrected_b = Contest(2, 1.0, 5, 3, (PayoutTier(1, 2, 7.0),))
    out = rescore(entries, [corrected_a, corrected_b])
    assert out.lineup_payout.tolist() == [100.0, 1.0, 7.0, 0.0]
    assert out.lineup_rank.tolist() == entries.lineup_rank.tolist()
    assert out.manifest == entries.manifest
    assert entries.lineup_payout.tolist() == [10.0, 2.0, 5.0, 0.0]
    same = rescore(entries, [a, b])
    assert same.lineup_payout.tolist() == entries.lineup_payout.tolist()


def test_rescore_refuses_a_missing_contest_unless_told_to_drop() -> None:
    entries, a, _ = two_contest_entries()
    with pytest.raises(
        KeyError, match="2 row\\(s\\) belong to 1 contest\\(s\\) not given: \\['2'\\]"
    ):
        rescore(entries, [a])
    kept = rescore(entries, [a], drop_missing=True)
    assert kept.contest_id.tolist() == [1, 1]
    assert kept.lineup_payout.tolist() == [10.0, 2.0]
