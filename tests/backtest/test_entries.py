"""The entries column store and the run manifest."""

from __future__ import annotations

import numpy as np
import pytest
from dfs_solver.backtest.entries import COLUMNS, Entries, RunManifest, merge


def row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "strategy": "a",
        "slate_key": "2025-06-01",
        "contest_id": 1,
        "lineup_idx": 0,
        "entry_fee": 5.0,
        "field_size": 100,
        "max_entries_per_user": 20,
        "sim_p50": 90.0,
        "sim_p99": 140.0,
        "ev_pred": float("nan"),
        "lineup_score": 95.0,
        "lineup_rank": 7,
        "lineup_payout": 12.0,
        "lineup_pit": 0.6,
    }
    base.update(overrides)
    return base


# --- RunManifest ----------------------------------------------------------------


def test_manifest_compares_by_value_and_freezes_extra() -> None:
    a = RunManifest(seed=1, n_sims=100, label="x", extra={"model": "v3"})
    b = RunManifest(seed=1, n_sims=100, label="x", extra={"model": "v3"})
    assert a == b
    assert hash(a) == hash(b)
    assert a != RunManifest(seed=2, n_sims=100, label="x", extra={"model": "v3"})
    assert a != RunManifest(seed=1, n_sims=100, label="x", extra={"model": "v4"})
    assert a.__eq__("not a manifest") is NotImplemented
    with pytest.raises(TypeError):
        a.extra["model"] = "v5"  # type: ignore[index]


# --- construction -----------------------------------------------------------------


def test_from_dicts_round_trips_through_to_dicts() -> None:
    rows = [row(), row(lineup_idx=1, contest_id="c2", lineup_payout=0.0)]
    entries = Entries.from_dicts(rows)
    assert len(entries) == 2
    assert entries.contest_id.dtype == object
    assert entries.lineup_idx.dtype == np.int64
    assert entries.lineup_payout.dtype == np.float64
    back = entries.to_dicts()
    assert back[1]["contest_id"] == "c2"
    assert back[0]["lineup_rank"] == 7
    assert isinstance(back[0]["lineup_rank"], int)
    assert np.isnan(back[0]["ev_pred"])


def test_from_dicts_rejects_a_missing_column() -> None:
    incomplete = row()
    del incomplete["lineup_pit"]
    with pytest.raises(KeyError, match="row 0 is missing column\\(s\\): \\['lineup_pit'\\]"):
        Entries.from_dicts([incomplete])


def test_from_columns_rejects_missing_and_unknown() -> None:
    columns = {name: [] for name in COLUMNS}
    columns["bogus"] = []
    del columns["sim_p50"]
    with pytest.raises(KeyError, match="missing \\['sim_p50'\\] and unknown \\['bogus'\\]"):
        Entries.from_columns(columns)


def test_columns_must_be_parallel_and_one_dimensional() -> None:
    columns = {name: [0] for name in COLUMNS}
    columns["strategy"] = ["a", "b"]
    with pytest.raises(ValueError, match="every column must be parallel"):
        Entries.from_columns(columns)
    two_d = {name: np.zeros(2) for name in COLUMNS}
    two_d["strategy"] = np.empty(2, dtype=object)
    two_d["sim_p50"] = np.zeros((2, 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        Entries(**two_d)


def test_empty_carries_its_manifest() -> None:
    manifest = RunManifest(seed=0, n_sims=1)
    entries = Entries.empty(manifest)
    assert len(entries) == 0
    assert entries.manifest == manifest
    assert entries.to_dicts() == []


# --- filter / where ------------------------------------------------------------------


def test_filter_and_where() -> None:
    entries = Entries.from_dicts(
        [row(), row(strategy="b", contest_id=2), row(strategy="a", contest_id=2)],
        manifest=RunManifest(seed=0, n_sims=1),
    )
    kept = entries.filter(entries.lineup_payout > 0)
    assert len(kept) == 3
    assert kept.manifest == entries.manifest
    assert len(entries.where(strategy="a")) == 2
    assert len(entries.where(strategy="a", contest_id=2)) == 1
    with pytest.raises(ValueError, match="mask has shape"):
        entries.filter(np.array([True]))
    with pytest.raises(KeyError, match="no column 'nope'"):
        entries.where(nope=1)


# --- concat / merge --------------------------------------------------------------------


def test_concat_keeps_a_shared_manifest_and_drops_a_disputed_one() -> None:
    m = RunManifest(seed=0, n_sims=1)
    a = Entries.from_dicts([row()], manifest=m)
    b = Entries.from_dicts([row(contest_id=2)], manifest=m)
    c = Entries.from_dicts([row(contest_id=3)], manifest=RunManifest(seed=9, n_sims=1))
    assert Entries.concat([a, b]).manifest == m
    assert len(Entries.concat([a, b, c])) == 3
    assert Entries.concat([a, b, c]).manifest is None
    assert len(Entries.concat([])) == 0


def test_merge_refuses_to_mix_runs_unless_told() -> None:
    a = Entries.from_dicts([row()], manifest=RunManifest(seed=0, n_sims=1))
    b = Entries.from_dicts([row()], manifest=RunManifest(seed=1, n_sims=1))
    with pytest.raises(ValueError, match="refusing to merge entries from 2 different runs"):
        merge([a, b])
    assert len(merge([a, a])) == 2
    pooled = merge([a, b], allow_mismatched_manifests=True)
    assert len(pooled) == 2
    assert pooled.manifest is None
