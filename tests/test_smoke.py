"""Scaffold-level checks: the package imports, the extension loads, and dispatch works.

These stay after mlb_dfs_solver is implemented. Because the compiled kernel is a hard
requirement rather than an optional accelerator, an import failure here is the fastest
signal that a wheel, a maturin build, or an abi3 tag is wrong — and it runs in
milliseconds.
"""

import mlb_dfs_solver


def test_version_is_exposed():
    assert isinstance(mlb_dfs_solver.__version__, str)
    assert mlb_dfs_solver.__version__


def test_native_extension_is_loaded():
    # Reaching this at all means mlb_dfs_solver._native imported, which is the thing most
    # likely to break across platforms and interpreter versions.
    assert mlb_dfs_solver.native_version()


def test_active_isa_reports_a_known_dispatch_path():
    # Wheels are built without target-cpu=native, so the SIMD path is chosen at
    # runtime. Anything outside this set means dispatch is broken, not merely slow.
    assert mlb_dfs_solver.active_isa() in {"avx2", "scalar"}
