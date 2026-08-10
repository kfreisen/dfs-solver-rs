# mlb_dfs_solver — Rust workspace

Two crates, deliberately.

**`core/`** holds every algorithm and depends on no Python. That is what makes
`cargo test` and `cargo llvm-cov` work with no interpreter, no `libpython` on the
loader path, and no coverage diluted by binding glue.

**`native/`** is the PyO3 extension: a `#[pymodule]`, argument marshalling, and
nothing else worth measuring.

The single-crate version of this could not run its own unit tests. Enabling pyo3's
`extension-module` feature tells it not to link `libpython`, which is required for
the wheel and fatal for a test binary; disabling it makes the tests link and then
fail to load `libpython` at runtime. Splitting sidesteps the choice rather than
papering over it with `LD_LIBRARY_PATH`. `extension-module` is added by maturin at
build time, via the `features` key in `pyproject.toml`.

## Working here

```bash
cargo test -p mlb_dfs_solver-core          # no Python needed
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

To build the extension into the Python environment:

```bash
cd ..            # packages/mlb_dfs_solver
uv sync --all-extras     # builds the extension as part of the install
```

## Coverage

```bash
cargo llvm-cov -p mlb_dfs_solver-core --fail-under-lines 90
```

90 rather than the 100 the Python side holds to. The gap is the parallel merge and
a few defensive branches that a unit test cannot provoke without reaching into
rayon's scheduling.

Cross-FFI coverage — instrumenting the extension and running pytest against it — is
deliberately not wired up. It works, it is fiddly, and the binding layer it would
cover is thin enough that the cost outweighs the signal.

## Two things that must not be undone

**No `.cargo/config.toml` with `-C target-cpu=native`.** That flag bakes the build
machine's instruction set into the wheel, so a wheel built on a runner with AVX-512
executes an illegal instruction on a user's older CPU. It is a tempting one-line
"free speedup" and it produces crash reports nobody can reproduce. The AVX2 kernels
use runtime `is_x86_feature_detected!` dispatch instead, and `mlb_dfs_solver.active_isa()`
reports which path a given process took.

**No `panic = "abort"`.** PyO3 converts Rust panics into Python exceptions, which
requires unwinding. Aborting would take the interpreter down instead of raising.

## The test that matters most

`core/src/simd.rs` asserts the AVX2 and scalar paths agree **bit for bit** at every
input length from 0 to 64. Floating-point reduction makes that a real constraint
rather than a formality: summing eight lanes in parallel and folding at the end
rounds differently from a left-to-right scalar sum. The scalar reference therefore
reproduces the vector accumulator layout on purpose. Writing the obvious
`iter().sum()` reference instead would force a tolerance, and a tolerance hides
exactly the lane-handling and tail-handling bugs the test exists to catch.
