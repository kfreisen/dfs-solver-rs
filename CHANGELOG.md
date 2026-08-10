# Changelog

All notable changes to `mlb_dfs_solver` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tested and advertised support through Python 3.14. The `abi3-py311` wheel already
  ran there; CI now proves it at the ceiling as well as the floor, and the release
  job loads the built wheel on both.

### Changed

- Replaced the `Makefile` with a `Taskfile.yml`. Same recipes; run `task` to list
  them. No install needed — `uvx --from go-task-bin task <name>`.
- Bumped `pyo3` and `rust-numpy` 0.24 → 0.29, which moves the Rust MSRV to 1.83.
  No effect on the Python API.

## [0.0.1.dev0]

### Added

- Name reservation and release-pipeline validation. No functionality yet.
