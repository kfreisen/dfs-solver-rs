# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/kfreisen/mlb-dfs-solver/security/advisories/new)
rather than opening a public issue.

Include the affected version, what an attacker can do with the issue, and a
reproduction if you have one. You should get an acknowledgement within a week.

This is a small library maintained by one person in their own time — please size
your expectations accordingly.

## Supported versions

Only the latest released version is supported. There are no long-term support
branches.

## Scope

In scope: anything under `src/` (or `python/`), and the release workflow.

Out of scope: the benchmark baselines under `benchmarks/baselines/`, which exist to
be compared against and are not part of any published distribution.
