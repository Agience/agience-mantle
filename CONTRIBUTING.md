# Contributing to Agience Mantle

## Build and test

```bash
pip install -e '.[all]'
KEYS_DIR=<tmp-dir> MANTLE_LATTICE_PATH=<tmp-file> python -m pytest tests src/mantle -q
```

**Read the security invariants in [README.md](README.md#security-invariants).** A change that
weakens one will not be merged, and a change touching routing, key derivation, the light cone,
grants or cell encryption must add or extend an invariant test.

**Call the attenuation operator; do not write a second one.** Mask intersection is single-sourced in
[`src/mantle/attenuation.py`](src/mantle/attenuation.py). Do not intersect masks inline, and do not
add a helper named `intersect`, `meet`, `narrow` or `attenuate` anywhere else —
`tests/test_attenuation_is_single_sourced.py` sweeps for both shapes a re-implementation takes.

## Contributing

Fork, branch from `main`, sign off every commit (`git commit -s`) to certify the
[DCO](https://developercertificate.org/), open a PR. Commit format: `fix:` · `feat(scope):` ·
`docs:` · `test:` · `chore:`.

By contributing you agree your contribution is Apache-2.0 (per section 5), including its
section 3 patent grant.

Licensed under Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
