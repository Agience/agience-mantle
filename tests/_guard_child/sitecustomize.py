"""Loaded by every child Python a test starts (the root `conftest.py` puts this directory first
on `PYTHONPATH`): the same guard over the install-root store, in the child."""
try:
    import mantle_test_guard

    mantle_test_guard.install()
except Exception:  # noqa: BLE001 — a child with no guard is reported by the parent's test
    pass
