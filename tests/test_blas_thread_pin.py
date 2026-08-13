"""The BLAS thread pin travels with the package, and it is measured by its effect.

The failure modes this file exists to catch, stated before the assertions, because a check
whose failure mode is unstated cannot be shown to have one:

  1. The pin is deleted from `mantle/__init__.py`. `test_pin_is_set_by_importing_the_package`
     fails: it imports mantle in a subprocess with `OPENBLAS_NUM_THREADS` removed from the
     environment and finds the variable still unset.

  2. The pin goes inert without going missing — it is moved below an import that pulls numpy,
     or something mantle imports earlier starts importing numpy. The variable would still read
     "1" and a string check would still pass, but OpenBLAS would already have sized its pool.
     `test_pin_actually_sizes_the_openblas_pool` fails, because it reads the pool back through
     `threadpoolctl` instead of reading the string the pin wrote.

  3. A new BLAS caller appears outside the covered path. The parametrised cases below name
     the modules AST-measured to call `numpy.linalg.*`; each is imported directly, so a module
     that stops going through `mantle/__init__.py` is caught rather than assumed covered.

  4. The pin is "strengthened" into an unconditional set, discarding a value an operator chose
     deliberately. `test_operator_value_is_not_overridden` fails.

`test_late_pin_is_inert_negative_control` is the negative control for (2): it sets the variable
after `import numpy` and asserts the pool did not shrink. Without it, (2) could be green for a
reason having nothing to do with the pin.

Why this matters here specifically: mantle's semantic arm runs SVD on any machine that installs
it, and two threads inside `numpy.linalg.eigh` fault the reference box's OpenBLAS 3/3 (exit 139)
— and can hang instead of faulting, so a single green run proves very little.

Everything runs in a subprocess: this process has already imported numpy, so its own pool was
sized long before any assertion here could run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

PACKAGE = "mantle"
PIN = "OPENBLAS_NUM_THREADS"

# AST-measured: modules importing numpy at module level and calling `numpy.linalg.*`.
# Only the ones that import standalone without a live store are listed; the pin is a package-scope
# line, so these are a representative probe of the path, not a roster to keep in sync by hand.
BLAS_MODULES = [
    "mantle.shard.cache",              # numpy.dot
    "mantle.search.beacon.engine",     # numpy.linalg.svd
    "mantle.search.beacon.instrument",  # numpy.linalg.svd / eigvalsh / qr / pinv
    "mantle.search.anchors.anchorset",  # numpy.linalg.norm
]


def _run(body: str, env_pin: str | None) -> dict:
    """Run `body` in a clean interpreter and return its JSON verdict.

    `env_pin=None` removes the variable from the child's environment — the unset case the pin
    exists to cover. The parent's `sys.path` is handed over in-band rather than through
    PYTHONPATH, so this works from a bare `pytest` with nothing exported.
    """
    prelude = f"import sys, os, json\nsys.path[:0] = {json.dumps(sys.path)}\n"
    env = dict(os.environ)
    env.pop(PIN, None)
    if env_pin is not None:
        env[PIN] = env_pin
    proc = subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_POOL_READER = (
    "import threadpoolctl\n"
    "pools = [d['num_threads'] for d in threadpoolctl.threadpool_info()\n"
    "         if d.get('internal_api') == 'openblas']\n"
)


def test_threadpoolctl_is_available() -> None:
    """The measurement tool is a declared dev dependency, not a lucky import.

    Fails if it is missing, instead of letting the pool assertions turn into skips — which would
    leave the suite green while nothing had been measured at all."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:  # pragma: no cover - the point is that this is loud
        pytest.fail(
            "threadpoolctl is not installed, so the BLAS pin can only be checked as a string and "
            "not as an effect. Install it (`pip install -e .[dev]`). Deliberately a failure, "
            "not a skip."
        )


def test_pin_is_set_by_importing_the_package() -> None:
    """Importing `mantle` with the variable unset must leave it pinned to "1".

    Fails if the `os.environ.setdefault` block is removed from `mantle/__init__.py`."""
    v = _run(f"import {PACKAGE}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", None)
    assert v["val"] == "1", (
        f"{PACKAGE} did not pin {PIN} on import (got {v['val']!r}). The pin must be set by the "
        f"package, not by whoever remembers to export it."
    )


@pytest.mark.parametrize("mod", BLAS_MODULES)
def test_pin_covers_each_measured_blas_module(mod: str) -> None:
    """Importing a measured LAPACK caller directly must also be covered.

    Python initialises parent packages before submodules, so `import mantle.search.beacon.engine`
    runs `mantle/__init__.py` first. Fails if the pin moves out of the package root into some
    module these do not go through."""
    v = _run(f"import {mod}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", None)
    assert v["val"] == "1", f"importing {mod} left {PIN}={v['val']!r}"


@pytest.mark.parametrize("mod", BLAS_MODULES)
def test_pin_actually_sizes_the_openblas_pool(mod: str) -> None:
    """The pin must take effect: after importing the module, OpenBLAS reports one thread.

    This is the assertion that survives failure mode (2) — a pin that is present but late still
    writes "1" into the environment, and only the pool size shows that it did nothing."""
    v = _run(f"import {mod}\n" + _POOL_READER + "print(json.dumps({'pools': pools}))", None)
    assert v["pools"], "no OpenBLAS pool reported; threadpoolctl saw no BLAS backend to measure"
    assert all(p == 1 for p in v["pools"]), (
        f"OpenBLAS pool is {v['pools']} after importing {mod}. The variable may well be set — a "
        f"late set is inert, because the pool is sized when the library loads."
    )


def test_late_pin_is_inert_negative_control() -> None:
    """Negative control — proves the measurement above can actually fail.

    Import numpy first, then set the variable, then force a LAPACK call. The pool must not be 1,
    because OpenBLAS sized it when the library loaded. If this ever reports 1, then
    `test_pin_actually_sizes_the_openblas_pool` is passing for free and proves nothing.

    A machine that reports 1 thread unpinned (a single-core box, or an OpenBLAS built without
    threading) cannot discriminate, and this fails loudly saying so rather than skipping — the
    point of a control is to make "we measured nothing" visible.
    """
    v = _run(
        "import numpy\n"
        f"os.environ[{PIN!r}] = '1'\n"
        "numpy.linalg.eigh(numpy.eye(64))\n" + _POOL_READER + "print(json.dumps({'pools': pools}))",
        None,
    )
    assert v["pools"], "no OpenBLAS pool reported; the control cannot run"
    assert all(p > 1 for p in v["pools"]), (
        f"unpinned OpenBLAS pool is {v['pools']}, not >1. Either this machine has one usable core "
        f"or OpenBLAS is single-threaded here; either way the pin cannot be distinguished from a "
        f"no-op on this hardware, so test_pin_actually_sizes_the_openblas_pool proves nothing. "
        f"cpu_count={os.cpu_count()}"
    )


def test_operator_value_is_not_overridden() -> None:
    """An operator who exported a value keeps it — the pin is a default, never a policy.

    This makes the guard weaker by design: a deployment that exports `OPENBLAS_NUM_THREADS=8`
    reinstates exactly the fault the pin exists to avoid, and nothing here stops it. What the
    package owes is the unset case."""
    v = _run(f"import {PACKAGE}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", "3")
    assert v["val"] == "3", (
        f"{PACKAGE} overwrote an operator-set {PIN} (got {v['val']!r}, expected '3'). Use "
        f"os.environ.setdefault, not assignment."
    )
