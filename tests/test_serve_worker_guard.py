"""`mantle-serve --workers N` must REFUSE, not crash-loop.

The multi-worker/no-back-plane guard lives in `main.py`'s lifespan, and that copy is necessary
(a worker started by a container command or a bare `uvicorn --workers 2` never runs
`scripts/serve.py` at all). But it is not sufficient, and the gap is not theoretical:

    $ mantle-serve --host 127.0.0.1 --port 8313 --workers 2      # no back-plane configured
    INFO:     Uvicorn running on http://127.0.0.1:8313 (Press CTRL+C to quit)
    INFO:     Started parent process [30236]
    ... RuntimeError: MANTLE_WORKERS=2 with no signalling back-plane configured ...
    ERROR - uvicorn.error - Application startup failed. Exiting.
    ... x18 in 100 seconds, until the run was killed from outside ...

uvicorn's multiprocess supervisor replaces a worker that exits, so a lifespan that raises in every
child is an endless spawn/crash loop under a parent that keeps announcing itself as running. The
refusal has to happen in the process that can act on it — the parent — so these tests pin the
check to `main()` itself rather than to the child's lifespan.
"""

import pytest

from mantle.scripts import serve


@pytest.fixture
def no_uvicorn(monkeypatch):
    """Fail loudly if `main()` reaches `uvicorn.run` — reaching it IS the bug."""
    import uvicorn

    calls = []

    def _run(target, **options):
        calls.append((target, options))

    monkeypatch.setattr(uvicorn, "run", _run)
    return calls


@pytest.fixture(autouse=True)
def clean_backplane_env(monkeypatch):
    for name in ("MANTLE_WORKERS", "MANTLE_BACKPLANE_KIND", "MANTLE_BACKPLANE_URI"):
        monkeypatch.delenv(name, raising=False)


def test_multi_worker_without_backplane_refuses_before_uvicorn(no_uvicorn, capsys):
    rc = serve.main(["--workers", "2", "--port", "0"])

    assert rc != 0, "a refused configuration must not exit 0"
    assert no_uvicorn == [], "uvicorn must never be started for a configuration already refused"
    err = capsys.readouterr().err
    # The message names the setting an operator has to change, per the guard's own contract.
    assert "MANTLE_WORKERS=2" in err
    assert "MANTLE_BACKPLANE_KIND" in err


def test_multi_worker_with_backplane_starts(no_uvicorn, monkeypatch):
    monkeypatch.setenv("MANTLE_BACKPLANE_KIND", "redis")
    monkeypatch.setenv("MANTLE_BACKPLANE_URI", "redis://localhost:6379/0")

    rc = serve.main(["--workers", "2", "--port", "0"])

    assert rc == 0
    assert len(no_uvicorn) == 1, "the supported multi-worker shape must reach uvicorn"
    target, options = no_uvicorn[0]
    assert target == "mantle.main:app"
    assert options["workers"] == 2


def test_single_worker_needs_no_backplane(no_uvicorn):
    rc = serve.main(["--workers", "1", "--port", "0"])

    assert rc == 0
    assert len(no_uvicorn) == 1
    assert no_uvicorn[0][1]["workers"] == 1


def test_default_invocation_is_unaffected(no_uvicorn):
    """No `--workers` at all: the standalone configuration, and the common one."""
    rc = serve.main(["--port", "0"])

    assert rc == 0
    assert len(no_uvicorn) == 1
    assert "workers" not in no_uvicorn[0][1]


def test_workers_is_published_to_the_environment_for_the_children(no_uvicorn):
    """The children read the count from the environment; the parent still has to publish it."""
    import os

    serve.main(["--workers", "1", "--port", "0"])
    assert os.environ["MANTLE_WORKERS"] == "1"


def test_unknown_backplane_kind_is_refused_in_the_parent(no_uvicorn, monkeypatch, capsys):
    """A typo must not be discovered separately by each of N children."""
    monkeypatch.setenv("MANTLE_BACKPLANE_KIND", "redsi")
    monkeypatch.setenv("MANTLE_BACKPLANE_URI", "redis://localhost:6379/0")

    rc = serve.main(["--workers", "2", "--port", "0"])

    assert rc != 0
    assert no_uvicorn == []
    assert "redsi" in capsys.readouterr().err
