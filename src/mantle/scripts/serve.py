#!/usr/bin/env python3
"""Run the Mantle service. The `mantle-serve` console script.

    mantle-serve                                  # 0.0.0.0:8081
    mantle-serve --port 9000 --reload             # development
    python -m mantle.scripts.serve --host 127.0.0.1

An install should yield a command. Without this, `pip install agience-mantle[service]` produces a
package whose consumer must already know the ASGI path (`mantle.main:app`), the port the rest of
the platform expects it on, and where the log config lives — three facts that would otherwise live
in a deployment recipe rather than in the wheel. Those defaults belong to the package, so they
travel with it.

This is a thin wrapper and stays one: every flag below maps to a uvicorn parameter of the same
name, and anything uvicorn offers that is not here is still reachable by invoking uvicorn
directly. `uvicorn mantle.main:app --host 0.0.0.0 --port 8081` remains exactly equivalent.

`KEYS_DIR` must hold a keyset before this will boot — `main.py`'s lifespan loads key material
off disk and refuses to invent it. `mantle-init-keys` writes a throwaway one for development.
"""
from __future__ import annotations

import argparse
import os
import sys

#: The port the rest of the platform addresses this service on (every `MANTLE_URI` default, and
#: what a reverse proxy in front of a node is pointed at). A different default here would make the
#: package's own default disagree with where the rest of the platform looks for Mantle.
_DEFAULT_PORT = 8081

#: How long uvicorn waits for in-flight requests before dropping them on shutdown, so a supervised
#: restart and a local Ctrl-C behave the same way.
_GRACEFUL_SHUTDOWN_S = 10


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mantle-serve",
        description="Run the Mantle service (uvicorn over mantle.main:app).",
        epilog="KEYS_DIR and MANTLE_LATTICE_PATH configure the keyset and the store; "
               "see `mantle-init-keys` for a development keyset.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: %(default)s).")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT,
                        help="Bind port (default: %(default)s).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes. Single-process by default.")
    parser.add_argument("--reload", action="store_true",
                        help="Reload on source change. Development only.")
    parser.add_argument("--log-level", default=None,
                        help="uvicorn log level; otherwise the app's own configuration applies.")
    args = parser.parse_args(argv)

    # Imported here rather than at module scope so `--help` works, and reports the real defaults,
    # in an environment where the `[service]` extra is not installed.
    #
    # The base install puts this command on the path without being able to run it:
    # `[project.scripts]` declares `mantle-serve` unconditionally, while uvicorn arrives only with
    # `[service]`, so `pip install agience-mantle` followed by `mantle-serve` is a reachable first
    # move. A bare `ModuleNotFoundError: No module named 'uvicorn'` names the missing module but
    # not the remedy, and leaves the reader unable to tell whether the install is broken, the wheel
    # is incomplete, or a second command exists. The handler below names the remedy, and exits like
    # the other refusal in this file rather than unwinding a traceback out of a console script.
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        if exc.name != "uvicorn":
            raise
        # ASCII in the printed text on purpose: this lands on a stranger's terminal, and a Windows
        # console on a legacy code page renders an em dash as a replacement character — which
        # makes the one message they have to read look corrupted. `dev_init_keys` prints its own
        # refusals the same way.
        print("mantle-serve: uvicorn is not installed - it ships with the `service` extra, "
              "not with the base install.\n"
              "    pip install 'agience-mantle[service]'\n"
              "The base install is the embeddable store (artifacts, grants, lexical search); "
              "running it as an HTTP service is what `[service]` adds.", file=sys.stderr)
        return 2

    options = {
        "host": args.host,
        "port": args.port,
        "timeout_graceful_shutdown": _GRACEFUL_SHUTDOWN_S,
    }
    if args.reload:
        options["reload"] = True
    if args.workers is not None:
        options["workers"] = args.workers
        # The app's boot check reads the worker count from the environment, because each worker is
        # a separate process that never sees these arguments. Publishing it here is what lets that
        # check see the real shape and refuse a multi-worker start with no signalling back-plane —
        # an in-process bus cannot reach a subscriber attached to a sibling worker, so the change
        # feed would lose most of its deliveries while every worker reported healthy.
        os.environ["MANTLE_WORKERS"] = str(args.workers)

        # ...and the SAME check runs HERE, in the parent, before uvicorn is asked to supervise
        # anything. The lifespan's copy alone is not a refusal: uvicorn's multiprocess supervisor
        # treats a worker that exits as a worker to replace, so a `RuntimeError` raised in every
        # child produces an endless spawn/crash loop under a parent that keeps reporting "Uvicorn
        # running on http://…" and never exits. MEASURED: `mantle-serve --workers 2` with no
        # back-plane respawned 18 crashing workers in 100 seconds and had to be killed. A guard
        # whose whole purpose is to refuse a configuration must refuse it in the process that owns
        # the decision — the parent — where exiting is possible.
        #
        # `backplane_from_env()` only constructs; no broker connection is opened here, so this
        # costs nothing on the supported path. It also surfaces a typo'd `MANTLE_BACKPLANE_KIND`
        # (a `ValueError` from `make_backplane`) before N children each discover it separately.
        from mantle.events import event_backplane

        try:
            event_backplane.require_backplane_for_workers(
                args.workers, event_backplane.backplane_from_env())
        except (RuntimeError, ValueError) as exc:
            print("mantle-serve: %s" % exc, file=sys.stderr)
            return 2
    if args.log_level:
        options["log_level"] = args.log_level

    # The import STRING, not the app object: uvicorn needs a target it can re-import in a child
    # process, and passing the object silently disables both --reload and --workers.
    uvicorn.run("mantle.main:app", **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
