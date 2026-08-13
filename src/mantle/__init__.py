"""Agience Mantle — the lattice: where the data lives.

The lattice the whole universe persists in: the encrypted-by-default artifact
store and search service (standalone lattice store: one SQLite file + a
filesystem CAS), where authorization is the encryption.

Mantle HOLDS the data; it does not back it up. There is no backup, snapshot,
restore or corruption-detection machinery in this package — the store is a
SQLite file and a directory, and copying them is an operator procedure written
down in README.md ("Backing a node up") rather than anything that runs here.
"""

# ── The BLAS thread pin — must run before numpy is imported, or it is inert ──────────────────────
#
# OpenBLAS sizes its worker pool once, when the library loads under `import numpy`; setting the
# variable afterwards measurably does nothing, verified by reading it back through
# `threadpoolctl.threadpool_info()` (the effect, not the string).
#
# It sits at package scope, not in the module that calls LAPACK, because Python initialises parent
# packages before submodules: `import mantle.search.beacon.engine` runs this line first, so the
# whole semantic arm (`search/anchors/*`, `search/beacon/*`, `search/mantle/engine`, `shard/cache`,
# `mesh/anchor_routing` — 14 modules, AST-measured as importing numpy) is covered by this one line.
# That matters because the semantic arm runs SVD on any machine that installs it.
#
# Pinned at 1 because two threads inside `numpy.linalg.eigh` crash this box's OpenBLAS 3 times out
# of 3 unpinned (exit 139), 0 times out of 3 pinned, and can hang instead of crashing, so a single
# green run proves little. The full crash table and the reachability analysis are in
# `agience-prism/py/src/prism/pump.py::PumpLoop.tick`.
#
# `setdefault` — an operator who exported the variable keeps their value, including one that
# reinstates the fault. Deliberate (a default for the unset case, not a policy), and it makes this
# guard weaker than an unconditional set. It also cannot help a process that imported numpy before
# mantle. Both limits are measured by `tests/test_blas_thread_pin.py`.
import os as _os

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
del _os
