"""Review 27: the seed planner reaches neither the real Origin nor the real object store.

`seed_plan.main` imported `mantle.db.lattice_api` (and with it `mantle.config`) before it popped
`ORIGIN_URI` and the buckets, popped too few settings (the edge store's credentials, the root user
and AWS's stayed), and then let the copied store's own platform settings rebind Origin's address
and the bucket. Now the process is isolated before any `mantle` module is imported, every
object-store, AWS and mesh setting goes, and after the copy's settings load `_seal` takes Origin
and the buckets away again, with object-store clients and the content tier's remote that fail.

Run in a child interpreter, in `main`'s order, with harmless local stand-ins for production's
settings. Nothing here opens a connection.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(r"""
    import json, os, sys, tempfile
    from mantle.system import seed_plan as sp            # what `python -m` imports first
    imported_config_early = "mantle.config" in sys.modules
    folder = tempfile.mkdtemp(prefix="r27-plan-")
    sp._isolate(folder, os.path.join(folder, "plan-copy.db"))
    left = sorted(k for k in os.environ if k.startswith(("CONTENT_", "AWS_")) or k == "ORIGIN_URI")
    from mantle import config
    copied = {"branding.origin_uri": "http://127.0.0.1:1/origin-from-the-copied-store",
              "storage.content_bucket": "bucket-from-the-copied-store"}
    config.set_settings_provider(copied.get)
    config.load_settings_from_db()
    sp._seal()
    from mantle.services import content_service as cs
    from mantle.services import authority_ledger as al
    from mantle.shard import content_tier
    def fails(fn):
        try:
            fn()
        except Exception:
            return True
        return False
    print(json.dumps({
        "imported_config_early": imported_config_early,
        "left": left,
        "config_ORIGIN_URI": config.ORIGIN_URI,
        "edge_bucket": cs._EDGE_BUCKET,
        "edge_store_configured": bool(cs.edge_store_configured()),
        "durable": bool(cs._durable_store_enabled()),
        "ledger_mode": al.mode(),
        "client_fails": fails(lambda: cs._make_s3_client(access_key_id="a",
                              secret_access_key="b", region_name="r")),
        "edge_call_fails": fails(lambda: cs._s3_edge_internal.put_object(Bucket="b", Key="k")),
        "remote_fails": fails(lambda: content_tier.open_ovh_store(None)),
    }))
""")


def test_planning_reaches_no_origin_and_no_object_store(tmp_path):
    env = dict(os.environ)
    base = tmp_path / "base"
    base.mkdir()
    env.update({
        "OMP_NUM_THREADS": "1", "AGIENCE_NO_DOTENV": "1", "AGIENCE_BASE_DIR": str(base),
        "MANTLE_LATTICE_PATH": str(base / "real-store.db"),
        "ORIGIN_URI": "http://127.0.0.1:1/origin-from-the-environment",
        "CONTENT_EDGE_BUCKET": "edge-bucket-from-the-environment",
        "CONTENT_EDGE_ACCESS_KEY_ID": "not-a-real-key-id",
        "CONTENT_EDGE_SECRET_ACCESS_KEY": "not-a-real-secret",
        "CONTENT_ROOT_USER": "not-a-real-user", "CONTENT_ROOT_PASSWORD": "not-a-real-password",
        "AWS_ENDPOINT_URL": "http://127.0.0.1:1", "AWS_ACCESS_KEY_ID": "not-a-real-key-id",
    })
    out = subprocess.run([sys.executable, "-I", "-c", CHILD], env=env, capture_output=True,
                         text=True, timeout=120, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-4000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen["imported_config_early"] is False
    assert seen["left"] == []
    assert seen["config_ORIGIN_URI"] == ""
    assert seen["edge_bucket"] == ""
    assert seen["edge_store_configured"] is False
    assert seen["durable"] is False
    assert seen["ledger_mode"] == "off"
    assert seen["client_fails"] and seen["edge_call_fails"] and seen["remote_fails"]


def test_the_planner_names_its_store_rather_than_reading_the_config(tmp_path):
    out = subprocess.run([sys.executable, "-m", "mantle.system.seed_plan", "--pass", "startup",
                          "--out", str(tmp_path / "p.json")],
                         env={**os.environ, "MANTLE_LATTICE_PATH": ""}, capture_output=True,
                         text=True, timeout=120, cwd=str(ROOT))
    assert out.returncode != 0 and "name the store" in out.stderr
