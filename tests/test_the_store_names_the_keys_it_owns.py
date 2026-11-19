"""Mantle names the context keys it owns, so an editor can show them read-only without a list of
its own."""
from __future__ import annotations


from mantle.routers import artifacts_router as artifacts
from mantle.services.dependencies import AuthContext
from mantle.services.mint_context import SERVER_OWNED_KEYS

async def test_the_answer_is_every_key_the_store_owns():
    auth = AuthContext(principal_id="u-keys", principal_type="user", user_id="u-keys")
    answer = await artifacts.store_keys(auth=auth)
    assert answer == {"keys": sorted(SERVER_OWNED_KEYS)}
    assert {"minted_by", "provenance", "content_sha256", "upload", "access"} <= set(answer["keys"])

def test_the_path_is_not_read_as_an_artifact_id():
    paths = [r.path for r in artifacts.router.routes]
    assert paths.index("/artifacts/store-keys") < paths.index("/artifacts/{artifact_id}")
