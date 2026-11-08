"""Review 22, F1: a persona's own delegation is addressed to Mantle, and Mantle takes it only so.

Origin addresses the server delegation a persona asks for (its actor proven by its own key,
`act.iss`) to Mantle, so the persona never takes it back from Mantle. Mantle refuses one addressed
to anyone else, and bounds the narrowed one exactly as the persona's own.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as deps


def _payload(aud, *, iss="lumen"):
    act = {"sub": "agience-server-lumen", **({"iss": iss} if iss else {})}
    return {"principal_type": "delegation", "aud": aud, "act": act, "sub": "p-1"}


def test_a_persona_s_delegation_addressed_to_the_persona_is_refused():
    with pytest.raises(HTTPException) as refused:
        deps._validate_aud_for_principal(_payload("agience-server-lumen"))
    assert refused.value.status_code == 401


def test_one_addressed_to_mantle_is_taken():
    deps._validate_aud_for_principal(_payload("mantle"))


def test_a_describe_delegation_carrying_no_proven_actor_is_not_this_rule_s():
    deps._validate_aud_for_principal(_payload("agience-server-lumen", iss=None))


def test_the_narrowed_one_is_bounded_as_the_persona_s_own():
    auth = deps.AuthContext(principal_id="p-1", principal_type="user", user_id="p-1",
                            actor="agience-server-lumen", actor_signer="lumen",
                            delegation_aud="mantle", narrowed=True, scope="persona:lumen")
    assert auth.narrowed_audience == "agience-server-lumen"
    req = SimpleNamespace(method="POST", url=SimpleNamespace(path="/artifacts/a-1/use"))
    deps.enforce_narrowed_bounds(auth, req)
