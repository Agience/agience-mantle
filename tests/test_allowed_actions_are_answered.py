"""Mantle answers what the caller may do with each artifact it reads or lists.

A client shows only the controls in `allowed_actions`, so this answer is the one place that
decides. It is the `check_access` walk, once per action, and asking it records nothing.
"""
from __future__ import annotations

import ast
import io
from unittest.mock import patch

from mantle.routers import artifacts_router
from mantle.services import dependencies
from mantle.services.dependencies import allowed_actions

from .test_dependencies_check_access import (
    _STORE_GRANTS, _STORE_PARENT, _deny_grant, _grants_side_effect, _lattice_with, _make_grant,
    _user,
)


def test_the_answer_is_the_actions_the_grants_hold_in_vocabulary_order():
    db = _lattice_with({"_key": "a-1", "root_id": "a-1"})
    grants = _grants_side_effect({"a-1": [_make_grant("a-1", "share", "read", "delete")]})
    with patch(_STORE_GRANTS, side_effect=grants), patch(_STORE_PARENT, return_value=None):
        assert allowed_actions(_user(), "a-1", db) == ["read", "delete", "share"]


def test_a_deny_leaves_nothing():
    db = _lattice_with({"_key": "a-1", "root_id": "a-1"})
    grants = _grants_side_effect({"a-1": [_make_grant("a-1", "read", "admin"), _deny_grant("a-1")]})
    with patch(_STORE_GRANTS, side_effect=grants), patch(_STORE_PARENT, return_value=None):
        assert allowed_actions(_user(), "a-1", db) == []


def test_asking_is_not_recorded_as_an_access():
    db = _lattice_with({"_key": "a-1", "root_id": "a-1"})
    grants = _grants_side_effect({"a-1": [_make_grant("a-1", "read")]})
    with patch(_STORE_GRANTS, side_effect=grants), patch(_STORE_PARENT, return_value=None), \
            patch("mantle.services.audit_service.record_access") as rec:
        allowed_actions(_user(), "a-1", db)
        assert not rec.called
        dependencies.check_access(_user(), "a-1", "read", db)
        assert rec.called


def _calls(name, fn):
    tree = ast.parse(io.open(artifacts_router.__file__, encoding="utf-8").read())
    [node] = [n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn]
    return any(isinstance(a, ast.Name) and a.id == name
               for c in ast.walk(node) if isinstance(c, ast.Call) for a in [c.func, *c.args])


def test_a_read_and_both_listings_answer_it():
    for fn in ("read_artifact", "list_visible", "list_children"):
        assert _calls("_attach_allowed_actions", fn), fn


def test_a_create_and_an_update_answer_it():
    for fn in ("create_artifact", "update_artifact"):
        assert _calls("_attach_allowed_actions", fn), fn


def test_mantle_answers_the_grants_for_every_doc_the_inbox_included():
    """What a type keeps from trash or delete is refused by Crystal, not hidden here."""
    ws = artifacts_router.WORKSPACE_CONTENT_TYPE
    docs = [{"id": "inbox-1", "root_id": "inbox-1", "content_type": ws},
            {"id": "a-2", "root_id": "a-2", "content_type": "text/plain"}]
    with patch.object(artifacts_router, "allowed_actions",
                      return_value=["read", "update", "delete"]):
        artifacts_router._attach_allowed_actions(object(), _user(), docs)
    assert all(d["allowed_actions"] == ["read", "update", "delete"] for d in docs)
