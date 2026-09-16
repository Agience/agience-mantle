"""Count the fields a request sends that the model does not have.

MCP and REST reach the same handlers and disagree about strictness. Every MCP tool declares
``additionalProperties: false``, so a typo'd key is refused. The REST models default to pydantic's
``extra="ignore"``, so the identical typo is accepted and dropped and the caller gets a ``201``
describing a write that did not include their field.

``extra="forbid"`` is the end state. It is not taken yet, and the reason is a rule rather than
timidity: ``crystal/dispatcher.py`` posts caller bodies verbatim, so turning REST strict would start
returning ``422`` to callers in the field, and nobody knows how many that is. Characterise the
population before migrating it.

This is the instrument that characterises it. It logs, at WARNING, every unknown key by name, and
refuses nothing — behaviour is unchanged, so it can be turned on everywhere without a decision. Once
the log is quiet across a release, ``extra="forbid"`` becomes a measured decision instead of a bet.

Why a mixin rather than a copied validator: the same twelve lines pasted into each model is a shadow
implementation, and they drift. Inheriting keeps one definition, and a model's own ``model_config``
is untouched by it — ``populate_by_name`` and friends still belong to the model that declares them.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class CountsUnknownFields(BaseModel):
    """Base for a request model that reports, but does not refuse, fields it does not declare."""

    @model_validator(mode="before")
    @classmethod
    def _warn_about_unknown_fields(cls, data):
        if isinstance(data, dict):
            #: An alias is a legal key wherever `populate_by_name` is set, so both spellings count as
            #: known — `from_`/`from` must never be reported as a typo.
            known = set(cls.model_fields) | {
                f.alias for f in cls.model_fields.values() if f.alias}
            unknown = sorted(set(data) - known)
            if unknown:
                #: The model name is in the message because the log is read across endpoints: a
                #: count without a subject cannot tell you which caller to go and fix.
                logger.warning(
                    "%s ignored %d unknown field(s): %s — accepted and dropped",
                    cls.__name__, len(unknown), ", ".join(unknown))
        return data
