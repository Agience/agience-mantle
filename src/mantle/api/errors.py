"""The error vocabulary every router on this node declares from, extended to
`/grants` by grants C-1.

One home for the prose, not one home for the builder. What must never be duplicated is what a
code MEANS on this node: the day two surfaces describe the same `404` differently, a client learns
that the answer depends on which endpoint it asked, which is exactly the confusion these
descriptions exist to remove. The per-route assembly is three lines and each surface's differs —
`/artifacts` grew four success-shape keywords out of its own audit — so the builders stay local and
only this table is shared.

Extracted 2026-08-26 when `/grants` needed the same eight codes plus `410`. Before that it lived
in `artifacts_router` and was the only copy; giving `/grants` a second one was the alternative, and
it is the thing this module exists to prevent.
"""
from __future__ import annotations

#: What each status code means on this node, in the caller's terms.
#:
#: These are DESCRIPTIONS, not policy. A route declares the subset it can actually raise, derived
#: from the transitive closure of its handler — never the whole table. Declaring a code a handler
#: cannot produce tells every generated client to write a branch it will never enter, which is the
#: same defect as omitting one, pointed the other way.
ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "Malformed request. The `detail` says which field or value, and is meant for the caller.",
    401: "Credentials were supplied and are not valid. A request with no credentials is not "
         "refused here: it becomes an anonymous principal and is answered by the access rules, "
         "which is why many operations answer a stranger with 404 rather than 401.",
    403: "Refused for a reason the caller cannot fix by authenticating differently.",
    404: "Not found, or found and not permitted. The two are deliberately indistinguishable: the "
         "same 404 is returned for a missing artifact and for a denied one, so this endpoint "
         "cannot be used to discover whether an artifact exists.",
    409: "The operation conflicts with the current state of the resource.",
    #: Added for `/grants` 2026-08-26. An invite is single-use and time-bounded, so "this link is
    #: finished" is a DIFFERENT answer from "no such link" — and a client that cannot tell them
    #: apart shows the wrong thing to the person holding the invite. `404` would collapse them.
    410: "This invite is no longer available — it has already been claimed, or it has expired. "
         "Distinct from 404 on purpose: the link was real, and the person holding it should be "
         "told what became of it rather than that it never existed.",
    413: "The body is larger than this node accepts.",
    500: "The server failed. The `detail` is a stable message; the exception text stays in the log.",
    503: "A dependency this operation needs is not available.",
}
