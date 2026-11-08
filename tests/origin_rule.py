"""Origin's share routes, deciding with Origin's own rule (review 28).

A stand-in for Origin that decides by itself can pass what the real Origin refuses: review 27's
root split did, because its fake released a root's master key that Origin's `may_release` did not.
This one keeps shares as `FakeOrigin` does, but every release and every first-share store is
decided by Origin's real `Reach` (`origin.services.authority_reach`) over a replica the test
describes, so the path runs against the rule Origin applies.
"""
from __future__ import annotations

from origin.services.authority_reach import Grant, MemoryReplica, Reach

from mantle.attenuation import ACTIONS
from mantle.clients.origin_key_share_client import AuthorityRefused

from .test_a_split_master_key_needs_two_custodians import FakeOrigin


def replica_grant(gid, grantee, resource, actions=ACTIONS, effect="allow") -> Grant:
    return Grant(id=gid, grantee_id=grantee, grantee_type="user", resource_id=resource,
                 actions=frozenset(actions), effect=effect)


class RealRuleOrigin(FakeOrigin):
    """`tokens` maps each acting credential to its subject, as Origin's own verification would."""

    def __init__(self, replica: MemoryReplica, tokens: dict):
        super().__init__()
        self.replica = replica
        self.tokens = dict(tokens)
        self.stored_by = {}

    def _subject(self, credential):
        if credential not in self.tokens:
            raise AuthorityRefused(403, "acting credential refused")
        return self.tokens[credential]

    def get_share(self, principal_id, *, credential=None, collection_id=None, artifact_id=None):
        if self.down:
            raise RuntimeError("Origin is unreachable")
        if not str(credential or "").startswith("background:"):
            ok, why = Reach(self.replica).may_release(
                subject=self._subject(credential), principal_id=principal_id,
                collection_id=collection_id, artifact_id=artifact_id)
            if not ok:
                raise AuthorityRefused(403, f"not released: {why}")
        s = self.shares.get(principal_id)
        return {**s, "pending": principal_id in self.pending} if s else None

    def put_share(self, principal_id, share, *, supersedes=None, credential=None):
        subject = self._subject(credential) if credential and not str(credential).startswith(
            "background:") else principal_id
        rc = Reach(self.replica)
        old = self.shares.get(principal_id)
        if old is None:
            may, _why = rc.may_release(subject=subject, principal_id=principal_id,
                                       collection_id=None)
            node = principal_id[3:] if principal_id.startswith("ck:") else principal_id
            if subject != principal_id and not may and not rc.has_any_action(subject, node,
                                                                              "admin"):
                raise AuthorityRefused(403, "not stored: a first share is stored by the key's "
                                            "principal or its creator")
            self.stored_by[principal_id] = subject
        elif old != share:
            node = principal_id[3:] if principal_id.startswith("ck:") else principal_id
            if principal_id not in self.pending:
                raise AuthorityRefused(403, "not replaced: this share is not marked for re-key")
            if subject not in (principal_id, self.stored_by.get(principal_id)) and \
                    not rc.has_any_action(subject, node, "admin"):
                raise AuthorityRefused(403, "not replaced: a share is replaced by the key's "
                                            "principal, Admin on what it keys, or its storer")
        return super().put_share(principal_id, share, supersedes=supersedes,
                                 credential=credential)

    def activate_share(self, principal_id, credential=None):
        super().activate_share(principal_id, credential=credential)
        self.stored_by.pop(principal_id, None)

    def principal_kinds(self, ids):
        rc = Reach(self.replica)
        out = {}
        for i in ids:
            if i.startswith("ck:"):
                out[i] = "collection_key"
            elif self.replica.is_person(i):
                out[i] = "person"
            elif self.replica.root_entry(i) is not None or self.replica.node(i) is not None:
                out[i] = "member" if rc.origin_parent(i) is not None else "root"
            else:
                out[i] = "unknown"
        return out
