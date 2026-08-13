# entities/grant.py
"""
Grant entity -- unified authorization record.

CRUDEASIO permission model: Create, Read, Update, Delete, Evict, Add, Share, Invoke, Admin.

E (Evict) = remove item from container (delete edge, not the artifact).
A (Add) = fire-and-forget intake into a container (add edge).
S (Share) = ability to create invites / share the resource.
O (Admin) = meta-permission to manage grants on the resource.

See: .dev/features/unified-artifact-api.md
"""

from typing import Optional, Dict, Any

from mantle.attenuation import ACTIONS, FLAG_OF, Mask
from .base import BaseEntity


class Grant(BaseEntity):
    """
    A grant links a principal (user, api_key, invite, group, grant_key) to a
    resource (artifact) with CRUDEASIO permissions.
    """

    PREFIX = "Grant"

    # Valid grantee types
    GRANTEE_USER = "user"
    GRANTEE_INVITE = "invite"
    GRANTEE_GROUP = "group"
    GRANTEE_GRANT_KEY = "grant_key"
    #: A grant held BY another grant — the bundle edge.
    #:
    #: This is what makes a grant composable. A "bundle" is not a separate entity: it
    #: is a ``grant_key`` grant whose ``grantee_id`` other grants name as THEIR
    #: grantee. Expanding a bundle is therefore the same lookup as expanding any
    #: other grantee (`get_active_grants_for_grantee`), which is why composition
    #: needed no new storage, no new edge label, and no second traversal.
    #:
    #: Members carry their own independent CRUDEASIO bits, so one bundle can be
    #: read on one collection and read/write on another. See
    #: :func:`services.grant_key_service.resolve_bundle`.
    GRANTEE_GRANT = "grant"

    # Valid states
    STATE_ACTIVE = "active"
    STATE_REVOKED = "revoked"
    STATE_PENDING_ACCEPT = "pending_accept"

    # Valid effects
    EFFECT_ALLOW = "allow"
    EFFECT_DENY = "deny"
    VALID_EFFECTS = {EFFECT_ALLOW, EFFECT_DENY}

    # Role presets --- named permission bundles for the share/invite flow.
    #
    # Use via ``Grant.permissions_for_role(role_name)`` rather than reading
    # this dict directly so new roles can be added without callers needing
    # to know about every flag.
    ROLE_PRESETS: Dict[str, Dict[str, bool]] = {
        "viewer": {
            "can_read": True,
        },
        "editor": {
            "can_create": True,
            "can_read": True,
            "can_update": True,
            "can_delete": True,
            "can_evict": True,
        },
        "collaborator": {
            "can_create": True,
            "can_read": True,
            "can_update": True,
            "can_delete": True,
            "can_evict": True,
            "can_invoke": True,
            "can_add": True,
            "can_share": True,
        },
        "admin": {
            "can_create": True,
            "can_read": True,
            "can_update": True,
            "can_delete": True,
            "can_evict": True,
            "can_invoke": True,
            "can_add": True,
            "can_share": True,
            "can_admin": True,
        },
    }

    #: CRUDEASIO action name -> permission attribute. The single source of truth;
    #: `services.dependencies` and `services.grant_store` both read it rather than
    #: keeping parallel copies, because a map that disagrees about which flag gates
    #: which action is a silent authorization bug.
    #:
    #: Derived from :mod:`mantle.attenuation`, which owns the action vocabulary because it
    #: owns the operator those actions are composed by. The entity still owns the *storage*
    #: shape (one boolean column per action, named `can_<action>`); what moved out is only
    #: the list of actions, so the mask type and the ledger cannot come to disagree about
    #: what CRUDEASIO contains.
    ACTION_FLAGS: Dict[str, str] = {action: FLAG_OF[action] for action in ACTIONS}

    #: Every permission attribute, in CRUDEASIO order.
    PERMISSION_FLAGS = tuple(ACTION_FLAGS.values())

    def to_mask(self) -> Mask:
        """This grant as an attenuation :class:`~mantle.attenuation.Mask`. See :func:`mask_of`."""
        return mask_of(self)

    def masked_by(self, ceiling: "Grant") -> "Grant":
        """This grant's authority met with *ceiling*'s — the bundle ceiling.

        The bundle rule: a member grant is only ever as strong as the bundle grant
        that carries it, so revoking a bit on the bundle revokes it across every
        member at once without touching them. Intersection only ever removes
        permission, which is what makes it safe to apply unconditionally.

        The narrowing is :meth:`Mask.__and__` — the *one* attenuation operator, shared with
        the light-cone path (`search.mantle.lightcone`) and with the `edge.propagate`
        column. It is not re-implemented here, because the two implementations this replaced
        disagreed about the zero element and one of them was wrong.

        Two details the operator does not decide, and this method must:

        * **Bits are written back effect-blind.** A deny grant's CRUDEASIO bits say *which
          actions it denies* — `check_access` tests ``grant_is_deny(g) and getattr(g, flag)``
          — so masking must not clear them just because the composed authority is a deny.
        * **The effect string is only ever escalated toward deny.** A grant whose effect is
          already not a recognized allow confers nothing anyway; rewriting its string would
          lose information without changing any decision.

        Returns a detached copy — the stored member grant is never mutated, so the
        ceiling stays a property of the presented credential rather than something
        that could bake itself into the record.
        """
        masked = Grant.from_dict(self.to_dict())
        narrowed = mask_of(self) & mask_of(ceiling)
        for flag, value in narrowed.to_flags().items():
            setattr(masked, flag, value)
        # Deny is the absorbing element: a deny member inside an allow bundle must stay a
        # deny, and an allow member under a ceiling that is not a recognized allow becomes
        # one. The second half is why this reads the composed mask rather than the ceiling
        # alone — `grant_is_allow` is positive matching, so an unrecognized ceiling effect
        # narrows to deny instead of passing through.
        if grant_is_allow(self) and not narrowed.is_allow:
            masked.effect = self.EFFECT_DENY
        return masked

    @classmethod
    def permissions_for_role(cls, role: str) -> Dict[str, bool]:
        """Return the CRUDEASIO bit pattern for a named role.

        Raises :class:`ValueError` if *role* is not a known preset.
        """
        preset = cls.ROLE_PRESETS.get(role)
        if preset is None:
            raise ValueError(
                f"Unknown role {role!r}; valid roles: {sorted(cls.ROLE_PRESETS)}"
            )
        return dict(preset)

    def __init__(
        self,
        resource_id: str,             # artifact_id
        grantee_type: str,            # "user" | "api_key" | "invite" | "group" | "grant_key"
        grantee_id: str,              # user_id | api_key.id | claim_token_hash | group_artifact_id
        granted_by: str,              # user_id of the issuer
        effect: str = "allow",        # "allow" | "deny"
        # CRUDEASIO permission flags
        can_create: bool = False,
        can_read: bool = True,
        can_update: bool = False,
        can_delete: bool = False,
        can_evict: bool = False,
        can_invoke: bool = False,
        can_add: bool = False,
        can_share: bool = False,
        can_admin: bool = False,
        # Identity requirements
        requires_identity: bool = False,
        read_requires_identity: Optional[bool] = None,
        write_requires_identity: Optional[bool] = None,
        invoke_requires_identity: Optional[bool] = None,
        # Invite targeting (grantee_type == "invite" only)
        target_entity: Optional[str] = None,         # email, user_id, google_id, domain, etc.
        target_entity_type: Optional[str] = None,    # "email" | "user_id" | "google_id" | "domain"
        max_claims: Optional[int] = None,            # None = unlimited, 0 = frozen, 1 = single-use
        claims_count: int = 0,
        # Bearer-key grants (grantee_type == "grant_key" only)
        #: Non-secret tail of the raw token ("…a1b2"), for humans picking a key out of a
        #: list. The secret itself is never stored — `grantee_id` holds only its hash.
        key_hint: Optional[str] = None,
        last_used_at: Optional[str] = None,
        #: Demand a signed challenge header alongside the token (bot protection on
        #: inbound webhook keys). Enforced by `dependencies.check_inbound_nonce`,
        #: which routes must opt into explicitly.
        requires_nonce: bool = False,
        # Lifecycle
        state: str = "active",
        id: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        granted_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        accepted_by: Optional[str] = None,
        accepted_at: Optional[str] = None,
        revoked_by: Optional[str] = None,
        revoked_at: Optional[str] = None,
        created_time: Optional[str] = None,
        modified_time: Optional[str] = None,
    ):
        super().__init__(id=id, created_time=created_time, modified_time=modified_time)
        self.resource_id = resource_id
        self.grantee_type = grantee_type
        self.grantee_id = grantee_id
        self.granted_by = granted_by
        # Normalized at construction so storage can never hold a casing variant.
        # See `is_deny` / `is_allow` for why a raw comparison was dangerous.
        self.effect = (effect or "").strip().lower() or self.EFFECT_ALLOW
        # CRUDEASIO
        self.can_create = can_create
        self.can_read = can_read
        self.can_update = can_update
        self.can_delete = can_delete
        self.can_evict = can_evict
        self.can_invoke = can_invoke
        self.can_add = can_add
        self.can_share = can_share
        self.can_admin = can_admin
        # Identity requirements
        self.requires_identity = requires_identity
        self.read_requires_identity = read_requires_identity
        self.write_requires_identity = write_requires_identity
        self.invoke_requires_identity = invoke_requires_identity
        # Invite targeting
        self.target_entity = target_entity
        self.target_entity_type = target_entity_type
        self.max_claims = max_claims
        self.claims_count = claims_count
        # Bearer-key
        self.key_hint = key_hint
        self.last_used_at = last_used_at
        self.requires_nonce = requires_nonce
        # Lifecycle
        self.state = state
        self.name = name
        self.notes = notes
        self.granted_at = granted_at or self.created_time
        self.expires_at = expires_at
        self.accepted_by = accepted_by
        self.accepted_at = accepted_at
        self.revoked_by = revoked_by
        self.revoked_at = revoked_at

    def is_deny(self) -> bool:
        """True when this grant DENIES. See :func:`grant_is_deny`."""
        return grant_is_deny(self)

    def is_allow(self) -> bool:
        """True only for an explicit, recognized allow. See :func:`grant_is_allow`."""
        return grant_is_allow(self)

    def is_active(self) -> bool:
        return self.state == self.STATE_ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        base = self.to_dict_base()
        base.update({
            "resource_id": self.resource_id,
            "grantee_type": self.grantee_type,
            "grantee_id": self.grantee_id,
            "granted_by": self.granted_by,
            "effect": self.effect,
            # CRUDEASIO
            "can_create": self.can_create,
            "can_read": self.can_read,
            "can_update": self.can_update,
            "can_delete": self.can_delete,
            "can_evict": self.can_evict,
            "can_invoke": self.can_invoke,
            "can_add": self.can_add,
            "can_share": self.can_share,
            "can_admin": self.can_admin,
            # Identity
            "requires_identity": self.requires_identity,
            "read_requires_identity": self.read_requires_identity,
            "write_requires_identity": self.write_requires_identity,
            "invoke_requires_identity": self.invoke_requires_identity,
            # Invite
            "target_entity": self.target_entity,
            "target_entity_type": self.target_entity_type,
            "max_claims": self.max_claims,
            "claims_count": self.claims_count,
            # Bearer-key
            "key_hint": self.key_hint,
            "last_used_at": self.last_used_at,
            "requires_nonce": self.requires_nonce,
            # Lifecycle
            "state": self.state,
            "name": self.name,
            "notes": self.notes,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "accepted_by": self.accepted_by,
            "accepted_at": self.accepted_at,
            "revoked_by": self.revoked_by,
            "revoked_at": self.revoked_at,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Grant":
        base = cls.from_dict_base(data)
        can_update = data.get("can_update", False)
        return cls(
            id=base["id"],
            resource_id=data.get("resource_id", ""),
            grantee_type=data.get("grantee_type", ""),
            grantee_id=data.get("grantee_id", ""),
            granted_by=data.get("granted_by", ""),
            effect=data.get("effect", "allow"),
            # CRUDEASIO
            can_create=data.get("can_create", False),
            can_read=data.get("can_read", True),
            can_update=can_update,
            can_delete=data.get("can_delete", False),
            can_evict=data.get("can_evict", False),
            can_invoke=data.get("can_invoke", False),
            can_add=data.get("can_add", False),
            can_share=data.get("can_share", False),
            can_admin=data.get("can_admin", False),
            # Identity
            requires_identity=data.get("requires_identity", False),
            read_requires_identity=data.get("read_requires_identity"),
            write_requires_identity=data.get("write_requires_identity"),
            invoke_requires_identity=data.get("invoke_requires_identity"),
            # Invite
            target_entity=data.get("target_entity"),
            target_entity_type=data.get("target_entity_type"),
            max_claims=data.get("max_claims"),
            claims_count=data.get("claims_count", 0),
            # Bearer-key
            key_hint=data.get("key_hint"),
            last_used_at=data.get("last_used_at"),
            requires_nonce=data.get("requires_nonce", False),
            # Lifecycle
            state=data.get("state", "active"),
            name=data.get("name"),
            notes=data.get("notes"),
            granted_at=data.get("granted_at"),
            expires_at=data.get("expires_at"),
            accepted_by=data.get("accepted_by"),
            accepted_at=data.get("accepted_at"),
            revoked_by=data.get("revoked_by"),
            revoked_at=data.get("revoked_at"),
            created_time=base["created_time"],
            modified_time=base["modified_time"],
        )


# ---------------------------------------------------------------------------
# Effect predicates — module-level, duck-typed on purpose
# ---------------------------------------------------------------------------
#
#
# These are FUNCTIONS, not just entity methods, because grant-like objects reach
# the enforcement path from several producers (entities, AQL row shims, test
# doubles). Requiring a specific class here would make authorization depend on
# which producer happened to build the object.


def _effect_of(grant: Any) -> str:
    return str(getattr(grant, "effect", None) or "").strip().lower()


def grant_is_deny(grant: Any) -> bool:
    """True when *grant* denies."""
    return _effect_of(grant) == Grant.EFFECT_DENY


def grant_is_allow(grant: Any) -> bool:
    """True only for an explicit, recognized allow.

    Deliberately NOT ``not grant_is_deny(...)``: an unrecognized effect is
    neither an allow nor a deny and must confer nothing. Positive matching is
    what makes an unknown value fail closed instead of open.
    """
    return _effect_of(grant) == Grant.EFFECT_ALLOW


def mask_of(grant: Any) -> Mask:
    """*grant* as an attenuation :class:`~mantle.attenuation.Mask` — the adapter between the
    ledger's boolean columns and the one operator.

    This is the only place the two halves of a grant's authority are joined into one value:
    its nine CRUDEASIO booleans, and whether it is an allow at all. Every enforcement point
    that used to write ``getattr(grant, flag_attr, False)`` should ask
    ``mask_of(grant).allows(action)`` instead — the bare `getattr` answers only half the
    question, which is precisely how a deny grant came to read as authorizing on the
    light-cone path.

    Duck-typed for the same reason the effect predicates are: grant-like objects reach
    enforcement from several producers, and authorization must not depend on which one built
    the object. ``allow`` comes from :func:`grant_is_allow`, so an unrecognized effect lands
    as a non-allow mask and confers nothing.
    """
    return Mask.from_flags(grant, allow=grant_is_allow(grant))
