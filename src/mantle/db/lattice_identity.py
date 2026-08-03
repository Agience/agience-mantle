"""`db.lattice_identity_legacy`'s twin over the standalone lattice (MANTLE_DB=lattice).

Same function names and signatures; each store side-collection (people / platform_settings /
passkey_credentials / otp_codes) becomes a stamped-`content_type` plane in the one store, ids
namespaced by plane so they can never collide with artifacts or each other. Plain-dict in,
plain-dict out — exactly the store module's contract (`id` key, no store internals).

Selected by `db.identity_backend`; call sites keep importing one module.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INTERNAL = ("_origin", "_seq", "_rev", "_fp", "_type")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Plane:
    """One typed keyed-doc plane: get / find / insert / update / delete / count."""

    def __init__(self, name: str, ct: str):
        self.name = name                      # id prefix, e.g. "person"
        self.ct = ct

    def _id(self, key: str) -> str:
        return self.name + ":" + key

    def _strip(self, raw: Dict[str, Any], key_from_id: bool = True) -> Dict[str, Any]:
        out = {k: v for k, v in raw.items() if k not in _INTERNAL and k != "content_type"}
        if key_from_id and isinstance(out.get("id"), str) and out["id"].startswith(self.name + ":"):
            out["id"] = out["id"][len(self.name) + 1:]
        return out

    def get(self, db, key: str) -> Optional[Dict[str, Any]]:
        raw = db.artifacts.get_artifact(self._id(key))
        if raw is None or raw.get("content_type") != self.ct:
            return None
        return self._strip(raw)

    def all(self, db):
        for raw in db.artifacts.list_artifacts(content_type=self.ct):
            yield self._strip(raw)

    def put(self, db, key: str, doc: Dict[str, Any]) -> None:
        d = {k: v for k, v in doc.items() if v is not None and k != "id"}
        d["id"] = self._id(key)
        d["content_type"] = self.ct
        db.artifacts.put_artifact(d)

    def update(self, db, key: str, updates: Dict[str, Any]) -> bool:
        raw = db.artifacts.get_artifact(self._id(key))
        if raw is None or raw.get("content_type") != self.ct:
            return False
        raw.update(updates)
        db.artifacts.put_artifact(raw)
        return True

    def delete(self, db, key: str) -> bool:
        if self.get(db, key) is None:
            return False
        try:
            db.artifacts.delete_artifact(self._id(key))
            return True
        except Exception:
            return False


_PEOPLE = _Plane("person", "application/vnd.agience.person-record+json")
_SETTINGS = _Plane("setting", "application/vnd.agience.platform-setting+json")
_PASSKEYS = _Plane("passkey", "application/vnd.agience.passkey-credential+json")
_OTP = _Plane("otp", "application/vnd.agience.otp-code+json")


# ============================================================
#  PERSON functions
# ============================================================

def _find_person(db, **filters) -> Optional[Dict[str, Any]]:
    for p in _PEOPLE.all(db):
        if all(p.get(k) == v for k, v in filters.items()):
            return p
    return None


def get_person_by_oidc_identity(db, oidc_provider: str, oidc_subject: str) -> Optional[dict]:
    return _find_person(db, oidc_provider=oidc_provider, oidc_subject=oidc_subject)


def get_person_by_email(db, email: str) -> Optional[dict]:
    return _find_person(db, email=email)


def get_person_by_username(db, username: str) -> Optional[dict]:
    return _find_person(db, username=username)


def get_person_by_google_id(db, google_id: str) -> Optional[dict]:
    return _find_person(db, google_id=google_id)


def get_person_by_id(db, person_id: str) -> Optional[dict]:
    return _PEOPLE.get(db, person_id)


def create_person(db, person_dict: dict) -> Optional[str]:
    try:
        now = _now_iso()
        person_dict.setdefault("created_time", now)
        person_dict.setdefault("modified_time", now)
        person_dict.setdefault("preferences", {})
        key = person_dict.get("id") or person_dict.get("_key")
        if not key:
            import uuid
            key = str(uuid.uuid4())
        _PEOPLE.put(db, key, {k: v for k, v in person_dict.items() if k != "_key"})
        return key
    except Exception:
        logger.exception("Error creating person %s", person_dict.get("email"))
        return None


def update_person(db, person_id: str, updates: dict) -> bool:
    updates["modified_time"] = _now_iso()
    return _PEOPLE.update(db, person_id, updates)


def list_all_people(db) -> List[dict]:
    return list(_PEOPLE.all(db))


def count_people(db) -> int:
    return sum(1 for _ in _PEOPLE.all(db))


def update_person_preferences(db, person_id: str, preferences: dict) -> bool:
    """UPSERT preferences — a person not yet provisioned still gets a sparse doc,
    so writes before first provisioning are never lost (same contract as store)."""
    prefs = preferences or {}
    now = _now_iso()
    try:
        if not _PEOPLE.update(db, person_id, {"preferences": prefs, "modified_time": now}):
            _PEOPLE.put(db, person_id, {"preferences": prefs,
                                        "created_time": now, "modified_time": now})
        return True
    except Exception:
        logger.exception("Error updating preferences for person %s", person_id)
        return False


# ============================================================
#  PLATFORM SETTINGS functions
# ============================================================

def get_platform_setting(db, key: str) -> Optional[dict]:
    return _SETTINGS.get(db, key)


def set_platform_setting(db, key: str, value: str, category: Optional[str] = None,
                         is_secret: bool = False, updated_by: Optional[str] = None) -> bool:
    try:
        now = _now_iso()
        updates: dict = {"value": value, "is_secret": is_secret, "updated_time": now}
        if category is not None:
            updates["category"] = category
        if updated_by is not None:
            updates["updated_by"] = updated_by
        if not _SETTINGS.update(db, key, updates):
            updates["created_time"] = now
            _SETTINGS.put(db, key, updates)
        return True
    except Exception:
        logger.exception("Error setting platform setting %s", key)
        return False


def get_platform_settings_by_category(db, category: str) -> List[dict]:
    return [s for s in _SETTINGS.all(db) if s.get("category") == category]


def get_all_platform_settings(db) -> List[dict]:
    return list(_SETTINGS.all(db))


def delete_platform_setting(db, key: str) -> bool:
    return _SETTINGS.delete(db, key)


# ============================================================
#  PASSKEY CREDENTIAL functions
# ============================================================

def get_passkey_credential(db, credential_id: str) -> Optional[dict]:
    return _PASSKEYS.get(db, credential_id)


def get_passkey_credentials_for_person(db, person_id: str) -> List[dict]:
    return [c for c in _PASSKEYS.all(db) if c.get("person_id") == person_id]


def create_passkey_credential(db, credential_dict: dict) -> Optional[str]:
    try:
        credential_dict.setdefault("created_time", _now_iso())
        credential_dict.setdefault("sign_count", 0)
        key = credential_dict.get("id") or credential_dict.get("_key")
        if not key:
            import uuid
            key = str(uuid.uuid4())
        _PASSKEYS.put(db, key, {k: v for k, v in credential_dict.items() if k != "_key"})
        return key
    except Exception:
        logger.exception("Error creating passkey credential")
        return None


def update_passkey_sign_count(db, credential_id: str, new_sign_count: int) -> bool:
    return _PASSKEYS.update(db, credential_id, {"sign_count": new_sign_count})


def get_passkey_credential_by_id_and_person(db, credential_id: str,
                                            person_id: str) -> Optional[dict]:
    c = _PASSKEYS.get(db, credential_id)
    return c if c is not None and c.get("person_id") == person_id else None


def update_passkey_credential(db, credential_id: str, updates: dict) -> bool:
    return _PASSKEYS.update(db, credential_id, updates)


def delete_passkey_credential(db, credential_id: str) -> bool:
    return _PASSKEYS.delete(db, credential_id)


def delete_passkey_credential_for_person(db, credential_id: str, person_id: str) -> bool:
    if get_passkey_credential_by_id_and_person(db, credential_id, person_id) is None:
        return False
    return _PASSKEYS.delete(db, credential_id)


# ============================================================
#  OTP CODE functions
# ============================================================

def create_otp_code(db, otp_dict: dict) -> Optional[str]:
    try:
        otp_dict.setdefault("created_time", _now_iso())
        otp_dict.setdefault("attempts", 0)
        key = otp_dict.get("id") or otp_dict.get("_key")
        if not key:
            import uuid
            key = str(uuid.uuid4())
        _OTP.put(db, key, {k: v for k, v in otp_dict.items() if k != "_key"})
        return key
    except Exception:
        logger.exception("Error creating OTP code")
        return None


def get_otp_code_by_email(db, email: str) -> Optional[dict]:
    rows = [o for o in _OTP.all(db) if o.get("email") == email]
    rows.sort(key=lambda o: o.get("created_time") or "", reverse=True)
    return rows[0] if rows else None


def delete_otp_code(db, otp_id: str) -> bool:
    return _OTP.delete(db, otp_id)


def increment_otp_attempts(db, otp_id: str) -> bool:
    o = _OTP.get(db, otp_id)
    if o is None:
        return False
    return _OTP.update(db, otp_id, {"attempts": int(o.get("attempts") or 0) + 1})


def get_recent_failed_otp_count(db, email: str, since_iso: str, max_attempts: int) -> int:
    return sum(1 for o in _OTP.all(db)
               if o.get("email") == email
               and (o.get("created_time") or "") >= since_iso
               and o.get("used") is not True
               and int(o.get("attempts") or 0) >= max_attempts)


def get_valid_otp_codes(db, email: str, now_iso: str, max_attempts: int) -> List[dict]:
    rows = [o for o in _OTP.all(db)
            if o.get("email") == email
            and (o.get("expires_at") or "") > now_iso
            and o.get("used") is not True
            and int(o.get("attempts") or 0) < max_attempts]
    rows.sort(key=lambda o: o.get("created_time") or "", reverse=True)
    return rows


def mark_otp_used(db, otp_id: str) -> bool:
    return _OTP.update(db, otp_id, {"used": True})


def delete_expired_otp_codes(db, now_iso: str) -> int:
    doomed = [o["id"] for o in _OTP.all(db)
              if (o.get("expires_at") or "") <= now_iso or o.get("used") is True]
    n = 0
    for oid in doomed:
        if _OTP.delete(db, oid):
            n += 1
    return n
