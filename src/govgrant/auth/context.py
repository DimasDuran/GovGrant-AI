"""
Resolve AuthContext for a request (CLI / UI / future API).

Modes:
  - AUTH_ENABLED=false (default): use DEFAULT_TENANT_ID; optional api_key ignored.
  - AUTH_ENABLED=true: require valid API key → tenant; optional tenant override
    only if key maps to that tenant (or role admin — future).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from govgrant.auth.registry import AuthRegistry, TenantRecord, load_auth_registry
from govgrant.rag.config import get_settings


class AuthError(Exception):
    """Authentication / authorization failure."""


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    roles: tuple[str, ...]
    api_key_present: bool
    auth_enabled: bool
    allowed_doc_ids: frozenset[str] | None
    """None = unrestricted under tenant; set = allow-list (+ public docs)."""
    public_doc_ids: frozenset[str]
    source: str  # env_default | api_key | explicit_tenant

    def may_access_doc(self, doc_id: str | None) -> bool:
        if not doc_id:
            return True
        if doc_id in self.public_doc_ids:
            return True
        if self.allowed_doc_ids is None:
            return True
        return doc_id in self.allowed_doc_ids

    def filter_doc_id(self, doc_id: str | None) -> str | None:
        """Return doc_id if allowed, else raise AuthError."""
        if doc_id is None:
            return None
        if not self.may_access_doc(doc_id):
            raise AuthError(
                f"Document {doc_id!r} is not allowed for tenant {self.tenant_id!r}"
            )
        return doc_id

    def has_role(self, *roles: str) -> bool:
        want = {r.lower() for r in roles}
        return bool(want & {r.lower() for r in self.roles})

    def require_role(self, *roles: str) -> None:
        if not self.has_role(*roles):
            raise AuthError(
                f"Requires role in {roles!r}; tenant {self.tenant_id!r} has {self.roles!r}"
            )

    def require_admin_for_destructive(self) -> None:
        """
        When AUTH_ENABLED, destructive ops (proposal delete) require admin.

        Open local mode (auth disabled) allows any resolved tenant context.
        """
        if not self.auth_enabled:
            return
        self.require_role("admin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "roles": list(self.roles),
            "api_key_present": self.api_key_present,
            "auth_enabled": self.auth_enabled,
            "allowed_doc_ids": (
                None if self.allowed_doc_ids is None else sorted(self.allowed_doc_ids)
            ),
            "source": self.source,
        }


def auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_auth_registry() -> AuthRegistry:
    return load_auth_registry()


def clear_auth_cache() -> None:
    get_auth_registry.cache_clear()


def resolve_request_auth(
    *,
    api_key: str | None = None,
    tenant_id: str | None = None,
    registry: AuthRegistry | None = None,
    require_auth: bool | None = None,
) -> AuthContext:
    """
    Resolve tenant for this request.

    require_auth: override AUTH_ENABLED (useful in tests).
    """
    enabled = auth_enabled() if require_auth is None else require_auth
    settings = get_settings()
    reg = registry or get_auth_registry()
    key = (api_key or "").strip() or None
    explicit = (tenant_id or "").strip() or None

    if not enabled:
        tid = explicit or settings.default_tenant_id
        rec = reg.get_tenant(tid)
        return AuthContext(
            tenant_id=tid,
            roles=rec.roles if rec else ("user",),
            api_key_present=bool(key),
            auth_enabled=False,
            allowed_doc_ids=rec.allowed_doc_ids if rec else None,
            public_doc_ids=reg.public_doc_ids,
            source="explicit_tenant" if explicit else "env_default",
        )

    # Auth required
    if not key:
        raise AuthError("AUTH_ENABLED: missing API key")
    rec = reg.tenant_for_key(key)
    if rec is None:
        raise AuthError("AUTH_ENABLED: invalid API key")
    if explicit and explicit != rec.tenant_id:
        raise AuthError(
            f"AUTH_ENABLED: tenant {explicit!r} does not match API key binding "
            f"{rec.tenant_id!r}"
        )
    return AuthContext(
        tenant_id=rec.tenant_id,
        roles=rec.roles,
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=rec.allowed_doc_ids,
        public_doc_ids=reg.public_doc_ids,
        source="api_key",
    )
