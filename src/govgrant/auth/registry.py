"""
Tenant + API key registry.

Load order (first hit wins for file):
  1. AUTH_REGISTRY_PATH env
  2. data/auth/tenants.local.json  (gitignored secrets)
  3. data/auth/tenants.example.json (safe defaults for local-dev)

Schema (JSON):
{
  "version": 1,
  "tenants": [
    {
      "tenant_id": "local-dev",
      "name": "Local development",
      "api_keys": ["dev-local-key"],
      "roles": ["admin"],
      "allowed_doc_ids": null
    }
  ],
  "public_doc_ids": ["darpa-...", "SBA ...", "SF424 ..."]
}

allowed_doc_ids: null | omit → all docs under tenant + public corpus
allowed_doc_ids: [] → only public corpus
allowed_doc_ids: ["user-proposal-foo"] → those + public
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from govgrant.rag.config import REPO_ROOT

DEFAULT_EXAMPLE = REPO_ROOT / "data" / "auth" / "tenants.example.json"
DEFAULT_LOCAL = REPO_ROOT / "data" / "auth" / "tenants.local.json"


@dataclass(frozen=True)
class TenantRecord:
    tenant_id: str
    name: str = ""
    api_keys: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("user",)
    # None = unrestricted within tenant scope; frozenset = explicit allow-list
    allowed_doc_ids: frozenset[str] | None = None


@dataclass
class AuthRegistry:
    tenants: dict[str, TenantRecord] = field(default_factory=dict)
    key_to_tenant: dict[str, str] = field(default_factory=dict)
    public_doc_ids: frozenset[str] = field(default_factory=frozenset)
    version: int = 1

    def tenant_for_key(self, api_key: str | None) -> TenantRecord | None:
        if not api_key:
            return None
        tid = self.key_to_tenant.get(api_key.strip())
        if not tid:
            return None
        return self.tenants.get(tid)

    def get_tenant(self, tenant_id: str) -> TenantRecord | None:
        return self.tenants.get(tenant_id)


def _parse_tenant(raw: dict[str, Any]) -> TenantRecord:
    tid = str(raw["tenant_id"]).strip()
    keys = tuple(str(k).strip() for k in (raw.get("api_keys") or []) if str(k).strip())
    roles = tuple(str(r).strip() for r in (raw.get("roles") or ["user"]) if str(r).strip())
    allowed_raw = raw.get("allowed_doc_ids", None)
    allowed: frozenset[str] | None
    if allowed_raw is None:
        allowed = None
    else:
        allowed = frozenset(str(x) for x in allowed_raw)
    return TenantRecord(
        tenant_id=tid,
        name=str(raw.get("name") or tid),
        api_keys=keys,
        roles=roles or ("user",),
        allowed_doc_ids=allowed,
    )


def load_auth_registry(path: Path | str | None = None) -> AuthRegistry:
    """Load registry from explicit path or default cascade."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.getenv("AUTH_REGISTRY_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([DEFAULT_LOCAL, DEFAULT_EXAMPLE])

    data: dict[str, Any] | None = None
    used: Path | None = None
    for cand in candidates:
        if cand.is_file():
            data = json.loads(cand.read_text(encoding="utf-8"))
            used = cand
            break
    if data is None:
        # Minimal in-memory fallback
        return AuthRegistry(
            tenants={
                "local-dev": TenantRecord(
                    tenant_id="local-dev",
                    name="Local development",
                    api_keys=("dev-local-key",),
                    roles=("admin",),
                    allowed_doc_ids=None,
                )
            },
            key_to_tenant={"dev-local-key": "local-dev"},
            public_doc_ids=frozenset(),
            version=1,
        )

    tenants: dict[str, TenantRecord] = {}
    key_map: dict[str, str] = {}
    for raw in data.get("tenants") or []:
        rec = _parse_tenant(raw)
        tenants[rec.tenant_id] = rec
        for k in rec.api_keys:
            if k in key_map and key_map[k] != rec.tenant_id:
                raise ValueError(f"Duplicate API key mapping for key ending …{k[-4:]}")
            key_map[k] = rec.tenant_id

    public = frozenset(str(x) for x in (data.get("public_doc_ids") or []))
    reg = AuthRegistry(
        tenants=tenants,
        key_to_tenant=key_map,
        public_doc_ids=public,
        version=int(data.get("version") or 1),
    )
    # Attach path for debugging (not frozen on registry)
    reg._source_path = str(used) if used else None  # type: ignore[attr-defined]
    return reg
