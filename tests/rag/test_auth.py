"""Unit tests for dev auth / multi-tenant resolution (no Qdrant)."""

from __future__ import annotations

import pytest

from govgrant.auth.context import AuthError, clear_auth_cache, resolve_request_auth
from govgrant.auth.registry import AuthRegistry, TenantRecord, load_auth_registry


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_auth_cache()
    yield
    clear_auth_cache()


def test_load_example_registry():
    reg = load_auth_registry()
    assert "local-dev" in reg.tenants
    assert reg.tenant_for_key("dev-local-key") is not None
    assert "darpa-sbir-sttr-phase-II-instructions" in reg.public_doc_ids


def test_auth_disabled_uses_default_tenant(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "local-dev")
    ctx = resolve_request_auth()
    assert ctx.tenant_id == "local-dev"
    assert ctx.auth_enabled is False
    assert ctx.source == "env_default"


def test_auth_enabled_requires_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    with pytest.raises(AuthError, match="missing API key"):
        resolve_request_auth(require_auth=True)


def test_auth_enabled_valid_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    ctx = resolve_request_auth(api_key="dev-local-key", require_auth=True)
    assert ctx.tenant_id == "local-dev"
    assert ctx.api_key_present
    assert ctx.source == "api_key"


def test_auth_enabled_invalid_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    with pytest.raises(AuthError, match="invalid API key"):
        resolve_request_auth(api_key="nope", require_auth=True)


def test_tenant_mismatch_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    with pytest.raises(AuthError, match="does not match"):
        resolve_request_auth(
            api_key="dev-local-key",
            tenant_id="demo-acme",
            require_auth=True,
        )


def test_has_role_helpers():
    from govgrant.auth.context import AuthContext

    ctx = AuthContext(
        tenant_id="t",
        roles=("admin", "user"),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="api_key",
    )
    assert ctx.has_role("admin")
    assert ctx.has_role("ADMIN")  # case-insensitive
    assert not ctx.has_role("viewer")
    ctx.require_role("user")
    with pytest.raises(AuthError):
        ctx.require_role("superadmin")


def test_capabilities_admin_vs_user():
    from govgrant.auth.context import AuthContext

    admin = AuthContext(
        tenant_id="t",
        roles=("admin",),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="api_key",
    )
    user = AuthContext(
        tenant_id="t",
        roles=("user",),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="api_key",
    )
    open_local = AuthContext(
        tenant_id="local-dev",
        roles=("user",),
        api_key_present=False,
        auth_enabled=False,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="env_default",
    )
    restricted = AuthContext(
        tenant_id="beta",
        roles=("user",),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=frozenset(),  # public only
        public_doc_ids=frozenset({"darpa-x"}),
        source="api_key",
    )

    assert admin.can_delete_proposals() is True
    assert admin.can_upload_proposals() is True
    assert admin.capabilities()["admin"] is True

    assert user.can_delete_proposals() is False
    assert user.can_upload_proposals() is True
    assert user.capabilities()["delete_proposals"] is False

    # Open local mode: delete allowed even without admin role
    assert open_local.can_delete_proposals() is True

    assert restricted.can_upload_proposals() is False
    assert "capabilities" in admin.to_dict()


def test_doc_allow_list():
    reg = AuthRegistry(
        tenants={
            "t1": TenantRecord(
                tenant_id="t1",
                api_keys=("k1",),
                allowed_doc_ids=frozenset({"user-proposal-a"}),
            )
        },
        key_to_tenant={"k1": "t1"},
        public_doc_ids=frozenset({"darpa-sbir-sttr-phase-II-instructions"}),
    )
    ctx = resolve_request_auth(
        api_key="k1",
        require_auth=True,
        registry=reg,
    )
    assert ctx.may_access_doc("darpa-sbir-sttr-phase-II-instructions")
    assert ctx.may_access_doc("user-proposal-a")
    assert not ctx.may_access_doc("user-proposal-secret")
    with pytest.raises(AuthError):
        ctx.filter_doc_id("user-proposal-secret")
