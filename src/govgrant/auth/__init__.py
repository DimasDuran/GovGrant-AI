"""Dev auth / multi-tenant resolution (scalable foundation)."""

from govgrant.auth.context import AuthContext, AuthError, resolve_request_auth
from govgrant.auth.registry import AuthRegistry, load_auth_registry

__all__ = [
    "AuthContext",
    "AuthError",
    "AuthRegistry",
    "load_auth_registry",
    "resolve_request_auth",
]
