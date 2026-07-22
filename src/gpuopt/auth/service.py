from __future__ import annotations

import logging
from typing import Any

from .backends import (
    AuthResult,
    AuthenticationBackend,
    LDAPBackend,
    OAuth2PasswordBackend,
    OIDCBackend,
)

logger = logging.getLogger(__name__)


class ExternalAuthService:
    def __init__(self, settings: Any = None) -> None:
        self._backends: list[AuthenticationBackend] = []
        self._enabled = False
        if settings is not None:
            self._configure(settings)

    def _configure(self, settings: Any) -> None:
        if settings.oauth2_token_url or settings.oauth2_verify_url:
            self._backends.append(
                OAuth2PasswordBackend(
                    token_url=settings.oauth2_token_url,
                    client_id=settings.oauth2_client_id,
                    client_secret=settings.oauth2_client_secret,
                    verify_url=settings.oauth2_verify_url,
                )
            )
            self._enabled = True
            logger.info("OAuth2 backend configured")
        if settings.oidc_issuer_url:
            self._backends.append(
                OIDCBackend(
                    issuer_url=settings.oidc_issuer_url,
                    client_id=settings.oidc_client_id,
                )
            )
            self._enabled = True
            logger.info("OIDC backend configured")
        if settings.ldap_server_url:
            self._backends.append(
                LDAPBackend(
                    server_url=settings.ldap_server_url,
                    bind_dn=settings.ldap_bind_dn,
                    bind_password=settings.ldap_bind_password,
                    search_base=settings.ldap_search_base,
                    search_filter=settings.ldap_search_filter,
                )
            )
            self._enabled = True
            logger.info("LDAP backend configured")

    def authenticate(self, token: str) -> AuthResult:
        if not self._enabled or not self._backends:
            return AuthResult(error="No external auth backends configured")
        for backend in self._backends:
            result = backend.authenticate(token)
            if result.authenticated:
                return result
            logger.debug("Backend %s failed: %s", type(backend).__name__, result.error)
        return AuthResult(error="All authentication backends rejected the token")
