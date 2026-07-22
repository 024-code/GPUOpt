from __future__ import annotations

import pytest

from gpuopt.auth import AuthenticationBackend, ExternalAuthService, OAuth2PasswordBackend, OIDCBackend, LDAPBackend
from gpuopt.auth.backends import AuthResult


class TestAuthBackends:
    def test_oauth2_not_configured(self):
        backend = OAuth2PasswordBackend()
        result = backend.authenticate("test-token")
        assert result.authenticated is False
        assert "not configured" in result.error

    def test_oidc_not_configured(self):
        backend = OIDCBackend()
        result = backend.authenticate("test-token")
        assert result.authenticated is False
        assert "not configured" in result.error

    def test_ldap_not_configured(self):
        backend = LDAPBackend()
        result = backend.authenticate("test-token")
        assert result.authenticated is False
        assert "not configured" in result.error

    def test_oauth2_verify_failure_bad_url(self):
        backend = OAuth2PasswordBackend(verify_url="http://localhost:1/verify")
        result = backend.authenticate("Bearer test")
        assert result.authenticated is False
        assert result.error != ""

    def test_oidc_verify_failure_no_jwks(self):
        backend = OIDCBackend(issuer_url="http://localhost:1")
        result = backend.authenticate("test-token")
        assert result.authenticated is False


class TestExternalAuthService:
    def test_no_backends(self):
        service = ExternalAuthService()
        result = service.authenticate("test")
        assert result.authenticated is False
        assert "No external auth backends" in result.error

    def test_configure_from_settings(self):
        settings = type("Settings", (), {
            "oauth2_token_url": "",
            "oauth2_verify_url": "",
            "oauth2_client_id": "",
            "oauth2_client_secret": "",
            "oidc_issuer_url": "",
            "oidc_client_id": "",
            "ldap_server_url": "",
            "ldap_bind_dn": "",
            "ldap_bind_password": "",
            "ldap_search_base": "",
            "ldap_search_filter": "(uid={username})",
        })()
        service = ExternalAuthService(settings)
        assert service._enabled is False

    def test_oauth2_enabled(self):
        settings = type("Settings", (), {
            "oauth2_token_url": "http://example.com/token",
            "oauth2_verify_url": "http://example.com/verify",
            "oauth2_client_id": "test",
            "oauth2_client_secret": "secret",
            "oidc_issuer_url": "",
            "oidc_client_id": "",
            "ldap_server_url": "",
            "ldap_bind_dn": "",
            "ldap_bind_password": "",
            "ldap_search_base": "",
            "ldap_search_filter": "(uid={username})",
        })()
        service = ExternalAuthService(settings)
        assert service._enabled is True
        assert len(service._backends) == 1

    def test_oidc_enabled(self):
        settings = type("Settings", (), {
            "oauth2_token_url": "",
            "oauth2_verify_url": "",
            "oauth2_client_id": "",
            "oauth2_client_secret": "",
            "oidc_issuer_url": "http://example.com/auth",
            "oidc_client_id": "test",
            "ldap_server_url": "",
            "ldap_bind_dn": "",
            "ldap_bind_password": "",
            "ldap_search_base": "",
            "ldap_search_filter": "(uid={username})",
        })()
        service = ExternalAuthService(settings)
        assert service._enabled is True
        assert len(service._backends) == 1

    def test_ldap_enabled(self):
        settings = type("Settings", (), {
            "oauth2_token_url": "",
            "oauth2_verify_url": "",
            "oauth2_client_id": "",
            "oauth2_client_secret": "",
            "oidc_issuer_url": "",
            "oidc_client_id": "",
            "ldap_server_url": "ldap://localhost:389",
            "ldap_bind_dn": "cn=admin,dc=example,dc=com",
            "ldap_bind_password": "secret",
            "ldap_search_base": "dc=example,dc=com",
            "ldap_search_filter": "(uid={username})",
        })()
        service = ExternalAuthService(settings)
        assert service._enabled is True
        assert len(service._backends) == 1

    def test_multiple_backends(self):
        settings = type("Settings", (), {
            "oauth2_token_url": "http://example.com/token",
            "oauth2_verify_url": "http://example.com/verify",
            "oauth2_client_id": "test",
            "oauth2_client_secret": "secret",
            "oidc_issuer_url": "http://example.com/auth",
            "oidc_client_id": "test",
            "ldap_server_url": "ldap://localhost:389",
            "ldap_bind_dn": "cn=admin,dc=example,dc=com",
            "ldap_bind_password": "secret",
            "ldap_search_base": "dc=example,dc=com",
            "ldap_search_filter": "(uid={username})",
        })()
        service = ExternalAuthService(settings)
        assert service._enabled is True
        assert len(service._backends) == 3

    def test_config_settings_have_new_fields(self):
        from gpuopt.config import get_settings
        get_settings.cache_clear()
        settings = get_settings()
        assert hasattr(settings, "oauth2_token_url")
        assert hasattr(settings, "oidc_issuer_url")
        assert hasattr(settings, "ldap_server_url")
        assert hasattr(settings, "healing_monitor_interval")
