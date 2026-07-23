from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = "development"
    database_path: Path = Path("/data/gpuopt.db")
    database_url: str = ""
    database_pool_min: int = 2
    database_pool_max: int = 10
    database_shards: dict[str, str] = {}
    log_level: str = "INFO"
    allow_mock_gpu: bool = True
    check_timeout_seconds: int = 15
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1
    cors_origins: list[str] = ["*"]
    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 120
    rate_limit_per_hour: int = 5000
    api_keyless_mode: bool = True
    default_admin_key: str = ""

    oauth2_token_url: str = ""
    oauth2_verify_url: str = ""
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""
    oidc_issuer_url: str = ""
    oidc_client_id: str = ""
    ldap_server_url: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    ldap_search_base: str = ""
    ldap_search_filter: str = "(uid={username})"

    healing_monitor_interval: int = 60

    gatekeeper_api_url: str = ""
    gatekeeper_enabled: bool = False

    agent_heartbeat_timeout: int = 180
    agent_stale_check_interval: int = 60
    agent_mtls_enabled: bool = False
    agent_mtls_cert_file: str = ""
    agent_mtls_key_file: str = ""
    agent_mtls_ca_cert_file: str = ""
    agent_mtls_verify_client: bool = True

    slack_webhook_url: str = ""
    pagerduty_routing_key: str = ""
    opsgenie_api_key: str = ""
    deepseek_api_key: str = ""
    rtx_partitions_gb: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_addr: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GPUOPT_",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.database_path}"

    @property
    def is_postgres(self) -> bool:
        return self.resolved_database_url.startswith("postgres")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.is_postgres:
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
