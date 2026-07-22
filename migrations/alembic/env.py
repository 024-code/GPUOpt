"""Alembic environment configuration for GPUOpt.

Supports both SQLite (dev) and PostgreSQL (production) targets.
The active database URL is resolved from Settings when available.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from gpuopt.config import get_settings

settings = get_settings()
if settings.resolved_database_url:
    config.set_main_option("sqlalchemy.url", settings.resolved_database_url)

target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine

    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
