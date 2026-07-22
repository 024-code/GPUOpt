from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.migrate import _sql_migrate


@pytest.fixture()
def tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestSQLMigration:
    def test_sql_migrate_creates_tracking_table(self, tmp_db: str):
        code = _sql_migrate(tmp_db)
        assert code == 0
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_sql_migrate_applies_pending(self, tmp_db: str):
        code = _sql_migrate(tmp_db)
        assert code == 0
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        # Should have created the clusters table (from 001_initial_schema.sql)
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
        )
        assert cur.fetchone() is not None
        # Should have created actuations table (from 008_actuations.sql)
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='actuations'"
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_sql_migrate_idempotent(self, tmp_db: str):
        code1 = _sql_migrate(tmp_db)
        code2 = _sql_migrate(tmp_db)
        assert code1 == 0
        assert code2 == 0

    def test_sql_migrate_tracks_applied(self, tmp_db: str):
        _sql_migrate(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute("SELECT filename FROM _migrations ORDER BY id")
        applied = [row[0] for row in cur.fetchall()]
        conn.close()
        # Should have at least 001 through 008 (minus comments-only 006)
        assert len(applied) >= 5
        assert "001_initial_schema.sql" in applied
        assert "008_actuations.sql" in applied

    def test_sql_migrate_no_files(self, tmp_db: str, monkeypatch: pytest.MonkeyPatch):
        with tempfile.TemporaryDirectory() as tmp:
            from scripts import migrate as migrate_mod
            old = migrate_mod.MIGRATIONS_DIR
            migrate_mod.MIGRATIONS_DIR = Path(tmp)
            code = _sql_migrate(tmp_db)
            migrate_mod.MIGRATIONS_DIR = old
        assert code == 0


class TestAlembicMigration:
    def test_alembic_upgrade_head(self):
        from alembic.config import Config
        from alembic import command
        from gpuopt.config import get_settings

        get_settings.cache_clear()
        db_path = tempfile.mktemp(suffix=".db")
        os.environ["GPUOPT_DATABASE_PATH"] = db_path
        os.environ.pop("GPUOPT_DATABASE_URL", None)

        try:
            cfg = Config("alembic.ini")
            command.upgrade(cfg, "head")

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
            )
            assert cur.fetchone() is not None
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
            )
            row = cur.fetchone()
            assert row is not None
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            os.environ.pop("GPUOPT_DATABASE_PATH", None)

    def test_alembic_upgrade_idempotent(self):
        from alembic.config import Config
        from alembic import command
        from gpuopt.config import get_settings

        get_settings.cache_clear()
        db_path = tempfile.mktemp(suffix=".db")
        os.environ["GPUOPT_DATABASE_PATH"] = db_path

        try:
            cfg = Config("alembic.ini")
            command.upgrade(cfg, "head")
            command.upgrade(cfg, "head")  # second run should be a no-op

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT version_num FROM alembic_version")
            version = cur.fetchone()
            assert version is not None
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            os.environ.pop("GPUOPT_DATABASE_PATH", None)

    def test_alembic_stamp(self):
        from alembic.config import Config
        from alembic import command
        from gpuopt.config import get_settings

        get_settings.cache_clear()
        db_path = tempfile.mktemp(suffix=".db")
        os.environ["GPUOPT_DATABASE_PATH"] = db_path

        try:
            cfg = Config("alembic.ini")
            command.stamp(cfg, "head")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            os.environ.pop("GPUOPT_DATABASE_PATH", None)

    def test_alembic_history(self):
        from alembic.config import Config
        from alembic import command
        from gpuopt.config import get_settings

        get_settings.cache_clear()
        db_path = tempfile.mktemp(suffix=".db")
        os.environ["GPUOPT_DATABASE_PATH"] = db_path

        try:
            cfg = Config("alembic.ini")
            command.history(cfg)
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            os.environ.pop("GPUOPT_DATABASE_PATH", None)


class TestMigrationScript:
    def test_migrate_script_sql_flag(self, tmp_db: str):
        from scripts.migrate import main as migrate_main
        import sys

        old_argv = sys.argv
        sys.argv = ["migrate.py", "--sql", tmp_db]
        try:
            code = migrate_main()
            assert code == 0
        finally:
            sys.argv = old_argv

    def test_migrate_script_stamp_flag(self, tmp_db: str):
        from scripts.migrate import main as migrate_main
        from gpuopt.config import get_settings
        import sys

        get_settings.cache_clear()
        os.environ["GPUOPT_DATABASE_PATH"] = tmp_db
        old_argv = sys.argv
        sys.argv = ["migrate.py", "--stamp"]
        try:
            code = migrate_main()
            assert code == 0
        finally:
            sys.argv = old_argv
            os.environ.pop("GPUOPT_DATABASE_PATH", None)
