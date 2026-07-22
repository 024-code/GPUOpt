"""Database migration runner for GPUOpt sandbox.

Supports both Alembic (preferred) and legacy SQL-file migrations.

Usage:
    # Run pending migrations (auto-detects Alembic vs SQL)
    python scripts/migrate.py

    # Run with explicit Alembic
    python scripts/migrate.py --alembic

    # Run with explicit SQL file migration
    python scripts/migrate.py --sql [./data/gpuopt.db]

    # Run Alembic against a specific database URL
    python scripts/migrate.py --alembic postgresql://user:pass@host:5432/gpuopt

    # Stamp a fresh database at the current Alembic head (no-op upgrade)
    python scripts/migrate.py --stamp
"""

import argparse
import subprocess
import sys
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
ALEMBIC_CFG = Path(__file__).resolve().parent.parent / "alembic.ini"
TRACKING_TABLE = "_migrations"


def _run_alembic(args: list[str]) -> int:
    """Run Alembic with the given arguments."""
    cmd = [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_CFG)] + args
    result = subprocess.run(cmd, cwd=ALEMBIC_CFG.parent)
    return result.returncode


def _alembic_already_used() -> bool:
    """Check if Alembic has already been used (revision table exists)."""
    import sqlite3

    db_path = Path("./data/gpuopt.db")
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
        )
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False


def _sql_migrate(database_url: str) -> int:
    """Run SQL-file-based migrations (legacy path)."""
    if database_url.startswith("postgresql"):
        import psycopg2

        conn = psycopg2.connect(database_url)
        conn.autocommit = False
        dialect = "postgresql"
    else:
        import sqlite3

        conn = sqlite3.connect(database_url)
        dialect = "sqlite"

    try:
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not files:
            print("No SQL migration files found.")
            return 0

        cur = conn.cursor()
        if dialect == "postgresql":
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} ("
                "id SERIAL PRIMARY KEY,"
                "filename TEXT NOT NULL UNIQUE,"
                "applied_at TIMESTAMP NOT NULL DEFAULT NOW()"
                ")"
            )
        else:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "filename TEXT NOT NULL UNIQUE,"
                "applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            )
        conn.commit()

        table = TRACKING_TABLE
        cur.execute(f"SELECT filename FROM {table} ORDER BY id")
        applied = {row[0] for row in cur.fetchall()}
        pending = [f for f in files if f.name not in applied]

        if not pending:
            print("All SQL migrations applied.")
            return 0

        for migration in pending:
            print(f"Applying {migration.name} ...")
            sql = migration.read_text(encoding="utf-8")
            try:
                if dialect == "postgresql":
                    cur.execute(sql)
                else:
                    conn.executescript(sql)
            except Exception as exc:
                print(f"FAILED: {migration.name}: {exc}", file=sys.stderr)
                conn.rollback()
                return 1
            if dialect == "postgresql":
                cur.execute(f"INSERT INTO {table} (filename) VALUES (%s)", (migration.name,))
            else:
                cur.execute(f"INSERT INTO {table} (filename) VALUES (?)", (migration.name,))
            conn.commit()
            print(f"Applied {migration.name}")

        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="GPUOpt database migration runner")
    parser.add_argument(
        "--alembic",
        action="store_true",
        help="Use Alembic for migrations (preferred)",
    )
    parser.add_argument(
        "--sql",
        nargs="?",
        const="./data/gpuopt.db",
        default=None,
        help="Use legacy SQL file migrations (default: ./data/gpuopt.db)",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Stamp the database at the current Alembic head without running migrations",
    )
    parser.add_argument(
        "database_url",
        nargs="?",
        default=None,
        help="Database URL (default: from Settings or alembic.ini)",
    )
    args = parser.parse_args()

    # --stamp: stamp the database at head
    if args.stamp:
        url_args = ["stamp", "head"]
        return _run_alembic(url_args)

    # --sql: explicit SQL file migration
    if args.sql is not None:
        return _sql_migrate(args.database_url or args.sql)

    # --alembic or auto-detect: prefer Alembic
    use_alembic = args.alembic or _alembic_already_used()

    if use_alembic or not args.sql:
        url_args = ["upgrade", "head"]
        if args.database_url:
            # Override the DB URL via env var (alembic picks it up from env.py)
            import os

            os.environ["GPUOPT_DATABASE_URL"] = args.database_url
        print("Running Alembic migrations...")
        code = _run_alembic(url_args)
        if code == 0:
            print("Alembic migrations complete.")
        return code

    # Fallback to SQL migration
    return _sql_migrate(args.database_url or "./data/gpuopt.db")


if __name__ == "__main__":
    sys.exit(main())
