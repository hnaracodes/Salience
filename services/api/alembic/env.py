"""Alembic environment configuration."""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# TribeV2 repo root — required for `from services.api.models import ...`
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_repo_dotenv() -> None:
    """Load TribeV2/.env when running Alembic locally (shell env takes precedence)."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    parsed: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        parsed[key.strip()] = val.strip().strip('"').strip("'")
    for key, val in parsed.items():
        if key not in os.environ:
            os.environ[key] = val


_load_repo_dotenv()

from services.api.models import Base

config = context.config

if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Allow DATABASE_URL to override the ini setting at runtime.
_db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
