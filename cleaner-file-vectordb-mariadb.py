"""
Reset all runtime data for this RAG project.

This script clears:
1) MariaDB tables with TRUNCATE (auto-increment IDs reset)
2) Redis current DB (FLUSHDB)
3) LanceDB directory content
4) Uploaded physical files and trash directory content
5) LlamaIndex storage metadata files under ./storage

Usage:
  python cleaner-file-vectordb-mariadb.py --dry-run
  python cleaner-file-vectordb-mariadb.py --yes
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import redis
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


SKIP_TRUNCATE_TABLES = {"alembic_version"}
TARGET_TRUNCATE_TABLES = {"files"}
DEFAULT_STORAGE_FILES = {
	"docstore.json": "{}\n",
	"graph_store.json": "{}\n",
	"image__vector_store.json": "{}\n",
	"index_store.json": "{}\n",
}


def _echo(message: str) -> None:
	print(message, flush=True)


def _resolve_paths() -> dict[str, Path | str]:
	project_root = Path(__file__).resolve().parent

	database_url = os.getenv("DATABASE_URL", "")
	redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

	upload_root = (project_root / os.getenv("UPLOAD_ROOT", "file_storage/uploads")).resolve()
	configured_trash = (project_root / os.getenv("TRASH_DIR", "file_storage/trash")).resolve()
	default_trash = (project_root / "file_storage/trash").resolve()
	lancedb_dir = (project_root / os.getenv("LANCEDB_DIR", "lancedb")).resolve()
	storage_dir = (project_root / os.getenv("STORAGE_DIR", "storage")).resolve()

	trash_roots = [configured_trash]
	if default_trash != configured_trash:
		trash_roots.append(default_trash)

	return {
		"project_root": project_root,
		"database_url": database_url,
		"redis_url": redis_url,
		"upload_root": upload_root,
		"trash_roots": trash_roots,
		"lancedb_dir": lancedb_dir,
		"storage_dir": storage_dir,
	}


def _mask_url_secret(url: str) -> str:
	if not url or "@" not in url:
		return url

	parts = urlsplit(url)
	if not parts.netloc or "@" not in parts.netloc:
		return url

	auth, host = parts.netloc.rsplit("@", 1)
	if ":" in auth:
		username, _ = auth.split(":", 1)
		masked_auth = f"{username}:***"
	else:
		masked_auth = auth

	return urlunsplit((parts.scheme, f"{masked_auth}@{host}", parts.path, parts.query, parts.fragment))


def _clear_directory_contents(path: Path, dry_run: bool) -> int:
	if not path.exists():
		if dry_run:
			_echo(f"[DRY-RUN] create directory: {path}")
		else:
			path.mkdir(parents=True, exist_ok=True)
		return 0

	deleted = 0
	for child in path.iterdir():
		deleted += 1
		if dry_run:
			_echo(f"[DRY-RUN] delete: {child}")
			continue
		if child.is_dir():
			shutil.rmtree(child)
		else:
			child.unlink(missing_ok=True)
	return deleted


def _reset_storage_files(storage_dir: Path, dry_run: bool) -> None:
	if not storage_dir.exists():
		if dry_run:
			_echo(f"[DRY-RUN] create directory: {storage_dir}")
		else:
			storage_dir.mkdir(parents=True, exist_ok=True)

	removed_count = _clear_directory_contents(storage_dir, dry_run=dry_run)
	_echo(f"Storage directory cleared: {removed_count} item(s)")

	for file_name, content in DEFAULT_STORAGE_FILES.items():
		file_path = storage_dir / file_name
		if dry_run:
			_echo(f"[DRY-RUN] write empty file: {file_path}")
			continue
		file_path.write_text(content, encoding="utf-8")


def _wipe_lancedb(path: Path, dry_run: bool) -> None:
	if path.exists():
		if dry_run:
			_echo(f"[DRY-RUN] remove directory tree: {path}")
		else:
			shutil.rmtree(path)

	if dry_run:
		_echo(f"[DRY-RUN] create directory: {path}")
	else:
		path.mkdir(parents=True, exist_ok=True)


def _flush_redis(redis_url: str, dry_run: bool) -> None:
	client = redis.Redis.from_url(redis_url)
	size = client.dbsize()
	_echo(f"Redis keys before cleanup: {size}")
	if dry_run:
		_echo("[DRY-RUN] skip FLUSHDB")
		return
	client.flushdb()
	_echo("Redis FLUSHDB completed")


async def _truncate_target_tables(database_url: str, dry_run: bool) -> list[str]:
	if not database_url:
		raise ValueError("DATABASE_URL is empty. Please set it in .env or environment variables.")

	engine = create_async_engine(database_url, echo=False)
	truncated_tables: list[str] = []

	try:
		async with engine.begin() as conn:
			rs = await conn.execute(
				text(
					"""
					SELECT table_name
					FROM information_schema.tables
					WHERE table_schema = DATABASE()
					AND table_type = 'BASE TABLE'
					ORDER BY table_name
					"""
				)
			)
			table_names = [
				name
				for name in rs.scalars().all()
				if name not in SKIP_TRUNCATE_TABLES and name in TARGET_TRUNCATE_TABLES
			]

			if not table_names:
				_echo("No target tables found for truncate")
				return truncated_tables

			_echo(f"MariaDB target tables to truncate: {len(table_names)}")
			for table_name in table_names:
				_echo(f" - {table_name}")

			if dry_run:
				_echo("[DRY-RUN] skip TRUNCATE")
				return table_names

			await conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
			try:
				for table_name in table_names:
					await conn.execute(text(f"TRUNCATE TABLE `{table_name}`"))
					truncated_tables.append(table_name)
			finally:
				await conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

		return truncated_tables
	finally:
		await engine.dispose()


async def _run(dry_run: bool) -> None:
	env = _resolve_paths()

	_echo("=== Cleanup plan ===")
	_echo(f"MariaDB : {_mask_url_secret(str(env['database_url']))}")
	_echo(f"Redis   : {env['redis_url']}")
	_echo(f"LanceDB : {env['lancedb_dir']}")
	_echo(f"Uploads : {env['upload_root']}")
	for trash_path in env["trash_roots"]:
		_echo(f"Trash   : {trash_path}")
	_echo(f"Storage : {env['storage_dir']}")

	truncated = await _truncate_target_tables(str(env["database_url"]), dry_run=dry_run)
	_echo(f"MariaDB cleanup done (tables affected: {len(truncated)})")

	_flush_redis(str(env["redis_url"]), dry_run=dry_run)

	_wipe_lancedb(env["lancedb_dir"], dry_run=dry_run)
	_echo("LanceDB directory reset done")

	upload_deleted = _clear_directory_contents(env["upload_root"], dry_run=dry_run)
	_echo(f"Upload directory cleared: {upload_deleted} item(s)")

	total_trash_deleted = 0
	for trash_path in env["trash_roots"]:
		deleted = _clear_directory_contents(trash_path, dry_run=dry_run)
		total_trash_deleted += deleted
		_echo(f"Trash directory cleared ({trash_path}): {deleted} item(s)")
	_echo(f"Trash cleanup total: {total_trash_deleted} item(s)")

	_reset_storage_files(env["storage_dir"], dry_run=dry_run)
	_echo("Storage metadata reset done")

	_echo("=== Cleanup completed ===")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Clear MariaDB, Redis, LanceDB, uploaded files, and storage metadata"
	)
	parser.add_argument(
		"--yes",
		action="store_true",
		help="Confirm destructive cleanup execution",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview actions without deleting data",
	)
	return parser.parse_args()


def main() -> int:
	load_dotenv()
	args = _parse_args()

	if not args.dry_run and not args.yes:
		_echo("Refusing to run destructive cleanup without --yes")
		_echo("Try: python cleaner-file-vectordb-mariadb.py --dry-run")
		_echo("Then: python cleaner-file-vectordb-mariadb.py --yes")
		return 1

	try:
		asyncio.run(_run(dry_run=args.dry_run))
		return 0
	except Exception as exc:
		_echo(f"Cleanup failed: {exc}")
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
