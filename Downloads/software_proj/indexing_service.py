"""
Folder scanning service.

Scope for this iteration: walk a folder, find supported files, record
their metadata + content hash in SQLite. Deliberately does NOT extract
text, chunk, or embed anything yet — that's Iterations 5+. A file with
status='pending' here just means "known to exist, not yet processed."
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from db.repositories import FileRepository
from processors.file_utils import compute_content_hash, is_supported


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def scan_folder(folder_id: int, folder_path: str, file_repo: FileRepository) -> dict:
    """
    Recursively walks folder_path, upserts a `files` row for every
    supported file found. Returns counts for the caller to report back
    via the API — useful feedback for "I just added a folder, what
    happened?" without needing a separate status endpoint yet.
    """
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    skipped_count = 0

    root = Path(folder_path)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories (., .git, .cache, etc.) in place so
        # os.walk doesn't descend into them at all.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for name in filenames:
            if name.startswith("."):
                continue

            file_path = Path(dirpath) / name

            if not is_supported(file_path):
                skipped_count += 1
                continue

            try:
                stat = file_path.stat()
                content_hash = compute_content_hash(file_path)
            except OSError:
                # Unreadable file (permissions, broken symlink, disappeared
                # mid-scan). Skip it rather than aborting the whole scan.
                skipped_count += 1
                continue

            _, status = file_repo.upsert(
                folder_id=folder_id,
                filename=name,
                path=str(file_path),
                extension=file_path.suffix.lower(),
                size_bytes=stat.st_size,
                created_at=_iso(stat.st_ctime),
                modified_at=_iso(stat.st_mtime),
                content_hash=content_hash,
            )

            if status == "new":
                new_count += 1
            elif status == "updated":
                updated_count += 1
            else:
                unchanged_count += 1

    return {
        "new_files": new_count,
        "updated_files": updated_count,
        "unchanged_files": unchanged_count,
        "skipped_files": skipped_count,
    }
