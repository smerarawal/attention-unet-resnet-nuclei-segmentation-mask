"""
Repository layer: every SQL statement in the app lives here.

Why bother with this layer for something as simple as SQLite: once
search/RAG services need to query files and chunks, they should call
`FileRepository.get(id)` and not care whether that's backed by SQLite,
a different DB, or a cache later. It also means there's exactly one
place to fix if a query is wrong, instead of the same SELECT duplicated
across api/services files.
"""

import sqlite3
from typing import Optional


class FolderRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, path: str) -> sqlite3.Row:
        cur = self.conn.execute(
            "INSERT INTO folders (path) VALUES (?)", (path,)
        )
        self.conn.commit()
        return self.get(cur.lastrowid)

    def get(self, folder_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM folders WHERE folder_id = ?", (folder_id,)
        ).fetchone()

    def get_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM folders WHERE path = ?", (path,)
        ).fetchone()

    def list(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM folders ORDER BY created_at"
        ).fetchall()

    def delete(self, folder_id: int) -> None:
        # ON DELETE CASCADE on files.folder_id / chunks.file_id handles
        # cleanup of files+chunks rows. It does NOT touch FAISS (no FAISS
        # yet) — that wiring has to happen explicitly once FAISS exists,
        # cascade alone won't be enough then.
        self.conn.execute("DELETE FROM folders WHERE folder_id = ?", (folder_id,))
        self.conn.commit()


class FileRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()

    def get(self, file_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()

    def list(self, folder_id: Optional[int] = None) -> list[sqlite3.Row]:
        if folder_id is not None:
            return self.conn.execute(
                "SELECT * FROM files WHERE folder_id = ? ORDER BY filename",
                (folder_id,),
            ).fetchall()
        return self.conn.execute("SELECT * FROM files ORDER BY filename").fetchall()

    def upsert(
        self,
        folder_id: int,
        filename: str,
        path: str,
        extension: str,
        size_bytes: int,
        created_at: Optional[str],
        modified_at: Optional[str],
        content_hash: str,
    ) -> tuple[sqlite3.Row, str]:
        """
        Insert a new file row, or update an existing one IF its content
        hash changed. Returns (row, status) where status is one of
        "new" / "updated" / "unchanged" — explicit, so the caller
        (scanner) doesn't have to re-derive which case happened.

        Unchanged files are left alone entirely — status/indexed_at stay
        whatever they were, since nothing downstream needs to redo work.
        """
        existing = self.get_by_path(path)

        if existing is None:
            cur = self.conn.execute(
                """INSERT INTO files
                   (folder_id, filename, path, extension, size_bytes,
                    created_at, modified_at, content_hash, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (folder_id, filename, path, extension, size_bytes,
                 created_at, modified_at, content_hash),
            )
            self.conn.commit()
            return self.get(cur.lastrowid), "new"

        if existing["content_hash"] == content_hash:
            return existing, "unchanged"

        # Hash changed: reset to 'pending' so the (future) extraction
        # step knows to redo this file. Chunk cleanup for the old
        # version happens in the extraction step, not here — this
        # repository only owns the `files` table.
        self.conn.execute(
            """UPDATE files
               SET size_bytes = ?, modified_at = ?, content_hash = ?, status = 'pending'
               WHERE file_id = ?""",
            (size_bytes, modified_at, content_hash, existing["file_id"]),
        )
        self.conn.commit()
        return self.get(existing["file_id"]), "updated"

    def mark_deleted(self, file_id: int) -> None:
        self.conn.execute(
            "UPDATE files SET status = 'deleted' WHERE file_id = ?", (file_id,)
        )
        self.conn.commit()
