"""SQLite storage for browser-scoped Copilot conversations."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4


class ChatStore:
    def __init__(self, path: Path, owner: str, source: str):
        self.path = path
        self.owner = owner
        self.source = source
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chats_owner_source ON chats(owner, source);
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    table_json TEXT,
                    note TEXT,
                    period TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS turns_chat ON turns(chat_id, id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def chats(self):
        with self.connect() as db:
            return [
                dict(row) for row in db.execute("""
                    SELECT id, title FROM chats
                    WHERE owner = ? AND source = ?
                    ORDER BY COALESCE((SELECT MAX(id) FROM turns WHERE chat_id = chats.id), 0) DESC
                """, (self.owner, self.source))
            ]

    def turns(self, chat_id):
        with self.connect() as db:
            return [
                dict(row) for row in db.execute("""
                    SELECT t.question, t.answer, t.sql, t.table_json, t.note, t.period, t.created_at
                    FROM turns t JOIN chats c ON t.chat_id = c.id
                    WHERE c.id = ? AND c.owner = ? AND c.source = ?
                    ORDER BY t.id
                """, (chat_id, self.owner, self.source))
            ]

    def save_turn(self, chat_id, question, response, period):
        table_json = None
        if not response.data.empty:
            table_json = response.data.head(50).to_json(orient="split", index=False, date_format="iso")
        with self.connect() as db:
            if chat_id is None:
                chat_id = uuid4().hex
                title = " ".join(question.split())
                if len(title) > 58:
                    title = title[:55].rstrip() + "…"
                db.execute(
                    "INSERT INTO chats (id, owner, source, title) VALUES (?, ?, ?, ?)",
                    (chat_id, self.owner, self.source, title),
                )
            cursor = db.execute("""
                INSERT INTO turns (chat_id, question, answer, sql, table_json, note, period)
                SELECT id, ?, ?, ?, ?, ?, ? FROM chats WHERE id = ? AND owner = ? AND source = ?
            """, (question, response.answer, response.sql, table_json, response.details.get("note"),
                  period, chat_id, self.owner, self.source))
            if cursor.rowcount != 1:
                raise ValueError("The selected chat is unavailable.")
        return chat_id
