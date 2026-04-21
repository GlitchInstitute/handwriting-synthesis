"""SQLite-backed persistent settings store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "settings.db"

DEFAULTS: dict[str, str | int | float] = {
    "style": 0,
    "bias": 0.75,
    "stroke_width": 2.0,
    "stroke_color": "black",
    "line_width": 60,
    "scale": 1.5,
    "num_versions": 1,
}


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    return db


def get(key: str) -> str | int | float:
    db = _conn()
    row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    db.close()
    if row is None:
        return DEFAULTS[key]
    return json.loads(row[0])


def get_all() -> dict[str, str | int | float]:
    db = _conn()
    rows = db.execute("SELECT key, value FROM settings").fetchall()
    db.close()
    result = dict(DEFAULTS)
    for k, v in rows:
        result[k] = json.loads(v)
    return result


def put(key: str, value: str | int | float) -> None:
    db = _conn()
    db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )
    db.commit()
    db.close()


def put_all(data: dict[str, str | int | float]) -> None:
    db = _conn()
    for k, v in data.items():
        db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (k, json.dumps(v)),
        )
    db.commit()
    db.close()
