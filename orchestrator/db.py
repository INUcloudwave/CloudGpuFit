"""SQLite job store.

One file (jobs.db) next to this module. Every call opens and closes its own
short-lived connection — throughput here is bounded by a single xMem
subprocess at a time, so there is no concurrency to optimize for.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent / "jobs.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            error_code TEXT,
            error_message TEXT,
            cache_key TEXT NOT NULL,
            cached INTEGER NOT NULL DEFAULT 0,
            progress_message TEXT,
            slo_json TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_cache_key ON jobs (cache_key)")
    # Added after the initial release — existing jobs.db files won't have it yet.
    for column_sql in ("ALTER TABLE jobs ADD COLUMN progress_message TEXT", "ALTER TABLE jobs ADD COLUMN slo_json TEXT"):
        try:
            conn.execute(column_sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def insert_job(
    job_id: str,
    status: str,
    payload: dict,
    cache_key: str,
    *,
    cached: bool = False,
    result: Optional[dict] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO jobs (id, status, payload_json, result_json, error_code, error_message,
                           cache_key, cached, created_at, started_at, finished_at)
        VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            status,
            json.dumps(payload),
            json.dumps(result) if result is not None else None,
            cache_key,
            int(cached),
            now_iso(),
            started_at,
            finished_at,
        ),
    )
    conn.commit()
    conn.close()


def find_succeeded_by_cache_key(cache_key: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM jobs WHERE cache_key = ? AND status = 'succeeded' ORDER BY created_at DESC LIMIT 1",
        (cache_key,),
    ).fetchone()
    conn.close()
    return row


def get_job(job_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return row


def mark_running(job_id: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE jobs SET status = 'running', started_at = ?, progress_message = NULL WHERE id = ?",
        (now_iso(), job_id),
    )
    conn.commit()
    conn.close()


def set_progress(job_id: str, message: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE jobs SET progress_message = ? WHERE id = ?", (message, job_id))
    conn.commit()
    conn.close()


def set_slo(job_id: str, slo: dict) -> None:
    """Overwrites any previously stored SLO for this job — resubmitting is a
    correction, not a new job (a real workload/config change gets a new job
    via a fresh cache key, same as always; this is just the pass/fail bar)."""
    conn = get_conn()
    conn.execute("UPDATE jobs SET slo_json = ? WHERE id = ?", (json.dumps(slo), job_id))
    conn.commit()
    conn.close()


def mark_finished(
    job_id: str,
    status: str,
    *,
    result: Optional[dict] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        UPDATE jobs
        SET status = ?, result_json = ?, error_code = ?, error_message = ?,
            finished_at = ?, progress_message = NULL
        WHERE id = ?
        """,
        (
            status,
            json.dumps(result) if result is not None else None,
            error_code,
            error_message,
            now_iso(),
            job_id,
        ),
    )
    conn.commit()
    conn.close()


def job_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    error = (
        {"code": row["error_code"], "message": row["error_message"]}
        if row["error_code"]
        else None
    )
    return {
        "job_id": row["id"],
        "status": row["status"],
        "cached": bool(row["cached"]),
        "progress": row["progress_message"],
        "payload": json.loads(row["payload_json"]),
        "slo": json.loads(row["slo_json"]) if row["slo_json"] else None,
        "result": result,
        "error": error,
    }
