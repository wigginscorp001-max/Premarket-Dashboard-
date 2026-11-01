"""Download cache so history is fetched once and re-used.

Uses DATABASE_URL (Supabase / any Postgres) when set; otherwise a local SQLite file
(backtester/data/cache.db). Each entry is a DataFrame stored as compressed Parquet.
"""
from __future__ import annotations

import datetime as dt
import io
import threading
from pathlib import Path

import pandas as pd
from sqlalchemy import (Column, DateTime, LargeBinary, MetaData, String, Table,
                        create_engine, delete, select)

from .config import secret

_meta = MetaData()
_cache = Table(
    "bt_cache", _meta,
    Column("k", String(300), primary_key=True),
    Column("fetched_at", DateTime, nullable=False),
    Column("payload", LargeBinary, nullable=False),
)

_engine = None
_lock = threading.Lock()
_mem: dict[str, tuple[pd.DataFrame, dt.datetime]] = {}
_MEM_MAX = 4000


def _get_engine():
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        url = secret("DATABASE_URL")
        if url:
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://"):]
            _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
        else:
            path = Path(__file__).resolve().parent.parent / "data" / "cache.db"
            path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{path}",
                                    connect_args={"check_same_thread": False, "timeout": 60})
        _meta.create_all(_engine)
        return _engine


def backend_name() -> str:
    return "Postgres (Supabase)" if secret("DATABASE_URL") else "Local SQLite"


def _to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(b))


def get(key: str) -> tuple[pd.DataFrame, dt.datetime] | None:
    hit = _mem.get(key)
    if hit is not None:
        return hit
    try:
        with _get_engine().connect() as c:
            row = c.execute(select(_cache.c.payload, _cache.c.fetched_at)
                            .where(_cache.c.k == key)).first()
    except Exception:
        return None
    if row is None:
        return None
    out = (_from_bytes(row[0]), row[1])
    _remember(key, out)
    return out


def put(key: str, df: pd.DataFrame) -> None:
    now = dt.datetime.utcnow()
    _remember(key, (df, now))
    try:
        payload = _to_bytes(df.reset_index(drop=True))
        with _get_engine().begin() as c:
            c.execute(delete(_cache).where(_cache.c.k == key))
            c.execute(_cache.insert().values(k=key, fetched_at=now, payload=payload))
    except Exception:
        pass   # cache failure must never break a backtest


def _remember(key, val):
    if len(_mem) > _MEM_MAX:
        _mem.clear()
    _mem[key] = val


def stats() -> dict:
    try:
        from sqlalchemy import func
        with _get_engine().connect() as c:
            n = c.execute(select(func.count()).select_from(_cache)).scalar()
        return {"entries": int(n or 0), "backend": backend_name()}
    except Exception as e:
        return {"entries": 0, "backend": backend_name(), "error": str(e)}


def clear(prefix: str = "") -> int:
    _mem.clear()
    with _get_engine().begin() as c:
        res = c.execute(delete(_cache).where(_cache.c.k.like(prefix + "%")))
        return res.rowcount or 0
