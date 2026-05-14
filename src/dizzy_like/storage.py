"""SQLite-backed storage for crawled pages and analysis results."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS domains (
    domain              TEXT PRIMARY KEY,
    first_seen          REAL NOT NULL,
    last_checked        REAL,
    is_v3               INTEGER,
    is_available        INTEGER,
    server              TEXT,
    has_tls             INTEGER,
    -- classifiers
    language            TEXT,
    language_conf       REAL,
    is_illicit          INTEGER,
    illicit_conf        REAL,
    category            TEXT,
    category_conf       REAL,
    template_group      TEXT,
    template_conf       REAL,
    has_tracking        INTEGER,
    tracking_conf       REAL,
    has_attribution     INTEGER,
    attribution_conf    REAL,
    num_crypto_addrs    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    domain          TEXT NOT NULL,
    status_code     INTEGER,
    crawled_at      REAL NOT NULL,
    content_hash    TEXT,
    num_links       INTEGER DEFAULT 0,
    FOREIGN KEY(domain) REFERENCES domains(domain)
);

CREATE TABLE IF NOT EXISTS crypto_addresses (
    address         TEXT PRIMARY KEY,
    currency        TEXT NOT NULL,
    domain          TEXT NOT NULL,
    self_attributed INTEGER DEFAULT 0,
    context_tags    TEXT DEFAULT '[]',
    wallet_id       TEXT,
    num_txns        INTEGER DEFAULT 0,
    deposits_usd    REAL DEFAULT 0.0,
    withdrawals_usd REAL DEFAULT 0.0,
    is_outlier      INTEGER DEFAULT 0,
    is_malicious    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS graph_edges (
    src     TEXT NOT NULL,
    dst     TEXT NOT NULL,
    weight  INTEGER DEFAULT 1,
    PRIMARY KEY (src, dst)
);

CREATE INDEX IF NOT EXISTS idx_pages_domain ON pages(domain);
CREATE INDEX IF NOT EXISTS idx_addrs_domain ON crypto_addresses(domain);
CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src);
"""


class DizzyStorage:
    def __init__(self, db_path: str = "dizzy.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── Domains ───────────────────────────────────────────────────────────────

    def upsert_domain(self, domain: str, **kwargs: Any) -> None:
        now = time.time()
        existing = self.conn.execute(
            "SELECT 1 FROM domains WHERE domain = ?", (domain,)
        ).fetchone()

        if existing:
            if not kwargs:
                return
            setters = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [now, domain]
            self.conn.execute(
                f"UPDATE domains SET {setters}, last_checked = ? WHERE domain = ?", vals
            )
        else:
            cols = ["domain", "first_seen", "last_checked"] + list(kwargs.keys())
            placeholders = ", ".join("?" * len(cols))
            vals = [domain, now, now] + list(kwargs.values())
            self.conn.execute(
                f"INSERT OR IGNORE INTO domains ({', '.join(cols)}) VALUES ({placeholders})", vals
            )
        self.conn.commit()

    def get_domain(self, domain: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM domains WHERE domain = ?", (domain,)).fetchone()
        return dict(row) if row else None

    def get_all_domains(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM domains").fetchall()]

    # ── Pages ─────────────────────────────────────────────────────────────────

    def upsert_page(
        self, url: str, domain: str, status_code: int, content_hash: str, num_links: int
    ) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO pages (url, domain, status_code, crawled_at, content_hash, num_links)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (url, domain, status_code, time.time(), content_hash, num_links))
        self.conn.commit()

    def get_pages_for_domain(self, domain: str) -> List[Dict]:
        return [
            dict(r) for r in self.conn.execute(
                "SELECT * FROM pages WHERE domain = ?", (domain,)
            ).fetchall()
        ]

    # ── Crypto addresses ──────────────────────────────────────────────────────

    def upsert_address(
        self,
        address: str,
        currency: str,
        domain: str,
        self_attributed: bool,
        context_tags: List[str],
    ) -> None:
        self.conn.execute("""
            INSERT OR IGNORE INTO crypto_addresses
                (address, currency, domain, self_attributed, context_tags)
            VALUES (?, ?, ?, ?, ?)
        """, (address, currency, domain, int(self_attributed), json.dumps(context_tags)))
        self.conn.commit()

    def update_address_wallet(
        self,
        address: str,
        wallet_id: str,
        num_txns: int = 0,
        deposits_usd: float = 0.0,
        withdrawals_usd: float = 0.0,
        is_outlier: bool = False,
    ) -> None:
        self.conn.execute("""
            UPDATE crypto_addresses
            SET wallet_id = ?, num_txns = ?, deposits_usd = ?,
                withdrawals_usd = ?, is_outlier = ?
            WHERE address = ?
        """, (wallet_id, num_txns, deposits_usd, withdrawals_usd, int(is_outlier), address))
        self.conn.commit()

    def get_addresses_for_domain(self, domain: str) -> List[Dict]:
        return [
            dict(r) for r in self.conn.execute(
                "SELECT * FROM crypto_addresses WHERE domain = ?", (domain,)
            ).fetchall()
        ]

    def get_all_addresses(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM crypto_addresses").fetchall()]

    # ── Graph edges ───────────────────────────────────────────────────────────

    def upsert_edge(self, src: str, dst: str) -> None:
        self.conn.execute("""
            INSERT INTO graph_edges (src, dst, weight) VALUES (?, ?, 1)
            ON CONFLICT(src, dst) DO UPDATE SET weight = weight + 1
        """, (src, dst))
        self.conn.commit()

    def get_all_edges(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM graph_edges").fetchall()]

    # ── Reporting helpers ─────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        def scalar(sql: str) -> Any:
            return self.conn.execute(sql).fetchone()[0]

        return {
            "domains": scalar("SELECT COUNT(*) FROM domains"),
            "available": scalar("SELECT COUNT(*) FROM domains WHERE is_available = 1"),
            "illicit": scalar("SELECT COUNT(*) FROM domains WHERE is_illicit = 1"),
            "tracked": scalar("SELECT COUNT(*) FROM domains WHERE has_tracking = 1"),
            "attributed": scalar("SELECT COUNT(*) FROM domains WHERE has_attribution = 1"),
            "pages": scalar("SELECT COUNT(*) FROM pages"),
            "addresses": scalar("SELECT COUNT(*) FROM crypto_addresses"),
            "edges": scalar("SELECT COUNT(*) FROM graph_edges"),
        }

    def close(self) -> None:
        self.conn.close()
