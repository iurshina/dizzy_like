"""Cryptocurrency analytics (§3.2.2).

  - Extract Bitcoin / Ethereum / Monero addresses from HTML
  - Cluster addresses into wallets via union-find (common-input-ownership)
  - Basic wallet stats: size, volume, deposit/withdrawal totals
  - Outlier wallet detection via Isolation Forest
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

_BTC_LEGACY = re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b')
_BTC_BECH32 = re.compile(r'\bbc1[a-zA-HJ-NP-Z0-9]{25,39}\b')
_ETH = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
_XMR = re.compile(r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b')

_ADDR_PATTERNS = [
    (_BTC_LEGACY, "bitcoin"),
    (_BTC_BECH32, "bitcoin"),
    (_ETH, "ethereum"),
    (_XMR, "monero"),
]


@dataclass
class RawAddress:
    address: str
    currency: str
    domain: str
    self_attributed: bool = False
    context_snippet: str = ""


@dataclass
class Transaction:
    txid: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    value_usd: float = 0.0
    is_deposit: bool = True


@dataclass
class Wallet:
    wallet_id: str
    addresses: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    num_transactions: int = 0
    num_deposits: int = 0
    num_withdrawals: int = 0
    total_deposits_usd: float = 0.0
    total_withdrawals_usd: float = 0.0
    is_outlier: bool = False

    @property
    def size(self) -> int:
        return len(self.addresses)

    @property
    def balance_usd(self) -> float:
        return self.total_deposits_usd - self.total_withdrawals_usd


def extract_addresses(html: str, domain: str) -> List[RawAddress]:
    """Extract all crypto addresses from raw HTML with attribution context."""
    found: List[RawAddress] = []
    seen: Set[str] = set()

    for pattern, currency in _ADDR_PATTERNS:
        for m in pattern.finditer(html):
            addr = m.group(0)
            if addr in seen:
                continue
            seen.add(addr)

            start = m.start()
            ctx = html[max(0, start - 300): start + len(addr) + 300].lower()

            self_attributed = (
                bool(re.search(r'\b(donat|our wallet|our address|payment|accept|tip)\b', ctx))
                and not bool(re.search(r'\bexample\b', ctx))
            )

            found.append(RawAddress(
                address=addr,
                currency=currency,
                domain=domain,
                self_attributed=self_attributed,
                context_snippet=ctx[:200],
            ))

    return found


class WalletClusterer:
    """
    Groups addresses into wallets using common-input-ownership heuristic.
    Each call to add_transaction_inputs() applies the multi-input heuristic:
    all inputs of a transaction are assumed to be controlled by the same entity.
    """

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}
        self._domains: Dict[str, Set[str]] = defaultdict(set)

    def _find(self, x: str) -> str:
        root = x
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        # Path compression
        while self._parent.get(x, x) != root:
            self._parent[x], x = root, self._parent.get(x, x)
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self._rank.get(ra, 0) < self._rank.get(rb, 0):
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank.get(ra, 0) == self._rank.get(rb, 0):
            self._rank[ra] = self._rank.get(ra, 0) + 1

    def add_address(self, address: str, domain: str = "") -> None:
        self._find(address)  # ensure exists
        if domain:
            self._domains[self._find(address)].add(domain)

    def add_transaction_inputs(self, input_addresses: List[str]) -> None:
        """Multi-input heuristic: co-spending addresses share a wallet."""
        for addr in input_addresses:
            self._find(addr)
        for i in range(1, len(input_addresses)):
            self._union(input_addresses[0], input_addresses[i])

    def get_wallets(self, transactions: Optional[List[Transaction]] = None) -> Dict[str, Wallet]:
        """Return mapping: wallet_id -> Wallet with aggregated stats."""
        groups: Dict[str, Set[str]] = defaultdict(set)
        for addr in self._parent:
            root = self._find(addr)
            groups[root].add(addr)
        # Also include addresses only seen via add_address
        for addr in self._domains:
            root = self._find(addr)
            groups[root].add(addr)

        wallets: Dict[str, Wallet] = {}
        for wid, addrs in groups.items():
            w = Wallet(wallet_id=wid, addresses=set(addrs))
            for addr in addrs:
                w.domains.update(self._domains.get(addr, set()))
            wallets[wid] = w

        if transactions:
            addr_to_wallet = {addr: self._find(addr) for addrs in groups.values() for addr in addrs}
            for tx in transactions:
                for addr in tx.inputs + tx.outputs:
                    wid = addr_to_wallet.get(addr)
                    if wid and wid in wallets:
                        w = wallets[wid]
                        w.num_transactions += 1
                        if tx.is_deposit:
                            w.num_deposits += 1
                            w.total_deposits_usd += tx.value_usd
                        else:
                            w.num_withdrawals += 1
                            w.total_withdrawals_usd += tx.value_usd

        return wallets

    def filter_outliers(self, wallets: Dict[str, Wallet]) -> Dict[str, Wallet]:
        """
        Mark wallets with significantly larger size or money flow as outliers
        using Isolation Forest (as described in §3.2.2).
        Requires scikit-learn.
        """
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np
        except ImportError:
            return wallets

        if len(wallets) < 10:
            return wallets

        wids = list(wallets.keys())
        X = np.array([
            [wallets[w].size, wallets[w].total_deposits_usd, wallets[w].num_transactions]
            for w in wids
        ], dtype=float)

        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X)

        for wid, pred in zip(wids, preds):
            wallets[wid].is_outlier = (pred == -1)

        return wallets
