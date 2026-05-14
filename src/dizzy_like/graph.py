"""Web graph construction and analytics (§3.2.3).

Builds a directed graph where:
  - type-1 nodes = onion domains
  - type-2 nodes = regular web domains
  - edges        = hyperlinks (URL references)

Runs four analytical tasks from the paper:
  1. Summary statistics (nodes, edges, SCC, clustering)
  2. Bow-tie decomposition
  3. Centrality measures (on type-1 subgraph)
  4. Dark-to-regular web linking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


@dataclass
class GraphStats:
    num_nodes: int = 0
    num_edges: int = 0
    num_onion_nodes: int = 0
    num_web_nodes: int = 0
    num_sccs: int = 0
    largest_scc_nodes: int = 0
    largest_scc_edges: int = 0
    avg_clustering_onion: float = 0.0
    # Bow-tie
    core_size: int = 0
    in_size: int = 0
    out_size: int = 0
    tendrils_size: int = 0
    # Connectivity
    dark_to_regular_links: int = 0
    malicious_regular_domains: int = 0


@dataclass
class NodeStats:
    domain: str
    is_onion: bool
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    pagerank: float = 0.0
    in_scc: bool = False
    in_bow_tie_core: bool = False


class OnionWebGraph:
    """Directed graph of onion and regular web domains linked by URLs."""

    def __init__(self):
        if not _HAS_NX:
            raise ImportError("networkx is required: pip install networkx")
        self.g: nx.DiGraph = nx.DiGraph()

    def _is_onion(self, domain: str) -> bool:
        return domain.lower().endswith(".onion")

    def add_domain(self, domain: str) -> None:
        if domain not in self.g:
            self.g.add_node(domain, is_onion=self._is_onion(domain))

    def add_link(self, src: str, dst: str, url: str = "") -> None:
        self.add_domain(src)
        self.add_domain(dst)
        if self.g.has_edge(src, dst):
            self.g[src][dst]["weight"] += 1
        else:
            self.g.add_edge(src, dst, weight=1, sample_url=url)

    def ingest_crawl_results(self, domain: str, links: List[str]) -> None:
        """Add all outgoing links discovered while crawling domain."""
        self.add_domain(domain)
        for url in links:
            dst = urlparse(url).hostname
            if dst:
                self.add_link(domain, dst.lower(), url)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def compute_stats(self) -> GraphStats:
        g = self.g
        stats = GraphStats()
        stats.num_nodes = g.number_of_nodes()
        stats.num_edges = g.number_of_edges()
        stats.num_onion_nodes = sum(1 for _, d in g.nodes(data=True) if d.get("is_onion"))
        stats.num_web_nodes = stats.num_nodes - stats.num_onion_nodes

        # Strongly connected components
        sccs = list(nx.strongly_connected_components(g))
        stats.num_sccs = len(sccs)
        if sccs:
            lscc = max(sccs, key=len)
            stats.largest_scc_nodes = len(lscc)
            stats.largest_scc_edges = g.subgraph(lscc).number_of_edges()

        # Clustering on onion-only undirected subgraph
        onion_nodes = [n for n, d in g.nodes(data=True) if d.get("is_onion")]
        if len(onion_nodes) >= 2:
            ug = g.subgraph(onion_nodes).to_undirected()
            stats.avg_clustering_onion = nx.average_clustering(ug)

        # Bow-tie decomposition
        core, in_c, out_c, tendrils = self._bow_tie()
        stats.core_size = len(core)
        stats.in_size = len(in_c)
        stats.out_size = len(out_c)
        stats.tendrils_size = len(tendrils)

        # Dark-to-regular links
        stats.dark_to_regular_links = len(self.dark_to_regular_links())

        return stats

    def _bow_tie(self) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        """
        Bow-tie decomposition of the full graph:
          core     = LSCC
          in       = nodes that can reach core but are not in core
          out      = nodes reachable from core but not in core
          tendrils = everything else
        """
        g = self.g
        if g.number_of_nodes() == 0:
            return set(), set(), set(), set()

        sccs = list(nx.strongly_connected_components(g))
        if not sccs:
            return set(), set(), set(), set()

        core: Set[str] = max(sccs, key=len)

        # Out-component: reachable from any core node
        out_reach: Set[str] = set()
        for n in core:
            out_reach.update(nx.descendants(g, n))
        out_c = out_reach - core

        # In-component: can reach any core node (use reversed graph)
        rg = g.reverse(copy=False)
        in_reach: Set[str] = set()
        for n in core:
            in_reach.update(nx.descendants(rg, n))
        in_c = in_reach - core

        tendrils = set(g.nodes()) - core - in_c - out_c
        return core, in_c, out_c, tendrils

    def node_stats(self, top_n: int = 100) -> List[NodeStats]:
        """Return NodeStats for the top_n nodes by total degree."""
        g = self.g
        core, _, _, _ = self._bow_tie()

        try:
            pr = nx.pagerank(g, max_iter=200)
        except Exception:
            pr = {n: 0.0 for n in g.nodes()}

        stats_list: List[NodeStats] = []
        for node, data in g.nodes(data=True):
            ns = NodeStats(
                domain=node,
                is_onion=data.get("is_onion", False),
                in_degree=g.in_degree(node),
                out_degree=g.out_degree(node),
                total_degree=g.degree(node),
                pagerank=pr.get(node, 0.0),
                in_bow_tie_core=node in core,
            )
            stats_list.append(ns)

        stats_list.sort(key=lambda s: -s.total_degree)
        return stats_list[:top_n]

    def dark_to_regular_links(self) -> List[Tuple[str, str]]:
        """Edges from onion (type-1) to clearweb (type-2) domains."""
        return [
            (src, dst)
            for src, dst in self.g.edges()
            if self.g.nodes[src].get("is_onion") and not self.g.nodes[dst].get("is_onion")
        ]

    def degree_distribution(self) -> Dict[int, int]:
        """Total-degree frequency distribution (for Figure 9 equivalent)."""
        from collections import Counter
        onion_nodes = [n for n, d in self.g.nodes(data=True) if d.get("is_onion")]
        degrees = [self.g.degree(n) for n in onion_nodes]
        return dict(Counter(degrees))

    def to_dict(self) -> dict:
        """Serialise graph to a JSON-friendly dict."""
        return {
            "nodes": [
                {"id": n, "is_onion": d.get("is_onion", False)}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": [
                {"src": u, "dst": v, "weight": d.get("weight", 1)}
                for u, v, d in self.g.edges(data=True)
            ],
        }
