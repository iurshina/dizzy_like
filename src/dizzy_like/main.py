"""
Dizzy — Large-Scale Crawling and Analysis of Onion Services
Python implementation of Boshmaf et al. (2023)

Usage:
  python -m dizzy_like crawl --seeds seeds.txt
  python -m dizzy_like check
  python -m dizzy_like report
  python -m dizzy_like graph
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .classifiers import DizzyClassifiers
from .crawler import OnionCrawler
from .crypto import WalletClusterer, extract_addresses
from .features import extract_features
from .graph import OnionWebGraph
from .seeder import INDEXER_URLS, collect_seeds
from .storage import DizzyStorage


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


# ── seed ──────────────────────────────────────────────────────────────────────

def cmd_seed(args: argparse.Namespace) -> None:
    extra_words = []
    if args.words:
        words_path = Path(args.words)
        if not words_path.exists():
            print(f"[!] Word list not found: {args.words}", file=sys.stderr)
            sys.exit(1)
        extra_words = [w.strip() for w in words_path.read_text().splitlines() if w.strip()]

    extra_indexers = [u.strip() for u in args.indexers.split(",") if u.strip()] if args.indexers else []

    collect_seeds(
        tor_proxy=args.tor_proxy,
        timeout=args.timeout,
        delay=args.delay,
        use_indexers=not args.no_indexers,
        use_torch=not args.no_torch,
        extra_indexers=extra_indexers,
        extra_words=extra_words,
        max_torch_queries=args.max_queries,
        output=args.output,
    )


# ── crawl ─────────────────────────────────────────────────────────────────────

def cmd_crawl(args: argparse.Namespace) -> None:
    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        print(f"[!] Seeds file not found: {args.seeds}", file=sys.stderr)
        sys.exit(1)

    seeds = [
        line.strip() for line in seeds_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not seeds:
        print("[!] Seeds file is empty.", file=sys.stderr)
        sys.exit(1)

    storage = DizzyStorage(args.db)
    graph = OnionWebGraph()
    classifiers = DizzyClassifiers()

    if args.model_dir and Path(args.model_dir).exists():
        print(f"[*] Loading classifiers from {args.model_dir}")
        classifiers.load(args.model_dir)

    crawler = OnionCrawler(
        tor_proxy=args.tor_proxy,
        timeout=args.timeout,
        delay=args.delay,
        max_pages_per_domain=args.max_pages,
    )

    mode = args.mode
    print(f"[*] Mode: {mode}  |  Seeds: {len(seeds)}  |  DB: {args.db}")

    if mode == "explore":
        domain_results = crawler.explore(seeds)
    else:
        domain_results = {
            (_hostname(s) if s.startswith("http") else s): crawler.crawl_domain(
                s if s.startswith("http") else f"http://{s}"
            )
            for s in seeds
        }

    for domain, results in domain_results.items():
        available = any(r.ok for r in results)
        server = next((r.server for r in results if r.server), "")
        has_tls = any(r.has_tls for r in results if r.ok)
        is_v3 = len(domain.split(".")[0]) == 56  # v3 = 56-char base32

        storage.upsert_domain(
            domain,
            is_available=int(available),
            server=server,
            has_tls=int(has_tls),
            is_v3=int(is_v3),
        )

        full_html = ""
        homepage_result = None

        for result in results:
            if not result.ok:
                continue
            full_html += result.html
            content_hash = hashlib.sha256(result.html.encode()).hexdigest()
            storage.upsert_page(
                url=result.url,
                domain=result.domain or domain,
                status_code=result.status_code or 0,
                content_hash=content_hash,
                num_links=len(result.links),
            )
            # Graph edges
            for link in result.links:
                dst = _hostname(link)
                if dst:
                    graph.add_link(domain, dst, link)
                    storage.upsert_edge(domain, dst)

            if homepage_result is None:
                homepage_result = result

        # Feature extraction & classification on homepage
        if homepage_result and homepage_result.html:
            features = extract_features(homepage_result.html, homepage_result.url)

            soup = BeautifulSoup(homepage_result.html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            clean_text = soup.get_text(" ", strip=True)

            clf_result = classifiers.classify(clean_text, features)

            storage.upsert_domain(
                domain,
                language=clf_result.language,
                language_conf=clf_result.language_confidence,
                is_illicit=int(clf_result.is_illicit),
                illicit_conf=clf_result.illicit_confidence,
                category=clf_result.category,
                category_conf=clf_result.category_confidence,
                template_group=clf_result.template_group,
                template_conf=clf_result.template_confidence,
                has_tracking=int(clf_result.has_tracking),
                tracking_conf=clf_result.tracking_confidence,
                has_attribution=int(clf_result.has_attribution),
                attribution_conf=clf_result.attribution_confidence,
                num_crypto_addrs=features.num_crypto_addresses,
            )

            # Crypto address extraction
            for addr in extract_addresses(full_html, domain):
                tags = addr.context_snippet[:50].split() if addr.context_snippet else []
                storage.upsert_address(
                    address=addr.address,
                    currency=addr.currency,
                    domain=addr.domain,
                    self_attributed=addr.self_attributed,
                    context_tags=tags,
                )

        print(
            f"  [{'+' if available else '-'}] {domain}"
            f"  pages={len(results)}"
            f"  cat={clf_result.category if homepage_result and homepage_result.html else '?'}"
        )

    print(f"\n[+] Crawl complete. Results saved to {args.db}")


# ── check ─────────────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> None:
    storage = DizzyStorage(args.db)
    domains = [d["domain"] for d in storage.get_all_domains()]
    if not domains:
        print("[!] No domains in database. Run 'crawl' first.", file=sys.stderr)
        return

    crawler = OnionCrawler(tor_proxy=args.tor_proxy, timeout=args.timeout)
    print(f"[*] Checking availability of {len(domains)} domains...")
    status = crawler.check(domains)

    online = sum(1 for v in status.values() if v)
    pct = 100 * online / len(domains) if domains else 0
    print(f"\n[+] {online}/{len(domains)} online ({pct:.1f}%)")

    for domain, is_up in status.items():
        storage.upsert_domain(domain, is_available=int(is_up))


# ── graph ─────────────────────────────────────────────────────────────────────

def cmd_graph(args: argparse.Namespace) -> None:
    storage = DizzyStorage(args.db)
    edges = storage.get_all_edges()
    domains = {d["domain"]: d for d in storage.get_all_domains()}

    graph = OnionWebGraph()
    for d in domains:
        graph.add_domain(d)
    for e in edges:
        graph.add_link(e["src"], e["dst"])

    stats = graph.compute_stats()
    print(f"\n{'='*55}")
    print("Graph Analytics")
    print(f"{'='*55}")
    print(f"  Nodes:            {stats.num_nodes} ({stats.num_onion_nodes} onion, {stats.num_web_nodes} clearweb)")
    print(f"  Edges:            {stats.num_edges}")
    print(f"  SCCs:             {stats.num_sccs} (largest: {stats.largest_scc_nodes} nodes, {stats.largest_scc_edges} edges)")
    print(f"  Avg clustering:   {stats.avg_clustering_onion:.4f}")
    print(f"\n  Bow-tie decomposition:")
    print(f"    Core (LSCC):    {stats.core_size}")
    print(f"    In-component:   {stats.in_size}")
    print(f"    Out-component:  {stats.out_size}")
    print(f"    Tendrils:       {stats.tendrils_size}")
    print(f"\n  Dark→clearweb links: {stats.dark_to_regular_links}")

    top = graph.node_stats(top_n=10)
    print(f"\n  Top 10 nodes by degree:")
    for ns in top:
        print(f"    {'[onion]' if ns.is_onion else '[web]  '} {ns.domain[:50]:50s} deg={ns.total_degree}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(graph.to_dict(), indent=2))
        print(f"\n[+] Graph saved to {args.output}")


# ── wallet analysis ───────────────────────────────────────────────────────────

def cmd_wallets(args: argparse.Namespace) -> None:
    storage = DizzyStorage(args.db)
    addresses = storage.get_all_addresses()
    if not addresses:
        print("[!] No crypto addresses in database.", file=sys.stderr)
        return

    clusterer = WalletClusterer()
    for a in addresses:
        clusterer.add_address(a["address"], a["domain"])

    wallets = clusterer.get_wallets()
    wallets = clusterer.filter_outliers(wallets)

    non_outlier = [w for w in wallets.values() if not w.is_outlier]
    outliers = [w for w in wallets.values() if w.is_outlier]

    print(f"\n{'='*55}")
    print("Cryptocurrency Analytics")
    print(f"{'='*55}")
    print(f"  Total addresses:  {len(addresses)}")
    print(f"  Wallets:          {len(wallets)}")
    print(f"  Outlier wallets:  {len(outliers)} (filtered out)")
    print(f"  Valid wallets:    {len(non_outlier)}")

    # Update storage
    for w in wallets.values():
        for addr in w.addresses:
            storage.update_address_wallet(
                address=addr,
                wallet_id=w.wallet_id,
                is_outlier=w.is_outlier,
            )

    print(f"\n[+] Wallet data saved to {args.db}")


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    storage = DizzyStorage(args.db)
    summary = storage.summary()
    domains = storage.get_all_domains()

    def pct(n: int, total: int) -> str:
        return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0"

    total = summary["domains"]
    print(f"\n{'='*55}")
    print(f"Dizzy Report  —  {total} domains")
    print(f"{'='*55}")

    print(f"\n§5.1 Domain Operations:")
    print(f"  Available:     {pct(summary['available'], total)}")

    v3_count = sum(1 for d in domains if d.get("is_v3"))
    v2_count = sum(1 for d in domains if d.get("is_v3") == 0)
    print(f"  v3 only:       {pct(v3_count, total)}")
    print(f"  v2 only:       {pct(v2_count, total)}")

    tls_count = sum(1 for d in domains if d.get("has_tls"))
    print(f"  Has TLS:       {pct(tls_count, total)}")

    servers = Counter(d.get("server", "") for d in domains if d.get("server"))
    if servers:
        print(f"\n  Top webservers:")
        for sv, cnt in servers.most_common(5):
            print(f"    {sv or 'unknown':20s}: {pct(cnt, total)}")

    print(f"\n§5.2 Web Content:")
    print(f"  Illicit:       {pct(summary['illicit'], total)}")
    print(f"  Tracked (JS):  {pct(summary['tracked'], total)}")
    print(f"  Crawled pages: {summary['pages']}")

    langs = Counter(d.get("language") for d in domains if d.get("language"))
    if langs:
        print(f"\n  Top languages:")
        for lang, cnt in langs.most_common(5):
            print(f"    {lang:15s}: {pct(cnt, total)}")

    cats = Counter(d.get("category") for d in domains if d.get("category"))
    if cats:
        print(f"\n  Categories:")
        for cat, cnt in cats.most_common(6):
            print(f"    {cat:20s}: {pct(cnt, total)}")

    print(f"\n§5.3 Cryptocurrency:")
    print(f"  Domains with crypto:  {pct(summary['attributed'], total)}")
    print(f"  Total addresses:      {summary['addresses']}")

    print(f"\n§5.4 Web Graph:")
    print(f"  Graph edges:          {summary['edges']}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dizzy: Onion Service Crawler & Analyzer (Boshmaf et al. 2023)"
    )
    parser.add_argument("--db", default="dizzy.db", help="SQLite database (default: dizzy.db)")
    parser.add_argument("--tor-proxy", default="socks5h://127.0.0.1:9150")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")

    sub = parser.add_subparsers(dest="command", required=True)

    # seed
    p = sub.add_parser("seed", help="Collect seed domains from onion indexers and Torch")
    p.add_argument("--output", default="seeds.txt", help="Output file (default: seeds.txt)")
    p.add_argument("--no-indexers", action="store_true", help="Skip onion indexer scraping")
    p.add_argument("--no-torch", action="store_true", help="Skip Torch search queries")
    p.add_argument("--indexers", default="", help="Comma-separated extra indexer URLs")
    p.add_argument("--words", default="", help="Path to extra word list file (one word per line)")
    p.add_argument("--max-queries", type=int, default=30, help="Max Torch queries (default: 30)")

    # crawl
    p = sub.add_parser("crawl", help="Crawl and analyze onion services")
    p.add_argument("--seeds", required=True, help="File with seed domains, one per line")
    p.add_argument("--mode", choices=["crawl", "explore"], default="crawl",
                   help="crawl=domains only, explore=follow discovered onions")
    p.add_argument("--max-pages", type=int, default=75, help="Max pages per domain")
    p.add_argument("--model-dir", default="", help="Directory with pre-trained classifier .pkl files")

    # check
    p = sub.add_parser("check", help="Re-check availability of known domains")

    # graph
    p = sub.add_parser("graph", help="Run graph analytics on crawled data")
    p.add_argument("--output", default="", help="Optional path to save graph JSON")

    # wallets
    p = sub.add_parser("wallets", help="Cluster crypto addresses into wallets")

    # report
    p = sub.add_parser("report", help="Print analysis summary report")

    args = parser.parse_args()
    {
        "seed":    cmd_seed,
        "crawl":   cmd_crawl,
        "check":   cmd_check,
        "graph":   cmd_graph,
        "wallets": cmd_wallets,
        "report":  cmd_report,
    }[args.command](args)


if __name__ == "__main__":
    main()
