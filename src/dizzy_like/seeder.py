"""Seed collection (§3.1.1 — State Orchestration).

Two sources, as described in the paper:
  1. Onion indexers  — parse known index pages for listed .onion domains
  2. Torch search    — query the Tor search engine with dictionary word-pairs
                       and collect result domains

Both use the Tor proxy so we never touch a .onion directly from clearnet.
"""
from __future__ import annotations

import itertools
import re
import time
from typing import Iterable, Iterator, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

_ONION_RE = re.compile(r'\b([a-z2-7]{56}\.onion|[a-z2-7]{16}\.onion)\b', re.I)

# ── Known public onion indexers (from paper footnote 2 + well-known mirrors) ──
INDEXER_URLS = [
    # OnionDir (cited in paper as footnote 2)
    "http://oniondlu6xudklblcijrwwkduu2tdle3rav7nlszrjhrxpjtkg4brmqqd.onion",
    # Torch search engine (cited in paper as footnote 3)
    "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion",
    # dark.fail clearnet mirror (lists curated onion links)
    "https://dark.fail/",
]

# Torch search URL template
_TORCH_SEARCH = (
    "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion"
    "/4a1f6b371c/search.cgi?cmd=Search&form=extended&q={query}&ps=20"
)

# ── Built-in seed word lists (§3.1.1: single words + 2-word combinations) ────
# Representative sample across domains; extend with full language dictionaries.
_SEED_WORDS = [
    # English — general
    "forum", "market", "shop", "service", "news", "blog", "wiki",
    "library", "privacy", "security", "anonymous", "hidden", "dark",
    "chat", "email", "hosting", "file", "search", "index", "link",
    # Common darkweb categories
    "crypto", "bitcoin", "exchange", "wallet", "monero",
    "drug", "pharmacy", "hacking", "exploit", "leak",
    # Non-English (Russian, German, French — paper covers 50 languages)
    "форум", "рынок", "новости",   # Russian
    "Forum", "Markt", "Nachrichten",  # German
    "forum", "marché", "actualités",  # French
]


def _tor_session(proxy: str, user_agent: str, timeout: int) -> requests.Session:
    s = requests.Session()
    s.proxies = {"http": proxy, "https": proxy}
    s.headers["User-Agent"] = user_agent
    s.timeout = timeout
    return s


def _extract_onions(text: str) -> Set[str]:
    return {m.group(1).lower() for m in _ONION_RE.finditer(text)}


def _fetch_text(url: str, session: requests.Session, timeout: int) -> str:
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        return resp.text
    except Exception as exc:
        print(f"  [!] {url}: {exc}")
        return ""


def collect_from_indexers(
    indexer_urls: List[str],
    session: requests.Session,
    timeout: int,
    delay: float,
) -> Set[str]:
    """Fetch each indexer page and scrape all .onion domains from it."""
    found: Set[str] = set()
    for url in indexer_urls:
        print(f"  [indexer] {url}")
        html = _fetch_text(url, session, timeout)
        if not html:
            time.sleep(delay)
            continue

        # Try to follow paginated links (common on onion indexes)
        soup = BeautifulSoup(html, "html.parser")
        page_urls = {url}
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r'page|next|p=\d', href, re.I):
                abs_url = urljoin(url, href)
                if urlparse(abs_url).hostname == urlparse(url).hostname:
                    page_urls.add(abs_url)

        for page_url in list(page_urls)[:5]:  # cap pagination depth
            text = _fetch_text(page_url, session, timeout)
            batch = _extract_onions(text)
            found.update(batch)
            print(f"    found {len(batch)} onions at {page_url}")
            time.sleep(delay)

    return found


def _word_queries(words: List[str], max_pairs: int = 50) -> Iterator[str]:
    """Single words first, then 2-word combinations (§3.1.1)."""
    yield from words
    pairs = list(itertools.combinations(words[:20], 2))  # keep it manageable
    for a, b in pairs[:max_pairs]:
        yield f"{a} {b}"


def collect_from_torch(
    queries: Iterable[str],
    session: requests.Session,
    timeout: int,
    delay: float,
    max_queries: int = 30,
) -> Set[str]:
    """Query Torch with each search term and extract result domains."""
    found: Set[str] = set()
    for i, query in enumerate(queries):
        if i >= max_queries:
            break
        url = _TORCH_SEARCH.format(query=requests.utils.quote(query))
        print(f"  [torch] query={query!r}")
        html = _fetch_text(url, session, timeout)
        batch = _extract_onions(html)
        found.update(batch)
        print(f"    found {len(batch)} onions")
        time.sleep(delay)
    return found


def collect_seeds(
    tor_proxy: str = "socks5h://127.0.0.1:9150",
    timeout: int = 45,
    delay: float = 2.0,
    use_indexers: bool = True,
    use_torch: bool = True,
    extra_indexers: List[str] | None = None,
    extra_words: List[str] | None = None,
    max_torch_queries: int = 30,
    output: str = "seeds.txt",
) -> List[str]:
    """
    Full seed collection pipeline from §3.1.1.
    Returns sorted list of discovered .onion domains and writes them to output.
    """
    user_agent = "Dizzy Research Crawler (passive, seed collection)"
    session = _tor_session(tor_proxy, user_agent, timeout)

    all_seeds: Set[str] = set()

    if use_indexers:
        indexers = INDEXER_URLS + (extra_indexers or [])
        print(f"[*] Collecting from {len(indexers)} indexer(s)...")
        found = collect_from_indexers(indexers, session, timeout, delay)
        print(f"[+] Indexers yielded {len(found)} unique domains")
        all_seeds.update(found)

    if use_torch:
        words = _SEED_WORDS + (extra_words or [])
        queries = _word_queries(words)
        print(f"[*] Querying Torch (up to {max_torch_queries} queries)...")
        found = collect_from_torch(queries, session, timeout, delay, max_torch_queries)
        print(f"[+] Torch yielded {len(found)} unique domains")
        all_seeds.update(found)

    seeds = sorted(all_seeds)
    print(f"\n[+] Total unique seeds: {len(seeds)}")

    if output:
        with open(output, "w") as fh:
            fh.write("\n".join(seeds) + "\n")
        print(f"[+] Written to {output}")

    return seeds
