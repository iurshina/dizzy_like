"""Tor-based onion service crawler with explore/update/check modes (§3.1)."""
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ONION_RE = re.compile(r'\b([a-z2-7]{56}\.onion|[a-z2-7]{16}\.onion)\b', re.I)


@dataclass
class CrawlResult:
    url: str
    domain: str
    status_code: Optional[int] = None
    html: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    onion_links: List[str] = field(default_factory=list)
    error: Optional[str] = None
    crawled_at: float = field(default_factory=time.time)
    server: str = ""
    has_tls: bool = False

    @property
    def ok(self) -> bool:
        return self.status_code is not None and self.status_code < 500


class OnionCrawler:
    """
    Polite, passive crawler that targets publicly available onion services.
    Does not authenticate, pay, or bypass robots.txt.
    Rate-limited to avoid overwhelming guard nodes.
    """

    def __init__(
        self,
        tor_proxy: str = "socks5h://127.0.0.1:9150",
        timeout: int = 45,
        delay: float = 2.0,
        max_pages_per_domain: int = 75,
        user_agent: str = "Dizzy Research Crawler (passive)",
    ):
        self.tor_proxy = tor_proxy
        self.timeout = timeout
        self.delay = delay
        self.max_pages_per_domain = max_pages_per_domain
        self.user_agent = user_agent
        self._global_visited: Set[str] = set()

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        s.proxies = {"http": self.tor_proxy, "https": self.tor_proxy}
        s.headers["User-Agent"] = self.user_agent
        return s

    @staticmethod
    def _url_key(url: str) -> str:
        return hashlib.sha256(url.lower().encode()).hexdigest()

    @staticmethod
    def _extract_links(base_url: str, html: str) -> List[str]:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in ("http", "https"):
                links.append(absolute)
        return links

    @staticmethod
    def _extract_onion_domains(html: str) -> Set[str]:
        return {m.group(1).lower() for m in ONION_RE.finditer(html)}

    def _fetch_once(self, url: str, session: requests.Session) -> CrawlResult:
        domain = urlparse(url).hostname or ""
        try:
            resp = session.get(url, timeout=self.timeout, allow_redirects=True)
            content_type = resp.headers.get("Content-Type", "")
            html = resp.text if "html" in content_type.lower() else ""
            links = self._extract_links(resp.url, html) if html else []
            onion_links = list(self._extract_onion_domains(html)) if html else []
            server = resp.headers.get("Server", "")
            has_tls = resp.url.startswith("https://")
            return CrawlResult(
                url=resp.url,
                domain=urlparse(resp.url).hostname or domain,
                status_code=resp.status_code,
                html=html,
                headers=dict(resp.headers),
                links=links,
                onion_links=onion_links,
                server=server,
                has_tls=has_tls,
            )
        except requests.exceptions.SSLError:
            # HTTPS failed — retry with HTTP
            return CrawlResult(url=url, domain=domain, error="ssl_error")
        except requests.exceptions.ConnectTimeout:
            return CrawlResult(url=url, domain=domain, error="connect_timeout")
        except requests.exceptions.ReadTimeout:
            return CrawlResult(url=url, domain=domain, error="read_timeout")
        except Exception as exc:
            return CrawlResult(url=url, domain=domain, error=str(exc))

    def fetch(self, url: str, session: Optional[requests.Session] = None) -> CrawlResult:
        """Fetch a single URL, trying HTTPS first then HTTP."""
        s = session or self._make_session()
        parsed = urlparse(url)

        # Try HTTPS first, fall back to HTTP
        candidates = [url]
        if parsed.scheme == "http":
            https_url = url.replace("http://", "https://", 1)
            candidates = [https_url, url]

        result = CrawlResult(url=url, domain=parsed.hostname or "")
        for candidate in candidates:
            result = self._fetch_once(candidate, s)
            if result.error != "ssl_error":
                break
            time.sleep(self.delay)

        return result

    def crawl_domain(self, seed_url: str) -> List[CrawlResult]:
        """BFS crawl of a single onion domain, capped at max_pages_per_domain."""
        session = self._make_session()
        domain = urlparse(seed_url).hostname or seed_url
        queue: deque = deque([seed_url])
        domain_visited: Set[str] = set()
        results: List[CrawlResult] = []

        while queue and len(results) < self.max_pages_per_domain:
            url = queue.popleft()
            key = self._url_key(url)
            if key in domain_visited:
                continue
            domain_visited.add(key)

            result = self.fetch(url, session)
            results.append(result)
            time.sleep(self.delay)

            if result.ok and result.html:
                for link in result.links:
                    link_host = urlparse(link).hostname or ""
                    if link_host == domain:
                        lk = self._url_key(link)
                        if lk not in domain_visited:
                            queue.append(link)

        return results

    def explore(self, seeds: List[str]) -> Dict[str, List[CrawlResult]]:
        """
        Explore mode: crawl seeds and follow discovered onion domains.
        Returns mapping domain -> list of CrawlResult.
        """
        pending: deque = deque(seeds)
        discovered: Set[str] = set(seeds)
        all_results: Dict[str, List[CrawlResult]] = {}

        while pending:
            seed = pending.popleft()
            seed_url = seed if seed.startswith("http") else f"http://{seed}"
            dom = urlparse(seed_url).hostname or seed

            print(f"[explore] {dom}")
            results = self.crawl_domain(seed_url)
            all_results[dom] = results

            for result in results:
                for onion in result.onion_links:
                    if onion not in discovered:
                        discovered.add(onion)
                        pending.append(onion)

        return all_results

    def check(self, domains: List[str]) -> Dict[str, bool]:
        """Check mode: verify which domains are currently online."""
        session = self._make_session()
        status: Dict[str, bool] = {}
        for domain in domains:
            url = f"http://{domain}" if not domain.startswith("http") else domain
            result = self._fetch_once(url, session)
            status[domain] = result.ok
            print(f"  {'✓' if result.ok else '✗'} {domain}")
            time.sleep(self.delay)
        return status
