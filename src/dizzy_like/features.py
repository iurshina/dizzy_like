"""Feature extraction from crawled onion webpages.

Implements all features from Table 1 of the paper:
  Language, Illicitness, Template, Tracking, Attribution, Image, Camera, Wallet.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

# ── Crypto address regexes ────────────────────────────────────────────────────
_BTC_LEGACY = re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b')
_BTC_BECH32 = re.compile(r'\bbc1[a-zA-HJ-NP-Z0-9]{25,39}\b')
_ETH = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
_XMR = re.compile(r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b')

# ── JS fingerprinting / tracking indicators ───────────────────────────────────
_BLACKLISTED_JS: Set[str] = {
    "fingerprint2", "clientjs", "evercookie",
    "navigator.plugins", "navigator.mimeTypes", "screen.colorDepth",
    "navigator.hardwareConcurrency", "navigator.deviceMemory",
    "getBattery", "getGamepads", "RTCPeerConnection",
    "AudioContext", "OfflineAudioContext",
    "canvas.toDataURL", "canvas.getContext",
    "document.cookie", "localStorage", "sessionStorage",
    "indexedDB", "webkitIndexedDB",
    "__utma", "__utmz", "_ga", "_gid",
}

_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "are", "was", "have",
    "not", "from", "but", "they", "you", "all", "had", "her", "his",
    "our", "one", "been", "has", "its", "can", "who", "him", "get",
    "did", "use", "two", "also", "more", "will", "just", "been",
})


@dataclass
class AddressContext:
    address: str
    currency: str
    in_a_tag: bool = False
    in_url: bool = False
    in_form: bool = False
    in_footer: bool = False
    in_table: bool = False
    in_list: bool = False
    in_img: bool = False
    near_donate: bool = False
    near_example: bool = False


@dataclass
class PageFeatures:
    # ── Language ─────────────────────────────────────────────────────────────
    char_3grams: Dict[str, int] = field(default_factory=dict)

    # ── Illicitness ───────────────────────────────────────────────────────────
    uses_css: bool = False
    uses_js: bool = False
    num_chars: int = 0
    num_img_tags: int = 0
    num_button_tags: int = 0
    num_input_tags: int = 0
    lda_top_terms: List[str] = field(default_factory=list)
    num_external_urls: int = 0

    # ── Template ──────────────────────────────────────────────────────────────
    css_rule_sequences: List[str] = field(default_factory=list)
    tfidf_top_terms: List[str] = field(default_factory=list)
    dom_tree_sequence: List[str] = field(default_factory=list)

    # ── Tracking ──────────────────────────────────────────────────────────────
    blacklisted_js_calls: Set[str] = field(default_factory=set)
    has_js_tracking: bool = False

    # ── Attribution (crypto) ──────────────────────────────────────────────────
    crypto_addresses: List[AddressContext] = field(default_factory=list)
    num_crypto_addresses: int = 0

    # ── Image / Camera (placeholders — need raw image bytes) ──────────────────
    perceptual_hash: Optional[str] = None
    prnu_hash: Optional[str] = None

    # Raw cleaned text (useful for text classifiers)
    clean_text: str = ""


def extract_features(
    html: str,
    base_url: str = "",
    external_css: str = "",
    external_js: str = "",
) -> PageFeatures:
    """Extract all Table-1 features from a rendered HTML page."""
    f = PageFeatures()
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return f

    # ── Collect CSS / JS text BEFORE decomposing tags ────────────────────────
    f.uses_css = bool(
        soup.find("link", rel=lambda r: isinstance(r, list) and "stylesheet" in r)
        or soup.find("style")
    )
    f.uses_js = bool(soup.find("script"))

    raw_css = external_css
    for style_tag in soup.find_all("style"):
        raw_css += (style_tag.string or "") + "\n"

    raw_js = external_js
    for script_tag in soup.find_all("script"):
        if not script_tag.get("src"):
            raw_js += (script_tag.string or "") + "\n"

    # ── Clean text (strip scripts/styles for text analysis) ──────────────────
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    f.clean_text = re.sub(r'\s+', ' ', soup.get_text(" ", strip=True))

    # ── Language features (character 3-grams) ─────────────────────────────────
    f.char_3grams = _char_ngrams(f.clean_text, n=3, top=500)

    # ── Remaining structural features ─────────────────────────────────────────
    f.num_chars = len(f.clean_text)
    f.num_img_tags = len(soup.find_all("img"))
    f.num_button_tags = len(soup.find_all("button"))
    f.num_input_tags = len(soup.find_all("input"))

    # ── External URLs ─────────────────────────────────────────────────────────
    base_host = urlparse(base_url).hostname or ""
    f.num_external_urls = sum(
        1 for a in soup.find_all("a", href=True)
        if _is_external(a["href"], base_host)
    )

    # ── TF-IDF top terms ──────────────────────────────────────────────────────
    f.tfidf_top_terms = _top_terms(f.clean_text, n=10)

    # ── LDA top terms (approximated via topic-word co-occurrence) ─────────────
    f.lda_top_terms = _lda_approx(f.clean_text, n_topics=10, top_per_topic=1)

    # ── DOM tree sequences ────────────────────────────────────────────────────
    f.dom_tree_sequence = _dom_sequences(soup, max_depth=4, limit=200)

    # ── CSS rule sequences ────────────────────────────────────────────────────
    f.css_rule_sequences = _css_sequences(raw_css, limit=100)

    # ── Blacklisted JS functions ──────────────────────────────────────────────
    f.blacklisted_js_calls = _find_blacklisted_js(raw_js)
    f.has_js_tracking = len(f.blacklisted_js_calls) > 0

    # ── Crypto address attribution features ───────────────────────────────────
    f.crypto_addresses = _extract_crypto_contexts(html)
    f.num_crypto_addresses = len(f.crypto_addresses)

    return f


# ── Helpers ───────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3, top: int = 500) -> Dict[str, int]:
    text = text.lower()
    counter: Counter = Counter(text[i:i+n] for i in range(len(text) - n + 1))
    return dict(counter.most_common(top))


def _top_terms(text: str, n: int = 10) -> List[str]:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    if not filtered:
        return []
    total = len(filtered)
    tf = Counter(filtered)
    return [w for w, _ in sorted(tf.items(), key=lambda x: -x[1] / total)[:n]]


def _lda_approx(text: str, n_topics: int = 10, top_per_topic: int = 1) -> List[str]:
    """
    Approximate LDA by partitioning vocabulary into n_topics buckets
    and returning the highest-frequency term per bucket.
    Real Dizzy uses sklearn LatentDirichletAllocation.
    """
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered = sorted(set(w for w in words if w not in _STOPWORDS))
    if not filtered:
        return []
    freq = Counter(words)
    bucket_size = max(1, len(filtered) // n_topics)
    top_terms = []
    for i in range(n_topics):
        bucket = filtered[i * bucket_size: (i + 1) * bucket_size]
        if bucket:
            best = max(bucket, key=lambda w: freq[w])
            top_terms.append(best)
    return top_terms[:n_topics]


def _dom_sequences(soup: BeautifulSoup, max_depth: int = 4, limit: int = 200) -> List[str]:
    sequences: List[str] = []

    def walk(node, depth: int, path: List[str]) -> None:
        if depth > max_depth or len(sequences) >= limit:
            return
        if isinstance(node, Tag) and node.name:
            current = path + [node.name]
            sequences.append(">".join(current))
            for child in node.children:
                walk(child, depth + 1, current)

    walk(soup, 0, [])
    return sequences


def _css_sequences(css_text: str, limit: int = 100) -> List[str]:
    rule_re = re.compile(r'([^{]+)\{[^}]*\}', re.S)
    sequences: List[str] = []
    for m in rule_re.finditer(css_text):
        selector = re.sub(r'\s+', ' ', m.group(1).strip())
        parts = selector.split()
        if parts:
            sequences.append(" ".join(parts[:6]))
        if len(sequences) >= limit:
            break
    return sequences


def _find_blacklisted_js(js_text: str) -> Set[str]:
    js_lower = js_text.lower()
    return {pattern for pattern in _BLACKLISTED_JS if pattern.lower() in js_lower}


def _is_external(href: str, base_host: str) -> bool:
    if not href or href.startswith(("#", "mailto:", "javascript:")):
        return False
    parsed = urlparse(href)
    host = parsed.hostname
    return host is not None and host != "" and host != base_host


def _extract_crypto_contexts(html: str) -> List[AddressContext]:
    """Extract cryptocurrency addresses and their HTML context features."""
    results: List[AddressContext] = []
    seen: set = set()

    for pattern, currency in [
        (_BTC_LEGACY, "bitcoin"),
        (_BTC_BECH32, "bitcoin"),
        (_ETH, "ethereum"),
        (_XMR, "monero"),
    ]:
        for m in pattern.finditer(html):
            addr = m.group(0)
            if addr in seen:
                continue
            seen.add(addr)

            start = m.start()
            ctx = html[max(0, start - 300): start + len(addr) + 300].lower()

            results.append(AddressContext(
                address=addr,
                currency=currency,
                in_a_tag=bool(re.search(r'<a\b', ctx)),
                in_url="href=" in ctx and addr.lower() in ctx,
                in_form=bool(re.search(r'<form\b', ctx)),
                in_footer=bool(re.search(r'<footer\b', ctx)),
                in_table=bool(re.search(r'<t[dhr]\b', ctx)),
                in_list=bool(re.search(r'<[ou]l\b', ctx)),
                in_img=bool(re.search(r'<img\b', ctx)),
                near_donate=bool(re.search(r'\b(donat|pay|send|wallet|tip)\b', ctx)),
                near_example=bool(re.search(r'\bexample\b', ctx)),
            ))

    return results
