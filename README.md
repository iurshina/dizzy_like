# Dizzy-like

> **Vibe coded** Python implementation of [Dizzy: Large-Scale Crawling and Analysis of Onion Services](https://arxiv.org/abs/2209.07202) (Boshmaf et al., 2023).

Dizzy is a system for crawling and analyzing Tor onion services. This implementation covers the core pipeline described in the paper: crawling, feature extraction, domain classification, cryptocurrency analytics, and web graph analysis.

## What it does

| Component | Paper section | What's implemented |
|-----------|--------------|-------------------|
| **Crawler** | §3.1 | Tor-based BFS crawler with explore / update / check modes, link extraction, onion domain discovery |
| **Feature extraction** | §3.2.1 / Table 1 | All features: char 3-grams, CSS/JS detection, DOM sequences, TF-IDF, LDA approx, blacklisted JS, crypto address context |
| **Classifiers** | §3.2.1 | Language (NB), Illicitness (RF), Category (RF+OvR, 6 classes), Template (NB), Tracking (SVM), Attribution (RF) |
| **Crypto analytics** | §3.2.2 | BTC/ETH/XMR address extraction, wallet clustering via union-find (common-input heuristic), outlier detection (Isolation Forest) |
| **Graph analytics** | §3.2.3 | Directed onion/clearweb graph, SCC, bow-tie decomposition, PageRank, dark→clearweb link analysis |
| **Storage** | §3.3 | SQLite database for domains, pages, addresses, and graph edges |

> **Note:** Classifiers require labeled training data to be useful. The framework is ready — bring your own ground-truth datasets (see §4.1 of the paper for the ones the authors used).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd dizzy_like
uv sync
```

You also need Tor running locally:

```bash
# macOS
brew install tor && brew services start tor

# Linux
sudo apt install tor && sudo systemctl start tor
```

By default the crawler connects through `socks5h://127.0.0.1:9150` (Tor Browser) or change `--tor-proxy` to `socks5h://127.0.0.1:9050` for the system Tor daemon.

## Usage

### Step 0: Collect seeds

The paper (§3.1.1) seeds the crawler from two sources: **onion indexers** (like OnionDir) and **Torch**, a darkweb search engine, queried with single words and 2-word combinations from language dictionaries. The `seed` command automates this:

```bash
# Collect from both OnionDir and Torch (requires Tor)
uv run dizzy seed

# Indexers only (faster, no search queries)
uv run dizzy seed --no-torch

# Provide your own extra word list
uv run dizzy seed --words wordlist.txt --max-queries 50

# Custom output path
uv run dizzy seed --output my_seeds.txt
```

This writes `seeds.txt` which is fed directly to `crawl`.

### Crawl onion services

```bash
# seeds.txt — one domain per line, e.g.:
# duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion

uv run dizzy crawl --seeds seeds.txt

# Follow discovered onions recursively (explore mode)
uv run dizzy crawl --seeds seeds.txt --mode explore

# Limit pages per domain (default: 75)
uv run dizzy crawl --seeds seeds.txt --max-pages 20
```

### Check availability of known domains

```bash
uv run dizzy check
```

### Graph analytics

```bash
uv run dizzy graph

# Also save graph JSON
uv run dizzy graph --output graph.json
```

### Wallet clustering

```bash
uv run dizzy wallets
```

### Print report

```bash
uv run dizzy report
```

### Global options

```
--db          SQLite database path (default: dizzy.db)
--tor-proxy   Tor SOCKS5 proxy (default: socks5h://127.0.0.1:9150)
--timeout     Request timeout in seconds (default: 45)
--delay       Delay between requests in seconds (default: 2.0)
```

## Training classifiers

```python
from dizzy_like.classifiers import DizzyClassifiers
from dizzy_like.features import extract_features

# Prepare labeled data
texts  = [...]   # list of page text strings
labels = [...]   # e.g. ["marketplace", "indexer", ...]

clf = DizzyClassifiers()
clf.category.fit(texts, labels)
clf.illicitness.fit(feature_list, illicit_labels)
# ... etc.

clf.save("models/")   # saves .pkl files
```

Then pass `--model-dir models/` to `crawl` to use them.

## Ethical use

This tool is intended for passive security research and e-crime investigation, consistent with the ethical guidelines described in §4.2 of the paper. It only targets publicly available onion services, does not authenticate, does not pay, and respects `robots.txt`. Do not use it to crawl at a scale that overwhelms Tor guard nodes.

## Reference

> Yazan Boshmaf, Isuranga Perera, Udesh Kumarasinghe, Sajitha Liyanage, Husam Al Jawaheri.
> **Dizzy: Large-Scale Crawling and Analysis of Onion Services.**
> arXiv:2209.07202 [cs.CR], 2023.
