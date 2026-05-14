# TODO

## Crawler

- [ ] Parallel crawling — async (`asyncio` + `aiohttp`) or multiprocessing pool; target: N concurrent workers per Tor circuit
- [ ] Per-guard-node rate limiting — track which guard node each circuit uses and cap requests per node (paper used 150 crawlers balanced across guard nodes)
- [ ] Retry logic — exponential backoff on timeouts/connection errors; onion services are flaky
- [ ] Resume support — persist crawl queue to SQLite so a crash doesn't restart from zero
- [ ] JS rendering — integrate Playwright or Splash to handle JS-heavy pages (paper used a cluster of JS rendering engines)
- [ ] `robots.txt` enforcement — actually fetch and parse it before crawling a domain
- [ ] Redirect following limits — cap redirect chains to avoid loops
- [ ] Distributed job queue — replace in-memory deque with Redis or a task queue (Celery, RQ) for multi-machine crawling

## Storage

- [ ] Replace SQLite with PostgreSQL for concurrent writes from parallel crawlers
- [ ] Stream pages to disk instead of accumulating `full_html` in memory
- [ ] Content deduplication — store HTML by content hash, avoid saving identical pages multiple times
- [ ] Graph database — migrate graph edges to Neo4j or a dedicated graph DB for large-scale analytics (paper used a distributed graph DB)

## Embeddings & Clustering

- [ ] Page embeddings — encode cleaned page text with `BAAI/bge-m3` (multilingual, 100+ languages, 8192 token context, dense+sparse+colbert); good fit since onion services span 40+ languages. Store vectors per page.
- [ ] Domain-level embeddings — aggregate page embeddings per domain (mean pool) to get a single domain representation
- [ ] Template clustering — replace the current NB template classifier with embedding-based clustering (HDBSCAN or AHC) to group visually/textually similar domains without needing labels
- [ ] Category clustering — use embeddings + k-means or HDBSCAN as an unsupervised alternative to the RF category classifier; useful before labeled data is available
- [ ] Image embeddings — extract CNN features (e.g. CLIP) from page screenshots for image-level similarity; complements the perceptual hash approach in the paper (maybe not...)
- [ ] Vector store — store embeddings in a vector DB (pgvector, Qdrant, or FAISS index on disk) for fast nearest-neighbour lookup and semantic search across crawled domains

## ML Classifiers

- [ ] Train and ship baseline weights — at minimum for language detection (NB on Wikipedia abstracts as in the paper) and illicitness (RF)
- [ ] Retraining pipeline — scheduled retraining as new labeled data accumulates
- [ ] Cross-validation reporting — AUC scores per classifier to match Table 3
- [ ] LDA — replace the current approximation with real `sklearn.decomposition.LatentDirichletAllocation`
- [ ] Active learning hook for template classifier — paper mentions a built-in labeling feature

## Crypto Analytics

- [ ] False positive filtering — validate extracted addresses with proper checksum verification (Base58Check for BTC, EIP-55 for ETH)
- [ ] Deposit address clustering heuristic — implement alongside the existing common-input heuristic (paper uses both)
- [ ] Blockchain connectivity — integrate a Bitcoin node RPC or block explorer API to fetch real transaction data
- [ ] USD conversion — fetch historical exchange rates (paper used CoinDesk API) to convert BTC values at transaction time

## Graph Analytics

- [ ] Stream graph construction — build graph edge-by-edge from DB instead of loading all edges into memory
- [ ] VirusTotal integration — flag clearweb domains linked from onion services as malicious/benign (paper used VT as malicious domain feed)
- [ ] Incremental analytics — recompute stats only on changed subgraph, not full recompute each run

## Seeder

- [ ] Full language dictionary word lists — paper used single words and 2-word combos from dictionaries in 50 languages; current list is a small sample
- [ ] Ahmia API pagination — sitemap may paginate; handle `<sitemapindex>` as well as `<urlset>`
- [ ] Periodic re-seeding — schedule seed collection to run automatically (new onions appear daily; paper found ~8.9 new domains/day)

## Ops / Infra

- [ ] Docker setup — containerise crawler workers, Tor daemon, and DB so the stack is reproducible
- [ ] Tor daemon management — programmatically spawn and health-check multiple `tor` processes with separate guard nodes (`SocksPort`, `ControlPort` per instance)
- [ ] Metrics — expose crawler throughput, availability rates, queue depth (Prometheus + Grafana or similar)
- [ ] Alerting — notify when crawler stalls or guard node gets overloaded
- [ ] Data access controls — paper stored all data on private infrastructure with restricted access; add auth to any API layer

## Ethics / Safety

- [ ] CSAM detection — integrate a perceptual hash blocklist (e.g. PhotoDNA or NCMEC hashes) to avoid storing illegal image content
- [ ] IRB / ethics review before large-scale deployment
- [ ] Audit log — record what was crawled, when, and by whom
