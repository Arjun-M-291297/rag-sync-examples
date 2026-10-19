# RAG Sync Monitor examples

Keep a vector database in sync with a docs site, help center or knowledge base, re-embedding only what changed.

These scripts run [RAG Sync Monitor](https://apify.com/cipherwise/rag-sync-monitor) on Apify and apply its output to **Chroma**, **Pinecone**, **Qdrant** or **Postgres + pgvector**. Each run returns only the chunks that were added, changed or removed since the last run, as `upsert` and `delete` records with stable IDs.

```
docs site ──▶ RAG Sync Monitor (Apify) ──▶ upserts + deletes ──▶ embed ──▶ your vector DB
```

## Quickstart (Chroma, about 5 minutes)

```bash
git clone https://github.com/Arjun-M-291297/rag-sync-examples && cd rag-sync-examples
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt chromadb
cp .env.example .env    # set APIFY_TOKEN and SITE_URL
set -a && source .env && set +a
python sync_chroma.py
```

Run the same command again later: only the changes come through. On an unchanged site the second run outputs nothing.

## Pick your database

| Database | Install | Run | Needs |
|---|---|---|---|
| Chroma | `pip install chromadb` | `python sync_chroma.py` | nothing else (local folder) |
| Pinecone | `pip install "pinecone>=7"` | `python sync_pinecone.py` | `PINECONE_API_KEY`; the index is created if missing |
| Qdrant | `pip install qdrant-client` | `python sync_qdrant.py` | `QDRANT_URL` (+ `QDRANT_API_KEY` for Qdrant Cloud); the collection is created if missing |
| pgvector | `pip install "psycopg[binary]>=3.2" pgvector numpy` | `python sync_pgvector.py` | `DATABASE_URL`; the table and HNSW index are created if missing |

All settings live in `.env` (see `.env.example`). Each script also accepts `--dataset-id <id>` to re-apply an earlier run's output without starting a new run.

## No-code: n8n workflow

Prefer n8n? Import [`n8n/rag-sync-qdrant.json`](n8n/rag-sync-qdrant.json): a ready-made workflow that syncs a site into Qdrant with OpenAI embeddings, using only built-in nodes. Setup notes in [`n8n/README.md`](n8n/README.md).

## How the sync works

1. The script starts a run of `cipherwise/rag-sync-monitor` for `SITE_URL` and waits for it.
2. It reads the run's dataset: `delete` records (removed chunks, gone or redirected pages) and `upsert` records (new or changed chunks).
3. It applies **deletes first, then upserts**, in batches. Both are idempotent, so re-applying a run is safe.
4. Upserts are embedded locally with [fastembed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, 384 dimensions, free). To use OpenAI, Cohere or another model, replace `default_embedder()` in `rag_sync/common.py`.

Every chunk is stored with citation metadata: `url`, `title`, `section` (heading path), `chunk_index`, `source_type`.

> Use the **same embedding model** for searching as for syncing. If you change models, re-embed everything (start a new `STATE_KEY` and a fresh collection).

## Searching what you synced

```python
from rag_sync.common import default_embedder
embed, _ = default_embedder()
query = embed(["how do I configure authentication?"])[0]

# Chroma
collection.query(query_embeddings=[query], n_results=5)
# Pinecone
index.query(vector=query, top_k=5, include_metadata=True)
# Qdrant
client.query_points("docs", query=query, limit=5)
# pgvector
conn.execute("SELECT url, section, content FROM doc_chunks ORDER BY embedding <=> %s LIMIT 5", (np.array(query),))
```

## Run it every night with GitHub Actions

[`github-actions/sync.yml`](github-actions/sync.yml) is a ready-made workflow that runs the sync daily:

1. Copy this repo (or just the scripts) into your own repository, and put the workflow file at `.github/workflows/sync.yml`.
2. In **Settings → Secrets and variables → Actions**, add:
   - secrets: `APIFY_TOKEN` and your database's credentials (`PINECONE_API_KEY`, `QDRANT_URL` / `QDRANT_API_KEY`, or `DATABASE_URL`)
   - variables: `SITE_URL`, `STATE_KEY`, and `TARGET` = `pinecone`, `qdrant` or `pgvector`
3. Run it once from the **Actions** tab, then let the schedule take over.

You can also schedule the Actor itself in Apify and trigger your own code with a webhook.

## Costs

The Actor is pay-per-event: about **$0.50 per 1,000 pages checked** and **$0.20 per 1,000 chunk changes** (see the [pricing tab](https://apify.com/cipherwise/rag-sync-monitor/pricing) for current prices). A quiet site mostly costs page checks, and unchanged pages are skipped with HTTP 304. Embedding with fastembed runs locally for free.

## License

MIT
