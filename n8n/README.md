# n8n workflow: RAG Sync Monitor → Qdrant

Keep a Qdrant collection in sync with a docs site from n8n, embedding only the chunks that changed.

```
Manual / daily trigger → Config → ensure collection → RAG Sync Monitor (Apify)
    → split changes ─┬─ deletes ─────────────────────────────▶ Qdrant: delete
                     └─ upserts → OpenAI embeddings (64/batch) → Qdrant: upsert
```

It uses only built-in nodes (HTTP Request, Code, If, Set, triggers), so it works on **n8n Cloud and self-hosted** with no community nodes to install.

## Import

1. Download [`rag-sync-qdrant.json`](rag-sync-qdrant.json).
2. In n8n: **Workflows → Import from File** (or paste the JSON onto the canvas).

## Set up (5 minutes)

1. **Config** node: set `siteUrl`, `stateKey` (one per site; keeps the baseline between runs), `qdrantUrl` and `collection`.
2. Create three **Header Auth** credentials and select them on the HTTP nodes:

   | Credential | Header name | Header value |
   |---|---|---|
   | Apify | `Authorization` | `Bearer <APIFY_TOKEN>` |
   | OpenAI | `Authorization` | `Bearer <OPENAI_API_KEY>` |
   | Qdrant | `api-key` | your Qdrant Cloud API key (any value for a local Qdrant without auth) |

3. Click **Test workflow**. The first run indexes the site; run it again and only changes come through.
4. **Activate** the workflow to run it every day at 03:00.

## Notes

- **Embedding model:** `text-embedding-3-small` (1536 dimensions). If you change the model, update `vectorSize` and use a new collection. `openAiApiBase` can point at any OpenAI-compatible endpoint.
- **Run length:** the Actor call waits up to 5 minutes, which suits sites up to a few hundred pages per run. Unchanged pages are cheap to check; raise `maxPages` gradually.
- **Point IDs** use the same mapping as `sync_qdrant.py`, so the Python script and this workflow can maintain the same collection.
- **Payload** on every point: `chunk_id`, `text`, `url`, `title`, `section`, `chunk_index`, `source_type`, `synced_at`, ready for citations.

## Tested

`test/run_n8n_test.py` imports the workflow into a real n8n (2.42) and runs it three times against local stand-ins for Apify, OpenAI and Qdrant: 70 new chunks (two embedding batches), then 2 deletes + 1 new chunk, then no changes. It checks auth headers, request shapes, batching, vector/text pairing and the final collection contents.
