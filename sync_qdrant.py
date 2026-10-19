"""Apply RAG Sync Monitor changes to a Qdrant collection."""

from __future__ import annotations

import argparse
import os
import re
import uuid
from typing import Any

from rag_sync.common import Embedder, batched, default_embedder, fetch_changes, metadata, split

# Qdrant point IDs must be integers or UUIDs. Chunk IDs are hex, so they are padded into UUID form: reversible,
# no hashing, and identical to the n8n template (n8n/rag-sync-qdrant.json). The chunk_id is also kept in the payload.
FALLBACK_NAMESPACE = uuid.UUID('6f1c2b8e-5d1a-4c2e-9a77-3b9e1f4d2c10')


def point_id(chunk_id: str) -> str:
    h = chunk_id.lower()
    if not re.fullmatch(r'[0-9a-f]{1,32}', h):
        return str(uuid.uuid5(FALLBACK_NAMESPACE, chunk_id))
    h = h.ljust(32, '0')
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'


def ensure_collection(client: Any, name: str, dim: int) -> None:
    from qdrant_client import models

    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))


def apply(items: list[dict[str, Any]], client: Any, collection: str, embed: Embedder) -> tuple[int, int]:
    from qdrant_client import models

    deletes, upserts = split(items)
    for ids in batched(deletes, 1000):
        client.delete(collection, points_selector=models.PointIdsList(points=[point_id(i) for i in ids]))
    for batch in batched(upserts, 100):
        vectors = embed([u['text'] for u in batch])
        client.upsert(collection, points=[
            models.PointStruct(id=point_id(u['chunk_id']), vector=v,
                               payload={**metadata(u), 'text': u['text'], 'chunk_id': u['chunk_id']})
            for u, v in zip(batch, vectors, strict=True)
        ])
    return len(upserts), len(deletes)


def main() -> None:
    from qdrant_client import QdrantClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', help='Re-apply the output of an earlier run instead of starting a new one')
    args = parser.parse_args()

    embed, dim = default_embedder()
    client = QdrantClient(url=os.environ.get('QDRANT_URL', 'http://localhost:6333'),
                          api_key=os.environ.get('QDRANT_API_KEY') or None)
    name = os.environ.get('COLLECTION', 'docs')
    ensure_collection(client, name, dim)
    upserted, deleted = apply(fetch_changes(dataset_id=args.dataset_id), client, name, embed)
    print(f'Qdrant: {upserted} upserted, {deleted} deleted in collection "{name}"')


if __name__ == '__main__':
    main()
