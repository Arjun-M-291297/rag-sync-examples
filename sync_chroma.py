"""Apply RAG Sync Monitor changes to a Chroma collection."""

from __future__ import annotations

import argparse
import os
from typing import Any

from rag_sync.common import Embedder, batched, default_embedder, fetch_changes, metadata, split


def apply(items: list[dict[str, Any]], collection: Any, embed: Embedder) -> tuple[int, int]:
    deletes, upserts = split(items)
    for ids in batched(deletes, 500):
        collection.delete(ids=ids)
    for batch in batched(upserts, 100):
        collection.upsert(
            ids=[u['chunk_id'] for u in batch],
            documents=[u['text'] for u in batch],
            metadatas=[metadata(u) for u in batch],
            embeddings=embed([u['text'] for u in batch]),
        )
    return len(upserts), len(deletes)


def main() -> None:
    import chromadb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', help='Re-apply the output of an earlier run instead of starting a new one')
    args = parser.parse_args()

    embed, _ = default_embedder()
    client = chromadb.PersistentClient(os.environ.get('CHROMA_PATH', './chroma'))
    collection = client.get_or_create_collection(os.environ.get('COLLECTION', 'docs'), metadata={'hnsw:space': 'cosine'})
    upserted, deleted = apply(fetch_changes(dataset_id=args.dataset_id), collection, embed)
    print(f'Chroma: {upserted} upserted, {deleted} deleted, {collection.count()} chunks in "{collection.name}"')


if __name__ == '__main__':
    main()
