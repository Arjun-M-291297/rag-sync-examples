"""Apply RAG Sync Monitor changes to a Pinecone index."""

from __future__ import annotations

import argparse
import os
from typing import Any

from rag_sync.common import Embedder, batched, default_embedder, fetch_changes, metadata, split


def apply(items: list[dict[str, Any]], index: Any, embed: Embedder, namespace: str = '') -> tuple[int, int]:
    deletes, upserts = split(items)
    for ids in batched(deletes, 1000):
        index.delete(ids=ids, namespace=namespace)
    for batch in batched(upserts, 100):
        vectors = embed([u['text'] for u in batch])
        index.upsert(
            vectors=[
                {'id': u['chunk_id'], 'values': v, 'metadata': {**metadata(u), 'text': u['text']}}
                for u, v in zip(batch, vectors, strict=True)
            ],
            namespace=namespace,
        )
    return len(upserts), len(deletes)


def main() -> None:
    from pinecone import Pinecone, ServerlessSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', help='Re-apply the output of an earlier run instead of starting a new one')
    args = parser.parse_args()

    embed, dim = default_embedder()
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    name = os.environ.get('PINECONE_INDEX', 'docs')
    if not pc.has_index(name):
        pc.create_index(
            name=name, dimension=dim, metric='cosine',
            spec=ServerlessSpec(cloud=os.environ.get('PINECONE_CLOUD', 'aws'),
                                region=os.environ.get('PINECONE_REGION', 'us-east-1')),
        )
    index = pc.Index(name)
    upserted, deleted = apply(fetch_changes(dataset_id=args.dataset_id), index, embed,
                              namespace=os.environ.get('PINECONE_NAMESPACE', ''))
    print(f'Pinecone: {upserted} upserted, {deleted} deleted in index "{name}"')


if __name__ == '__main__':
    main()
