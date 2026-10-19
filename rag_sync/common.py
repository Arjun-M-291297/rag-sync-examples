"""Shared helpers: run RAG Sync Monitor, split its output into deletes and upserts, embed text."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar

ACTOR_ID = 'cipherwise/rag-sync-monitor'
Embedder = Callable[[list[str]], list[list[float]]]
T = TypeVar('T')


def actor_input_from_env() -> dict[str, Any]:
    """Build the Actor input from environment variables (SITE_URL is required)."""
    site_url = os.environ['SITE_URL']
    actor_input: dict[str, Any] = {
        'startUrls': [{'url': site_url}],
        'maxPages': int(os.environ.get('MAX_PAGES', '200')),
    }
    if os.environ.get('STATE_KEY'):
        actor_input['stateKey'] = os.environ['STATE_KEY']
    return actor_input


def fetch_changes(actor_input: dict[str, Any] | None = None, dataset_id: str | None = None) -> list[dict[str, Any]]:
    """Run the Actor (or reuse an earlier run's dataset) and return its upsert/delete records."""
    from apify_client import ApifyClient

    client = ApifyClient(os.environ['APIFY_TOKEN'])
    if dataset_id is None:
        run = client.actor(ACTOR_ID).call(run_input=actor_input or actor_input_from_env())
        if run is None or run.status != 'SUCCEEDED':
            raise RuntimeError(f'Actor run did not succeed: {getattr(run, "status", "no run")}')
        dataset_id = run.default_dataset_id
    return list(client.dataset(dataset_id).iterate_items())


def split(items: Iterable[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (ids to delete, records to upsert). Apply the deletes first."""
    deletes, upserts = [], []
    for item in items:
        if item['op'] == 'delete':
            deletes.append(item['chunk_id'])
        elif item['op'] == 'upsert':
            upserts.append(item)
    return deletes, upserts


def metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Citation-friendly metadata. No None values, since several databases reject them."""
    return {
        'url': item['url'],
        'title': item.get('title') or '',
        'section': item.get('heading_path') or '',
        'chunk_index': int(item.get('chunk_index') or 0),
        'source_type': item.get('source_type') or 'html',
        'synced_at': item.get('synced_at') or '',
    }


def batched(values: list[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def default_embedder() -> tuple[Embedder, int]:
    """A local, free embedding model (fastembed). Swap in OpenAI, Cohere etc. by writing your own Embedder."""
    from fastembed import TextEmbedding

    model = TextEmbedding(os.environ.get('EMBED_MODEL', 'BAAI/bge-small-en-v1.5'))

    def embed(texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in model.embed(texts)]

    return embed, len(embed(['dimension probe'])[0])
