"""Apply two simulated RAG Sync Monitor runs to each database and check the final state.

Chroma, Qdrant (in-memory) and pgvector (via pgserver, if installed) are real databases; Pinecone uses a stand-in
index because it is a cloud-only service.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from rag_sync.common import batched, metadata, split

DIM = 8


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic, dependency-free vectors for tests."""
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        out.append([b / 255 for b in digest[:DIM]])
    return out


def upsert(chunk_id: str, url: str, text: str, section: str = 'Guide > Install') -> dict[str, Any]:
    return {'op': 'upsert', 'chunk_id': chunk_id, 'url': url, 'title': 'Guide', 'heading_path': section,
            'chunk_index': 0, 'text': text, 'page_status': 'new', 'source_type': 'html', 'description': '',
            'language': 'en', 'synced_at': '2026-10-19T10:00:00+00:00'}


def delete(chunk_id: str, url: str, reason: str = 'removed_from_page') -> dict[str, Any]:
    return {'op': 'delete', 'chunk_id': chunk_id, 'url': url, 'reason': reason, 'synced_at': '2026-10-19T11:00:00+00:00'}


A, B = 'https://docs.example.com/a', 'https://docs.example.com/b'
RUN_1 = [upsert('a1', A, 'Alpha one.'), upsert('a2', A, 'Alpha two.'), upsert('b1', B, 'Beta one.')]
RUN_2 = [delete('a2', A), delete('b1', B, 'page_gone'), upsert('a3', A, 'Alpha three.', 'Guide > Usage')]
EXPECTED = {'a1': 'Alpha one.', 'a3': 'Alpha three.'}


def test_split_and_metadata() -> None:
    deletes, upserts = split(RUN_2)
    assert deletes == ['a2', 'b1'] and [u['chunk_id'] for u in upserts] == ['a3']
    meta = metadata({'url': A, 'chunk_id': 'x', 'title': None, 'heading_path': None})
    assert all(v is not None for v in meta.values())
    assert list(batched(list(range(5)), 2)) == [[0, 1], [2, 3], [4]]


def test_chroma() -> None:
    chromadb = pytest.importorskip('chromadb')
    from sync_chroma import apply

    collection = chromadb.EphemeralClient().get_or_create_collection('docs_test', metadata={'hnsw:space': 'cosine'})
    for run in (RUN_1, RUN_2, RUN_2):  # the repeated run checks idempotency
        apply(run, collection, fake_embed)
    got = collection.get(include=['documents', 'metadatas'])
    assert dict(zip(got['ids'], got['documents'], strict=True)) == EXPECTED
    assert {m['section'] for m in got['metadatas']} == {'Guide > Install', 'Guide > Usage'}
    hits = collection.query(query_embeddings=fake_embed(['Alpha three.']), n_results=1)
    assert hits['ids'][0] == ['a3']


def test_qdrant() -> None:
    pytest.importorskip('qdrant_client')
    from qdrant_client import QdrantClient

    from sync_qdrant import apply, ensure_collection, point_id

    client = QdrantClient(':memory:')
    ensure_collection(client, 'docs', DIM)
    ensure_collection(client, 'docs', DIM)  # second call is a no-op
    for run in (RUN_1, RUN_2, RUN_2):
        apply(run, client, 'docs', fake_embed)
    points, _ = client.scroll('docs', limit=100, with_payload=True)
    assert {p.payload['chunk_id']: p.payload['text'] for p in points} == EXPECTED
    assert {str(p.id) for p in points} == {point_id('a1'), point_id('a3')}
    hit = client.query_points('docs', query=fake_embed(['Alpha one.'])[0], limit=1).points[0]
    assert hit.payload['chunk_id'] == 'a1' and hit.payload['url'] == A


class FakePineconeIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, dict[str, dict[str, Any]]] = {}

    def upsert(self, vectors: list[dict[str, Any]], namespace: str = '') -> None:
        assert len(vectors) <= 1000
        for v in vectors:
            assert all(x is not None for x in v['metadata'].values()), 'Pinecone rejects null metadata'
            self.vectors.setdefault(namespace, {})[v['id']] = v

    def delete(self, ids: list[str], namespace: str = '') -> None:
        for i in ids:
            self.vectors.get(namespace, {}).pop(i, None)


def test_pinecone_logic() -> None:
    from sync_pinecone import apply

    index = FakePineconeIndex()
    for run in (RUN_1, RUN_2, RUN_2):
        apply(run, index, fake_embed, namespace='docs')
    assert {k: v['metadata']['text'] for k, v in index.vectors['docs'].items()} == EXPECTED
    assert len(index.vectors['docs']['a1']['values']) == DIM


@pytest.fixture(scope='module')
def pg_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    pgserver = pytest.importorskip('pgserver')
    server = pgserver.get_server(tmp_path_factory.mktemp('pg'), cleanup_mode='stop')
    yield server.get_uri()
    server.cleanup()


def test_pgvector(pg_uri: str) -> None:
    psycopg = pytest.importorskip('psycopg')
    pytest.importorskip('pgvector')
    import numpy as np
    from pgvector.psycopg import register_vector

    from sync_pgvector import apply, ensure_table

    with psycopg.connect(pg_uri, autocommit=True) as conn:
        ensure_table(conn, 'doc_chunks', DIM)
        ensure_table(conn, 'doc_chunks', DIM)  # second call is a no-op
        for run in (RUN_1, RUN_2, RUN_2):
            apply(run, conn, 'doc_chunks', fake_embed)
        rows = dict(conn.execute('SELECT id, content FROM doc_chunks').fetchall())
        assert rows == EXPECTED
        register_vector(conn)
        nearest = conn.execute('SELECT id FROM doc_chunks ORDER BY embedding <=> %s LIMIT 1',
                               (np.array(fake_embed(['Alpha three.'])[0], dtype=np.float32),)).fetchone()[0]
        assert nearest == 'a3'

        # A failing batch must roll back everything, never leave the table half-synced.
        bad = [delete('a1', A), {**upsert('a4', A, 'Alpha four.'), 'url': None}]
        with pytest.raises(psycopg.errors.NotNullViolation):
            apply(bad, conn, 'doc_chunks', fake_embed)
        assert dict(conn.execute('SELECT id, content FROM doc_chunks').fetchall()) == EXPECTED


def test_qdrant_point_ids_are_stable_uuids() -> None:
    import uuid as _uuid

    from sync_qdrant import point_id

    real = '037ec235493f055422c37060'  # a real 24-hex chunk_id from the Actor
    assert point_id(real) == '037ec235-493f-0554-22c3-706000000000'
    assert str(_uuid.UUID(point_id(real))) == point_id(real)  # valid UUID syntax
    assert point_id(real) == point_id(real.upper()) and point_id('a1') != point_id('a3')
    assert _uuid.UUID(point_id('not-hex!'))  # fallback still gives a valid UUID
