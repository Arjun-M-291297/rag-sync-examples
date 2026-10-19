"""Apply RAG Sync Monitor changes to a Postgres table with pgvector."""

from __future__ import annotations

import argparse
import os
from typing import Any

from rag_sync.common import Embedder, batched, default_embedder, fetch_changes, metadata, split


def ensure_table(conn: Any, table: str, dim: int) -> None:
    from pgvector.psycopg import register_vector
    from psycopg import sql

    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    conn.execute(sql.SQL('''
        CREATE TABLE IF NOT EXISTS {t} (
            id text PRIMARY KEY,
            url text NOT NULL,
            title text NOT NULL DEFAULT '',
            section text NOT NULL DEFAULT '',
            chunk_index integer NOT NULL DEFAULT 0,
            source_type text NOT NULL DEFAULT 'html',
            content text NOT NULL,
            embedding vector({dim}) NOT NULL,
            synced_at timestamptz
        )''').format(t=sql.Identifier(table), dim=sql.Literal(dim)))
    conn.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)').format(
        i=sql.Identifier(f'{table}_embedding_idx'), t=sql.Identifier(table)))


def apply(items: list[dict[str, Any]], conn: Any, table: str, embed: Embedder) -> tuple[int, int]:
    import numpy as np
    from psycopg import sql

    deletes, upserts = split(items)
    upsert_sql = sql.SQL('''
        INSERT INTO {t} (id, url, title, section, chunk_index, source_type, content, embedding, synced_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULLIF(%s, '')::timestamptz)
        ON CONFLICT (id) DO UPDATE SET
            url = EXCLUDED.url, title = EXCLUDED.title, section = EXCLUDED.section,
            chunk_index = EXCLUDED.chunk_index, source_type = EXCLUDED.source_type,
            content = EXCLUDED.content, embedding = EXCLUDED.embedding, synced_at = EXCLUDED.synced_at
    ''').format(t=sql.Identifier(table))
    with conn.transaction():  # all-or-nothing, so a failed run never leaves the table half-synced
        for ids in batched(deletes, 1000):
            conn.execute(sql.SQL('DELETE FROM {t} WHERE id = ANY(%s)').format(t=sql.Identifier(table)), (ids,))
        for batch in batched(upserts, 100):
            vectors = embed([u['text'] for u in batch])
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, [
                    (u['chunk_id'], m['url'], m['title'], m['section'], m['chunk_index'], m['source_type'],
                     u['text'], np.array(v, dtype=np.float32), m['synced_at'])
                    for u, v in zip(batch, vectors, strict=True) for m in [metadata(u)]
                ])
    return len(upserts), len(deletes)


def main() -> None:
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', help='Re-apply the output of an earlier run instead of starting a new one')
    args = parser.parse_args()

    embed, dim = default_embedder()
    table = os.environ.get('PG_TABLE', 'doc_chunks')
    with psycopg.connect(os.environ['DATABASE_URL'], autocommit=True) as conn:
        ensure_table(conn, table, dim)
        upserted, deleted = apply(fetch_changes(dataset_id=args.dataset_id), conn, table, embed)
    print(f'pgvector: {upserted} upserted, {deleted} deleted in table "{table}"')


if __name__ == '__main__':
    main()
