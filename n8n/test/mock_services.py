"""Local stand-ins for the Apify, OpenAI and Qdrant APIs, used to execute the n8n workflow end to end.

Checks every request's auth header and body shape, keeps a real in-memory point store, and serves three scripted
Actor runs: 70 new chunks (batching), then 2 deletes + 1 upsert, then no changes.
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
URL = 'https://docs.example.com/guide'


def up(cid: str, text: str) -> dict:
    return {'op': 'upsert', 'chunk_id': cid, 'url': URL, 'title': 'Guide', 'heading_path': 'Guide > Install',
            'chunk_index': 0, 'text': text, 'source_type': 'html', 'synced_at': '2026-10-19T10:00:00+00:00'}


RUN_1 = [up(f'{i:024x}', f'Chunk {i}') for i in range(1, 71)]
RUN_2 = [{'op': 'delete', 'chunk_id': f'{1:024x}', 'url': URL, 'reason': 'removed_from_page'},
         {'op': 'delete', 'chunk_id': f'{2:024x}', 'url': URL, 'reason': 'page_gone'},
         up('abcdef' + '0' * 18, 'Brand new chunk')]
RUNS = [RUN_1, RUN_2, []]
STATE = {'points': {}, 'runs': 0, 'collection_created': 0, 'calls': [], 'errors': []}
AUTH = {'apify': ('authorization', 'Bearer test-apify'), 'openai': ('authorization', 'Bearer test-openai'),
        'qdrant': ('api-key', 'test-qdrant')}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get('content-length') or 0)
        return json.loads(self.rfile.read(length) or b'{}')

    def _auth(self, service: str) -> bool:
        header, value = AUTH[service]
        if self.headers.get(header) != value:
            STATE['errors'].append(f'{self.command} {self.path}: bad {service} auth {self.headers.get(header)!r}')
            self._send(401, {'error': 'unauthorized'})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == '/_state':
            self._send(200, {**STATE, 'points': {k: v['payload']['chunk_id'] for k, v in STATE['points'].items()}})
        else:
            self._send(404, {})

    def do_POST(self) -> None:  # noqa: N802
        STATE['calls'].append(f'POST {self.path.split("?")[0]}')
        if self.path.startswith('/v2/acts/cipherwise~rag-sync-monitor/run-sync-get-dataset-items'):
            if not self._auth('apify'):
                return
            body = self._body()
            if not (body.get('startUrls') and body.get('stateKey')):
                STATE['errors'].append(f'bad actor input {body}')
            run = RUNS[min(STATE['runs'], len(RUNS) - 1)]
            STATE['runs'] += 1
            self._send(201, run)
        elif self.path == '/v1/embeddings':
            if not self._auth('openai'):
                return
            body = self._body()
            texts = body.get('input') or []
            if not isinstance(texts, list) or not texts or len(texts) > 64 or body.get('model') != 'test-model':
                STATE['errors'].append(f'bad embeddings request: model={body.get("model")} n={len(texts)}')
            data = [{'index': i, 'embedding': [float(len(t)), float(i), 0.5, 1.0]} for i, t in enumerate(texts)]
            self._send(200, {'data': list(reversed(data)), 'model': 'test-model'})  # out of order on purpose
        elif self.path.startswith('/collections/docs/points/delete'):
            if not self._auth('qdrant'):
                return
            for pid in self._body().get('points', []):
                if not UUID_RE.match(pid):
                    STATE['errors'].append(f'bad point id {pid}')
                STATE['points'].pop(pid, None)
            self._send(200, {'result': {'status': 'completed'}, 'status': 'ok'})
        else:
            self._send(404, {})

    def do_PUT(self) -> None:  # noqa: N802
        STATE['calls'].append(f'PUT {self.path.split("?")[0]}')
        if not self._auth('qdrant'):
            return
        if self.path.split('?')[0] == '/collections/docs':
            body = self._body()
            if body != {'vectors': {'size': 4, 'distance': 'Cosine'}}:
                STATE['errors'].append(f'bad collection body {body}')
            if STATE['collection_created']:
                self._send(409, {'status': {'error': 'Collection `docs` already exists!'}})
            else:
                STATE['collection_created'] += 1
                self._send(200, {'result': True, 'status': 'ok'})
        elif self.path.startswith('/collections/docs/points'):
            for point in self._body().get('points', []):
                if not UUID_RE.match(point['id']) or len(point.get('vector', [])) != 4:
                    STATE['errors'].append(f'bad point {point.get("id")}')
                if point['vector'][0] != float(len(point['payload']['text'])):
                    STATE['errors'].append(f'vector paired with wrong text for {point["payload"]["chunk_id"]}')
                STATE['points'][point['id']] = point
            self._send(200, {'result': {'status': 'completed'}, 'status': 'ok'})
        else:
            self._send(404, {})

    def log_message(self, *args: object) -> None:
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
