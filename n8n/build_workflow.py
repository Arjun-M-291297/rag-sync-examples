"""Generates rag-sync-qdrant.json (the importable n8n workflow). Run: python n8n/build_workflow.py"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NS = uuid.UUID('9a0c7f6e-2b1d-4e8a-b3c5-71f0d2a4e6b8')


def node_id(name: str) -> str:
    return str(uuid.uuid5(NS, name))


CRED = {
    'apify': {'id': 'ragSyncApifyCred', 'name': 'Apify API token (header)'},
    'openai': {'id': 'ragSyncOpenAiCred', 'name': 'OpenAI API key (header)'},
    'qdrant': {'id': 'ragSyncQdrantCred', 'name': 'Qdrant API key (header)'},
}
CFG = "$('Config').first().json"


def http(name: str, pos: list[int], method: str, url: str, body: str, cred: str, extra_options: dict | None = None):
    return {
        'id': node_id(name), 'name': name, 'type': 'n8n-nodes-base.httpRequest', 'typeVersion': 4.2, 'position': pos,
        'parameters': {
            'method': method, 'url': url,
            'authentication': 'genericCredentialType', 'genericAuthType': 'httpHeaderAuth',
            'sendBody': True, 'specifyBody': 'json', 'jsonBody': body,
            'options': {'timeout': 300000, **(extra_options or {})},
        },
        'credentials': {'httpHeaderAuth': CRED[cred]},
    }


SPLIT_JS = r'''// RAG Sync Monitor returns one record per change: {op: "upsert" | "delete", chunk_id, url, text, ...}.
// Chunk IDs are hex; Qdrant needs UUIDs, so pad them into UUID form (same mapping as sync_qdrant.py).
const toPointId = (id) => {
  const h = String(id).toLowerCase();
  if (!/^[0-9a-f]{1,32}$/.test(h)) throw new Error(`Unexpected chunk_id: ${id}`);
  const p = h.padEnd(32, '0');
  return `${p.slice(0, 8)}-${p.slice(8, 12)}-${p.slice(12, 16)}-${p.slice(16, 20)}-${p.slice(20)}`;
};
const records = $input.all().map((item) => item.json).filter((r) => r && r.op);
const deletes = records.filter((r) => r.op === 'delete').map((r) => toPointId(r.chunk_id));
const upserts = records.filter((r) => r.op === 'upsert');
const BATCH = 64; // texts per embedding request
const out = [];
if (deletes.length) out.push({ json: { kind: 'delete', ids: deletes } });
for (let i = 0; i < upserts.length; i += BATCH) {
  const batch = upserts.slice(i, i + BATCH);
  out.push({ json: {
    kind: 'upsert',
    texts: batch.map((u) => u.text),
    points: batch.map((u) => ({
      id: toPointId(u.chunk_id),
      payload: {
        chunk_id: u.chunk_id, text: u.text, url: u.url, title: u.title || '',
        section: u.heading_path || '', chunk_index: u.chunk_index || 0,
        source_type: u.source_type || 'html', synced_at: u.synced_at || '',
      },
    })),
  } });
}
return out;'''

ATTACH_JS = r'''// Pair each embedding batch with the points it was made for.
const batch = $('Deletes or upserts?').item.json;
const vectors = [...$json.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);
if (vectors.length !== batch.points.length) throw new Error('Embedding count does not match the batch');
return { json: { points: batch.points.map((p, i) => ({ ...p, vector: vectors[i] })) } };'''

NOTE = '''## RAG Sync Monitor → Qdrant

Keeps a Qdrant collection in sync with a docs site. Each run only embeds and writes the chunks that **changed**; removed content is deleted.

**Setup (5 min)**
1. Edit **Config**: `siteUrl`, `stateKey` (one per site), `qdrantUrl`, `collection`.
2. Create 3 *Header Auth* credentials:
   - **Apify**: name `Authorization`, value `Bearer <APIFY_TOKEN>`
   - **OpenAI**: name `Authorization`, value `Bearer <OPENAI_API_KEY>`
   - **Qdrant**: name `api-key`, value `<QDRANT_API_KEY>` (any value for a local Qdrant without auth)
3. Click **Test workflow**. Run it again: only changes come through.
4. Activate to run daily.

The Actor call waits up to 5 min, which suits sites up to a few hundred pages per run (raise `maxPages` gradually; unchanged pages are cheap).
Uses `text-embedding-3-small` (1536 dims). If you change the model, change `vectorSize` and use a new collection.'''


def build() -> dict:
    nodes = [
        {'id': node_id('Note'), 'name': 'Read me', 'type': 'n8n-nodes-base.stickyNote', 'typeVersion': 1,
         'position': [-560, -380], 'parameters': {'content': NOTE, 'height': 520, 'width': 460}},
        {'id': node_id('Manual'), 'name': 'Run manually', 'type': 'n8n-nodes-base.manualTrigger', 'typeVersion': 1,
         'position': [0, 0], 'parameters': {}},
        {'id': node_id('Schedule'), 'name': 'Every day', 'type': 'n8n-nodes-base.scheduleTrigger', 'typeVersion': 1.2,
         'position': [0, 200],
         'parameters': {'rule': {'interval': [{'field': 'days', 'daysInterval': 1, 'triggerAtHour': 3}]}}},
        {'id': node_id('Config'), 'name': 'Config', 'type': 'n8n-nodes-base.set', 'typeVersion': 3.4,
         'position': [240, 100],
         'parameters': {'mode': 'manual', 'includeOtherFields': False, 'options': {}, 'assignments': {'assignments': [
             {'id': node_id(f'cfg-{k}'), 'name': k, 'value': v, 'type': t} for k, v, t in [
                 ('siteUrl', 'https://docs.example.com/', 'string'),
                 ('stateKey', 'example-docs', 'string'),
                 ('maxPages', 200, 'number'),
                 ('qdrantUrl', 'http://localhost:6333', 'string'),
                 ('collection', 'docs', 'string'),
                 ('embeddingModel', 'text-embedding-3-small', 'string'),
                 ('vectorSize', 1536, 'number'),
                 ('apifyApiBase', 'https://api.apify.com', 'string'),
                 ('openAiApiBase', 'https://api.openai.com', 'string'),
             ]]}}},
        http('Ensure Qdrant collection', [480, 100], 'PUT',
             f"={{{{ {CFG}.qdrantUrl }}}}/collections/{{{{ {CFG}.collection }}}}",
             f"={{{{ JSON.stringify({{ vectors: {{ size: {CFG}.vectorSize, distance: 'Cosine' }} }}) }}}}", 'qdrant',
             # "already exists" is fine; any other problem shows up on the next Qdrant call
             {'response': {'response': {'neverError': True}}}),
        http('Run RAG Sync Monitor', [720, 100], 'POST',
             f"={{{{ {CFG}.apifyApiBase }}}}/v2/acts/cipherwise~rag-sync-monitor/run-sync-get-dataset-items?timeout=290",
             f"={{{{ JSON.stringify({{ startUrls: [{{ url: {CFG}.siteUrl }}], stateKey: {CFG}.stateKey, "
             f"maxPages: {CFG}.maxPages }}) }}}}", 'apify'),
        {'id': node_id('Split'), 'name': 'Split changes', 'type': 'n8n-nodes-base.code', 'typeVersion': 2,
         'position': [960, 100], 'parameters': {'mode': 'runOnceForAllItems', 'jsCode': SPLIT_JS}},
        {'id': node_id('If'), 'name': 'Deletes or upserts?', 'type': 'n8n-nodes-base.if', 'typeVersion': 2.2,
         'position': [1200, 100], 'parameters': {'options': {}, 'conditions': {
             'options': {'caseSensitive': True, 'leftValue': '', 'typeValidation': 'strict', 'version': 2},
             'combinator': 'and',
             'conditions': [{'id': node_id('if-kind'), 'leftValue': '={{ $json.kind }}', 'rightValue': 'delete',
                             'operator': {'type': 'string', 'operation': 'equals'}}]}}},
        http('Qdrant: delete', [1440, -20], 'POST',
             f"={{{{ {CFG}.qdrantUrl }}}}/collections/{{{{ {CFG}.collection }}}}/points/delete?wait=true",
             '={{ JSON.stringify({ points: $json.ids }) }}', 'qdrant'),
        http('OpenAI: embed', [1440, 220], 'POST',
             f"={{{{ {CFG}.openAiApiBase }}}}/v1/embeddings",
             f"={{{{ JSON.stringify({{ model: {CFG}.embeddingModel, input: $json.texts }}) }}}}", 'openai'),
        {'id': node_id('Attach'), 'name': 'Attach vectors', 'type': 'n8n-nodes-base.code', 'typeVersion': 2,
         'position': [1680, 220], 'parameters': {'mode': 'runOnceForEachItem', 'jsCode': ATTACH_JS}},
        http('Qdrant: upsert', [1920, 220], 'PUT',
             f"={{{{ {CFG}.qdrantUrl }}}}/collections/{{{{ {CFG}.collection }}}}/points?wait=true",
             '={{ JSON.stringify({ points: $json.points }) }}', 'qdrant'),
    ]
    link = lambda target: [{'node': target, 'type': 'main', 'index': 0}]  # noqa: E731
    connections = {
        'Run manually': {'main': [link('Config')]},
        'Every day': {'main': [link('Config')]},
        'Config': {'main': [link('Ensure Qdrant collection')]},
        'Ensure Qdrant collection': {'main': [link('Run RAG Sync Monitor')]},
        'Run RAG Sync Monitor': {'main': [link('Split changes')]},
        'Split changes': {'main': [link('Deletes or upserts?')]},
        'Deletes or upserts?': {'main': [link('Qdrant: delete'), link('OpenAI: embed')]},
        'OpenAI: embed': {'main': [link('Attach vectors')]},
        'Attach vectors': {'main': [link('Qdrant: upsert')]},
    }
    return {'name': 'RAG Sync Monitor → Qdrant (only changed chunks)', 'nodes': nodes, 'connections': connections,
            'settings': {'executionOrder': 'v1'}, 'pinData': {}, 'meta': {'templateCredsSetupCompleted': False}}


if __name__ == '__main__':
    out = Path(__file__).with_name('rag-sync-qdrant.json')
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + '\n')
    print(f'wrote {out}')
