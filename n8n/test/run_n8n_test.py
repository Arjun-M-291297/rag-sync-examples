"""Execute n8n/rag-sync-qdrant.json in a real, isolated n8n against mock_services.py and check the results.

Usage: python n8n/test/run_n8n_test.py /path/to/n8n-binary
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 9988
BASE = f'http://127.0.0.1:{PORT}'


def state() -> dict:
    return json.loads(urllib.request.urlopen(f'{BASE}/_state').read())


def main() -> int:
    n8n = sys.argv[1]
    mock = subprocess.Popen([sys.executable, str(HERE / 'mock_services.py'), str(PORT)])
    home = Path(tempfile.mkdtemp(prefix='n8n-home-'))
    env = {**os.environ, 'N8N_USER_FOLDER': str(home), 'N8N_ENCRYPTION_KEY': 'local-test-only-key',
           'N8N_DIAGNOSTICS_ENABLED': 'false', 'N8N_VERSION_NOTIFICATIONS_ENABLED': 'false',
           'N8N_LOG_LEVEL': 'warn'}
    try:
        time.sleep(1)
        workflow = json.loads((HERE.parent / 'rag-sync-qdrant.json').read_text())
        workflow['id'] = 'ragSyncTestWf'
        for node in workflow['nodes']:
            if node['name'] == 'Config':
                for a in node['parameters']['assignments']['assignments']:
                    a['value'] = {'qdrantUrl': BASE, 'apifyApiBase': BASE, 'openAiApiBase': BASE,
                                  'embeddingModel': 'test-model', 'vectorSize': 4}.get(a['name'], a['value'])
        creds = [
            {'id': 'ragSyncApifyCred', 'name': 'Apify API token (header)', 'type': 'httpHeaderAuth',
             'data': {'name': 'Authorization', 'value': 'Bearer test-apify'}},
            {'id': 'ragSyncOpenAiCred', 'name': 'OpenAI API key (header)', 'type': 'httpHeaderAuth',
             'data': {'name': 'Authorization', 'value': 'Bearer test-openai'}},
            {'id': 'ragSyncQdrantCred', 'name': 'Qdrant API key (header)', 'type': 'httpHeaderAuth',
             'data': {'name': 'api-key', 'value': 'test-qdrant'}},
        ]
        (home / 'wf.json').write_text(json.dumps(workflow))
        (home / 'creds.json').write_text(json.dumps(creds))
        for cmd in (['import:credentials', f'--input={home / "creds.json"}'],
                    ['import:workflow', f'--input={home / "wf.json"}']):
            r = subprocess.run([n8n, *cmd], env=env, capture_output=True, text=True, timeout=300)
            print(f'$ n8n {cmd[0]}: exit {r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:]}')
            if r.returncode:
                return 1

        expectations = [
            ('run 1: 70 new chunks', 70, {'embeddings': 2}),
            ('run 2: 2 deletes + 1 new', 69, {'embeddings': 1}),
            ('run 3: no changes', 69, {'embeddings': 0}),
        ]
        ok = True
        for label, expected_points, expect in expectations:
            before = len([c for c in state()['calls'] if c.endswith('/v1/embeddings')])
            r = subprocess.run([n8n, 'execute', '--id=ragSyncTestWf'], env=env, capture_output=True, text=True,
                               timeout=600)
            s = state()
            embeds = len([c for c in s['calls'] if c.endswith('/v1/embeddings')]) - before
            checks = {
                'n8n exit 0': r.returncode == 0,
                f'{expected_points} points stored': len(s['points']) == expected_points,
                f'{expect["embeddings"]} embedding calls': embeds == expect['embeddings'],
                'no request errors': not s['errors'],
            }
            if label.startswith('run 2'):
                ids = set(s['points'].values())
                checks['deleted chunks gone'] = f'{1:024x}' not in ids and f'{2:024x}' not in ids
                checks['new chunk stored'] = 'abcdef' + '0' * 18 in ids
            passed = all(checks.values())
            ok &= passed
            print(f'{"PASS" if passed else "FAIL"} {label}: ' + ', '.join(f'{k}={"ok" if v else "NO"}' for k, v in checks.items()))
            if not passed:
                print('   stdout:', r.stdout[-1500:]); print('   stderr:', r.stderr[-1500:]); print('   errors:', s['errors'][:5])
        print('collection create calls (200 then 409 tolerated):', state()['collection_created'])
        return 0 if ok else 1
    finally:
        mock.terminate()


if __name__ == '__main__':
    sys.exit(main())
