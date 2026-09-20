"""Offline release integrity checks; never contacts GitHub or allocates a TPU."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check():
    project = json.loads((ROOT/'project.json').read_text())
    nb = json.loads((ROOT/'BreezeSprint25.ipynb').read_text())
    assert len(nb['cells']) == 2, 'The notebook must contain exactly two cells.'
    payload = {}
    for cell in nb['cells']:
        assert cell['cell_type'] == 'code'
        assert not cell['outputs'] and cell['execution_count'] is None
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {'PAYLOAD_B64', 'PAYLOAD_SHA256', 'PAYLOAD_FILES'}:
                    payload[name] = ast.literal_eval(node.value)
    data = base64.b64decode(payload['PAYLOAD_B64'], validate=True)
    assert hashlib.sha256(data).hexdigest() == payload['PAYLOAD_SHA256']
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == sorted(payload['PAYLOAD_FILES'])
        assert len(z.namelist()) == len(set(z.namelist()))
        for name in z.namelist():
            assert z.read(name) == (ROOT/name).read_bytes(), f'Stale embedded source: {name}'
    model = json.loads((ROOT/'src/breezesprint25/model.json').read_text())
    assert model['model_id'] == project['capabilities']['model']
    assert re.fullmatch(r'[0-9a-f]{40}', model['revision'])
    assert model['language_token']=='zh'
    assert model['weight_files'] and set(model['weight_files']).issubset(model['files'])
    assert not any(name.endswith(('.bin','.pkl','.pt')) for name in model['files'])
    assert 'tokenizer.json' not in model['files']
    assert hashlib.sha256((ROOT/'src/breezesprint25/core.py').read_bytes()).hexdigest() == (
        '8f1838a6457ebf952fdf6cbf6260662120beaf26c2db6d119a69630ed0332769')
    assert project['capabilities']['code_cells'] == 2
    assert project['capabilities']['formats'] == ['txt']
    assert project['capabilities']['input_ui'] == 'upload-only'
    assert project['capabilities']['automatic_download'] is True
    assert project['capabilities']['result_archive'] is False
    user_cell = ''.join(nb['cells'][1]['source'])
    assert 'upload_and_transcribe(sprint)' in user_cell
    assert '#@param' not in user_cell
    for removed in ['INPUT_PATH', 'MOUNT_DRIVE', 'OUTPUT_DIR', 'AUTO_DOWNLOAD', 'files.download', 'diagnostic_zip']:
        assert removed not in user_cell, f'Stale user-facing option: {removed}'
    controller = (ROOT/'src/breezesprint25/colab.py').read_text()
    assert 'import zipfile' not in controller and 'def archive(' not in controller
    assert 'def diagnostics(' not in controller
    topics = project['keywords']
    assert len(topics) <= 20 and len(topics) == len(set(topics))
    assert all(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}', t) for t in topics)
    for name in ['README.md','README.zh-TW.md','PUBLISH.md','LICENSE','THIRD_PARTY_NOTICES.md',
                 'AGENTS.md','llms.txt','CITATION.cff','docs/index.html','.github/workflows/ci.yml']:
        assert (ROOT/name).is_file(), f'Missing release file: {name}'
    for path in ROOT.glob('**/*.py'):
        if '.git' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    assert 'colab.research.google.com/github/' in (ROOT/'README.md').read_text()
    print('PASS: two clean cells, exact embedded sources, one pinned model, unchanged core, metadata and Python syntax.')
    return True


if __name__ == '__main__':
    check()
