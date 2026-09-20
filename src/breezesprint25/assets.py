"""Download an explicit allow-list at an immutable Hugging Face model commit.

All bytes are checked against the content ETag from the pinned resolve URL:
Git-blob SHA1 for small repository files, SHA256 for LFS weight files. The trust
root is the official publisher served over HTTPS, not an unauthenticated mirror.
Actual SHA256 digests are recorded in the run manifest for later comparison.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re


def profile():
    return json.loads(Path(__file__).with_name('model.json').read_text(encoding='utf-8'))


def verify_file(path, etag):
    path = Path(path)
    etag = str(etag).strip('"')
    if not re.fullmatch(r'[a-fA-F0-9]{40}|[a-fA-F0-9]{64}', etag):
        raise RuntimeError('Publisher returned an unsupported content ETag; refusing unverified data.')
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
            if len(etag) == 40: git_digest.update(block)
    actual = digest.hexdigest() if len(etag) == 64 else git_digest.hexdigest()
    if actual.lower() != etag.lower():
        raise RuntimeError('Model asset content verification failed: ' + path.name)
    return digest.hexdigest()


def fetch_snapshot(preset, cache_dir, names=None):
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
    p = profile()
    if preset != p['preset']: raise ValueError('This notebook accepts only its bundled Breeze checkpoint.')
    if not re.fullmatch(r'[0-9a-f]{40}', p['revision']): raise ValueError('Model revision must be a full commit.')
    requested = list(p['files']) if names is None else list(names)
    if (not requested or any(not isinstance(name, str) or name not in p['files'] for name in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('Requested assets must be a nonempty unique subset of the pinned model allow-list.')
    files, checks = {}, {}
    def get(name):
        if Path(name).name != name or name.endswith(('.bin', '.pt', '.pkl')):
            raise ValueError('Unexpected model asset path.')
        url = hf_hub_url(p['model_id'], name, revision=p['revision'])
        metadata = get_hf_file_metadata(url, token=False, timeout=60)
        if metadata.commit_hash != p['revision']:
            raise RuntimeError('Resolved model commit does not match pinned revision.')
        path = Path(hf_hub_download(p['model_id'], name, revision=p['revision'],
                                   cache_dir=str(cache_dir), token=False))
        digest = verify_file(path, metadata.etag)
        if metadata.size is not None and path.stat().st_size != metadata.size:
            raise RuntimeError('Model asset size mismatch: ' + name)
        return name, path, {'sha256':digest, 'etag':metadata.etag, 'bytes':path.stat().st_size}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix='breeze-assets') as pool:
        for name, path, record in pool.map(get, requested): files[name], checks[name] = path, record
    return files, {'model_id':p['model_id'], 'revision':p['revision'],
                   'files_sha256':{n:r['sha256'] for n,r in checks.items()},
                   'files_integrity':checks}
