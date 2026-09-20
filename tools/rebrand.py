"""Update local owner/repository URLs only. No network calls; no Git operations."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?', args.owner):
        raise ValueError('Invalid GitHub owner')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', args.repo):
        raise ValueError('Invalid GitHub repository name')
    project = json.loads((ROOT/'project.json').read_text())
    old = project['repository'].removeprefix('https://github.com/').rstrip('/')
    old_owner, old_repo = old.split('/')
    new = f'{args.owner}/{args.repo}'
    replacements = [(f'{old_owner}.github.io/{old_repo}', f'{args.owner}.github.io/{args.repo}'),
                    (old, new)]
    extensions = {'.md','.json','.html','.xml','.txt','.py','.toml','.yml','.yaml','.cff'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(part in {'.git','.pytest_cache','__pycache__','evidence','benchmarks'} for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text()
        changed = text
        for before, after in replacements:
            changed = changed.replace(before, after)
        if changed != text:
            path.write_text(changed)
    sums = ROOT/'SHA256SUMS.txt'
    if sums.exists():
        sums.unlink()  # a changed distribution must not keep stale release hashes
    subprocess.run([sys.executable, ROOT/'tools/build_notebook.py'], check=True)
    print(f'Local repository URLs changed to {new}. Rerun tests and tools/check_release.py before publishing.')
    print('No repository was created, and no commit or push was performed.')


if __name__ == '__main__':
    main()
