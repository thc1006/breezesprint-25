"""Print GitHub metadata; write only with --apply after explicit maintainer invocation."""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Update an already-existing remote repository using your gh login.')
    args = parser.parse_args()
    project = json.loads((ROOT/'project.json').read_text())
    repo = project['repository'].removeprefix('https://github.com/').rstrip('/')
    if not re.fullmatch(r'[A-Za-z0-9-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Invalid GitHub repository in project.json')
    edits = [
        ('PATCH', f'repos/{repo}', {'description': project['description'], 'homepage': project['homepage']}),
        ('PUT', f'repos/{repo}/topics', {'names': project['keywords']}),
    ]
    print(json.dumps(dict(repository=repo, writes=edits, apply=args.apply), indent=2))
    if not args.apply:
        print('DRY RUN: no requests sent. Use --apply only after creating the intended repository.')
        return
    if not shutil.which('gh'):
        raise RuntimeError('Install and authenticate the GitHub CLI (gh) before applying metadata.')
    for method, endpoint, body in edits:
        subprocess.run(['gh', 'api', '--method', method, endpoint, '--input', '-'],
                       input=json.dumps(body), text=True, check=True)
    print('Repository description, homepage and topics updated. No code was committed or pushed.')


if __name__ == '__main__':
    main()
