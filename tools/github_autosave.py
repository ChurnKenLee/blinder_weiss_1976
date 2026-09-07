#!/usr/bin/env python3
"""Snapshot a working tree to a dedicated branch without touching the user's index."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--branch', required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--notebook', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')

    def git(*words, input=None, allowed=(0,)):
        p = subprocess.run(['git', *words], cwd=repo, env=env, input=input,
                           text=True, capture_output=True, timeout=600)
        if p.returncode not in allowed:
            raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f'git {words[0]} failed')
        return p.stdout.strip()

    git('check-ref-format', 'refs/heads/' + args.branch)
    gitdir = Path(git('rev-parse', '--absolute-git-dir'))
    state = gitdir / 'marimo-autosave'
    state.mkdir(mode=0o700, exist_ok=True)
    lock = (state / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (state / 'pid').write_text(str(os.getpid()))
    env.update(GIT_INDEX_FILE=str(state / 'index'),
               GIT_AUTHOR_NAME='Marimo Autosave', GIT_AUTHOR_EMAIL='autosave@localhost',
               GIT_COMMITTER_NAME='Marimo Autosave', GIT_COMMITTER_EMAIL='autosave@localhost')
    status_file = state / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    status.update(pid=os.getpid(), branch=args.branch, repo=str(repo))
    stopping = False

    def now():
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def save(phase, **fields):
        status.update(phase=phase, updated_at=now(), **fields)
        temporary = state / 'status.tmp'
        temporary.write_text(json.dumps(status))
        temporary.replace(status_file)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True  # Allow the current commit/push to finish.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    ref = 'refs/heads/' + args.branch
    rc = 0
    while True:
        try:
            save('snapshotting', error=None)
            if any((gitdir / name).exists() for name in ['MERGE_HEAD', 'rebase-merge', 'rebase-apply']):
                raise RuntimeError('Paused while a Git merge or rebase is in progress.')
            head = git('rev-parse', 'HEAD')
            if args.notebook and args.notebook.exists():
                destination = repo / '.molab' / 'notebook.py'
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists() or args.notebook.read_bytes() != destination.read_bytes():
                    shutil.copy2(args.notebook, destination)
            previous = git('rev-parse', '--verify', ref, allowed=(0, 128)) or head
            git('read-tree', head)
            git('add', '-A', '--', '.')
            # Abort visibly instead of generating a commit GitHub cannot accept.
            objects = []
            for entry in git('ls-files', '--stage').splitlines():
                metadata, path = entry.split('\t', 1)
                mode, oid, stage = metadata.split()
                if mode != '160000':
                    objects.append((oid, path))
            if objects:
                sizes = git('cat-file', '--batch-check=%(objectsize)',
                            input='\n'.join(oid for oid, _ in objects) + '\n').splitlines()
                for (_, path), size in zip(objects, sizes, strict=True):
                    if int(size) > 95 * 1024 * 1024:
                        raise RuntimeError(f'File needs Git LFS tracking before backup: {path}')
            tree = git('write-tree')
            if git('rev-parse', 'HEAD') != head:
                raise RuntimeError('HEAD changed during snapshot; retrying next pass.')
            if tree != git('rev-parse', previous + '^{tree}'):
                commit = git('commit-tree', tree, '-p', previous,
                             input=f'Marimo autosave {now()}\n')
                git('update-ref', ref, commit, previous if previous != head else
                    git('rev-parse', '--verify', ref, allowed=(0, 128)) or '0' * 40)
            else:
                commit = previous
                if not git('rev-parse', '--verify', ref, allowed=(0, 128)):
                    git('update-ref', ref, commit, '0' * 40)
            save('pushing', local_commit=commit)
            git('push', '--porcelain', 'origin', ref + ':' + ref)
            remote = git('ls-remote', '--exit-code', 'origin', ref).split()[0]
            if remote != commit:
                raise RuntimeError('Remote commit differs from the snapshot; no force push attempted.')
            save('synced', remote_commit=remote, last_success=now(), error=None)
            rc = 0
        except Exception as error:
            save('error', error=str(error))
            rc = 1
        if args.once or stopping:
            break
        time.sleep(max(1, args.interval))
    if stopping:
        save('stopped')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
