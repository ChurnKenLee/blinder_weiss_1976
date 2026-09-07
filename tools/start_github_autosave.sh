#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
branch=${1:-marimo-autosave/sb-f20a5fc84ad154b5}
cd "$repo_dir"
gh auth status --hostname github.com
gh auth setup-git --hostname github.com
git lfs install --local
mkdir -p .git/marimo-autosave
python3 - "$repo_dir" "$branch" <<'LAUNCH'
import pathlib, subprocess, sys
repo=pathlib.Path(sys.argv[1])
with (repo/'.git/marimo-autosave/worker.log').open('a') as log:
    child=subprocess.Popen(
        ['python3', str(repo/'tools/github_autosave.py'), '--repo', str(repo),
         '--branch', sys.argv[2], '--interval', '5', '--notebook', '/marimo/notebook.py'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
print('Detached autosave worker started:', child.pid)
LAUNCH
echo "Verify .git/marimo-autosave/status.json reports a recent successful push."
