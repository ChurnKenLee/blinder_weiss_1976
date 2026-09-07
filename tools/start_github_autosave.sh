#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
branch=${1:-marimo-autosave/sb-f20a5fc84ad154b5}
cd "$repo_dir"
gh auth status --hostname github.com
gh auth setup-git --hostname github.com
git lfs install --local
mkdir -p .git/marimo-autosave
nohup python3 tools/github_autosave.py --repo "$repo_dir" --branch "$branch" --interval 5 --notebook /marimo/notebook.py > .git/marimo-autosave/worker.log 2>&1 < /dev/null &
echo "Autosave launched. Verify .git/marimo-autosave/status.json reports synced."
