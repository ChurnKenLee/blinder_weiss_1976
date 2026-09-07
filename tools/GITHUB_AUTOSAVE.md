# GitHub autosave

Run `tools/start_github_autosave.sh` after authenticating with `gh auth login`.
Snapshots are pushed to `marimo-autosave/sb-f20a5fc84ad154b5` in origin. The service
uses its own Git index and branch, leaving the current branch, staging area,
and normal commits available for project work. It never force-pushes or merges
remote changes automatically.

A new pass begins five seconds after the previous pass finishes. The active
`/marimo/notebook.py` is copied into `.molab/notebook.py`. Only files inside this
repository, plus that notebook copy, are backed up. Secrets, local environments,
and caches remain excluded by `.gitignore`. Data, outputs, and papers use Git
LFS; GitHub LFS size limits and account storage/bandwidth quotas still apply.
Write checkpoints as completed files, preferably using an atomic rename.

Read `.git/marimo-autosave/status.json` for the last remotely verified commit,
last successful push time, and any error. `synced` means that snapshot reached
GitHub; newer writes may still be pending. A crash can lose any unpushed changes.
Authorization failures, GitHub rejection, and network failures are reported and
retried; a stopped container cannot continue pushing. Reauthorize and restart
the service after recreating this temporary environment.

To stop, send SIGTERM to the PID in `.git/marimo-autosave/pid`. The current pass
finishes before exit. Restore work by cloning the autosave branch with Git LFS
installed (`git clone --branch <branch> <repository>` and `git lfs pull`).
