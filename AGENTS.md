# Temporary GPU environment

The user has authorized continuous project snapshots to the GitHub autosave
branch configured by tools/start_github_autosave.sh. Check
.git/marimo-autosave/status.json before substantial work and after important
changes. A local commit is not a completed backup. Never force-push to fix an
autosave rejection; preserve both histories and investigate.

The autosave service uses a separate index and branch. Normal commits and
staging remain available. Project data, outputs, papers, and checkpoint formats
listed in .gitattributes use Git LFS. Keep credentials out of tracked files.
Write completed checkpoints with an atomic rename where possible.

Read code/README.md and code/jax/README.md before changing solver behavior.
Edit the active live marimo notebook through marimo._code_mode using the
marimo-pair skill; do not write its source file directly. Reusable project
modules can be edited normally. Coordinate live notebook writes across agents.
