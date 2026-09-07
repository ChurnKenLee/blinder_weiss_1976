import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():

    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    github_sync_refresh = mo.ui.refresh(options=[5, 10, 30], default_interval=5, label="Refresh GitHub backup status")
    mo.vstack([mo.md("## GitHub autosave\nProject: `/marimo/blinder_weiss_1976`"), github_sync_refresh])
    return (github_sync_refresh,)


@app.cell(hide_code=True)
def _(github_sync_refresh, mo):
    github_sync_refresh.value
    import datetime as _dt
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    _file = _Path('/marimo/blinder_weiss_1976/.git/marimo-autosave/status.json')
    if _file.exists():
        _s = _json.loads(_file.read_text())
        try:
            _os.kill(_s['pid'], 0)
            _alive = True
        except ProcessLookupError:
            _alive = False
        _phase = _s['phase'] if _alive else 'worker not running'
        _branch = _s['branch']
        _commit = _s.get('remote_commit', '')
        _last = _s.get('last_success', 'No successful push yet')
        _message = f"**Status:** {_phase}\n\n**Last verified push (UTC):** {_last}\n\n**Backup branch:** [{_branch}](https://github.com/ChurnKenLee/blinder_weiss_1976/tree/{_branch})\n\n**Remote commit:** `{_commit[:12]}`"
        if _s.get('error'):
            _message += '\n\n**Push failed; newer files are not yet backed up.** Ask your pairing assistant to inspect the sync error.'
    else:
        _message = 'GitHub autosave is being configured.'
    mo.md(_message + '\n\nNew passes start five seconds after the previous pass finishes. A crash can lose unpushed writes. Data and outputs use Git LFS; credentials and local caches are excluded.')
    return


if __name__ == "__main__":
    app.run()
