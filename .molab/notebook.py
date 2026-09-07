# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "basedpyright>=1.39.9",
#     "diffrax>=0.7.2",
#     "jax[cuda13]==0.11.0",
#     "numpy==2.5.3",
# ]
# ///

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
            _proc_stat = _Path('/proc') / str(_s['pid']) / 'stat'
            _alive = _proc_stat.exists() and _proc_stat.read_text().split()[2] != 'Z' 
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GPU policy solver

    We are improving repeated calibration solves and checking policy smoothness.
    The charts below show saved, completed GPU runs. Feasibility checks and
    convergence comparisons remain separate from speed measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import os as _key_os
    import tempfile as _key_tempfile

    def _save_ipums_key(_key_value):
        if not _key_value or not _key_value.strip():
            return
        with _key_tempfile.NamedTemporaryFile(
            mode="w", dir="/tmp", prefix=".ipums-key-", delete=False
        ) as _key_file:
            _key_os.fchmod(_key_file.fileno(), 0o600)
            _key_file.write(_key_value.strip())
            _key_path = _key_file.name
        _key_os.replace(_key_path, "/tmp/ipums_api_key")
        mo.status.toast("IPUMS key received privately. The data agent can start downloading.")

    ipums_private_key_form = mo.ui.text(
        kind="password", label="IPUMS API key"
    ).form(
        clear_on_submit=True,
        on_change=_save_ipums_key,
        submit_button_label="Enable IPUMS downloads",
    )
    mo.vstack([
        mo.md("### IPUMS access\nEnter the key here to enable ACS/ATUS downloads. "
              "It will not be included in notebook source or GitHub backups."),
        ipums_private_key_form,
    ])
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    benchmark_directory = Path("/marimo/blinder_weiss_1976/output/solver_benchmarks")
    return benchmark_directory, json, np, plt


@app.cell(hide_code=True)
def _(mo):
    policy_age = mo.ui.slider(0, 69, value=25, label="Model age")
    policy_human_capital = mo.ui.slider(0.25, 3.0, step=0.05, value=1.0, label="/sHuman capital")
    mo.hstack([policy_age, policy_human_capital])
    return policy_age, policy_human_capital


@app.cell(hide_code=True)
def _(benchmark_directory, github_sync_refresh, json, mo):
    github_sync_refresh.value
    _timing_rows = []
    _baseline_file = benchmark_directory / "original_timing.json"
    if _baseline_file.exists():
        _baseline = json.loads(_baseline_file.read_text())
        for _row in _baseline["results"]:
            _timing_rows.append({"solver": "Original, default grid", "call": _row["label"],
                                 "seconds": round(_row["wall_seconds"], 3)})
    for _label, _folder in [("Optimized, default grid", "optimized_default"),
                            ("Optimized, finer grid", "optimized_fine"),
                            ("Experimental PCHIP", "pchip_default")]:
        _file = benchmark_directory / _folder / "report.json"
        if _file.exists():
            _report = json.loads(_file.read_text())
            for _row in _report["results"]:
                _timing_rows.append({"solver": _label, "call": _row["call"],
                                     "seconds": round(_row["wall_seconds"], 3)})
    mo.vstack([mo.md("**Synchronized GPU timings, including output transfer.** "
                     "The first call includes compilation; later calls vary a model parameter."),
               mo.ui.table(_timing_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(
    benchmark_directory,
    github_sync_refresh,
    mo,
    np,
    plt,
    policy_age,
    policy_human_capital,
):
    github_sync_refresh.value
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for _label, _relative, _style in [
        ("Original", "original_default.npz", "--"),
        ("Optimized + consumption polish", "optimized_default/policies.npz", "-"),
        ("Experimental PCHIP", "pchip_default/policies.npz", ":"),
    ]:
        _file = benchmark_directory / _relative
        if not _file.exists():
            continue
        with np.load(_file) as _data:
            _age_index = min(int(policy_age.value * _data["c"].shape[0] / 70),
                             _data["c"].shape[0] - 1)
            _log_k = np.log(policy_human_capital.value)
            for _ax, _key, _title in zip(_axes, ["c", "h", "q"],
                                        ["Consumption", "Active time", "Training time"], strict=True):
                _slice = np.array([np.interp(_log_k, _data["y"], _line)
                                   for _line in _data[_key][_age_index]])
                _ax.plot(_data["A"], _slice, _style, label=_label)
                _ax.set(xlabel="Assets", title=_title, xlim=(0, 20))
                _ax.grid(alpha=0.2)
    _axes[0].legend(fontsize=7)
    mo.vstack([mo.md("**Policies at the selected age and human capital.** "
                     "These curves connect stored policy nodes; they are not post-smoothed."),
               _fig])
    return


if __name__ == "__main__":
    app.run()
