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
    # Blinder–Weiss policies and population dynamics

    We are reducing numerical policy jitter and building a deterministic continuum
    population for calibration. The comparisons below retain the independent direct
    solutions and distinguish numerical validation from empirical estimation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **2024 IPUMS downloads validated and backed up to GitHub.** ACS: 3,422,888 person records. ATUS: 7,669 respondent diaries; every diary totals 1,440 minutes. Raw data, dictionaries, checksums, and provenance are in `blinder_weiss_1976/data/ipums/`. Empirical target definitions remain to be set; the calibration pilot uses synthetic targets.
    """)
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
    policy_human_capital = mo.ui.slider(0.05, 5.0, step=0.05, value=1.0, label="Human capital")
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
                            ("Cubic + adaptive search, default grid", "bicubic_adaptive_default"),
                            ("Cubic + exact propagation, finer grid", "bicubic_exact_fine"),
                            ("Cubic, wider human-capital domain", "bicubic_padded_fine")]:
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
        ("Cubic, wider domain", "bicubic_padded_fine/policies.npz", ":"),
    ]:
        _file = benchmark_directory / _relative
        if not _file.exists():
            continue
        with np.load(_file) as _data:
            _age_index = min(int(policy_age.value * _data["c"].shape[0] / 70),
                             _data["c"].shape[0] - 1)
            _log_k = np.log(policy_human_capital.value)
            if not _data["y"][0] <= _log_k <= _data["y"][-1]:
                continue
            for _ax, _key, _title in zip(_axes, ["c", "h", "q"],
                                        ["Consumption", "Active time", "Training time"], strict=True):
                _slice = np.array([np.interp(_log_k, _data["y"], _line)
                                   for _line in _data[_key][_age_index]])
                _ax.plot(_data["A"], _slice, _style, label=_label)
                _ax.set(xlabel="Assets", title=_title, xlim=(0, 20))
                _ax.grid(alpha=0.2)
    _axes[0].legend(fontsize=7)
    plt.close(_fig)
    mo.vstack([mo.md("**Policies at the selected age and human capital.** "
                     "These curves connect stored policy nodes. A solver is omitted when the selected state is outside its grid."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    reference_household = mo.ui.dropdown(
        options=["A = 5, K = 1", "A = 2, K = 0.8", "A = 8, K = 1.2"],
        value="A = 8, K = 1.2", label="Initial assets and human capital",
    )
    reference_household
    return (reference_household,)


@app.cell(hide_code=True)
def _(benchmark_directory, mo, np, plt, reference_household):
    _probe = ["A = 5, K = 1", "A = 2, K = 0.8", "A = 8, K = 1.2"].index(reference_household.value)
    _reference_name = ["direct_reference", "direct_type_low", "direct_type_high"][_probe]
    _comparison_file = benchmark_directory / "calibration_domain_validation.npz"
    _comparison_fig, _comparison_axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    if _comparison_file.exists():
        with np.load(_comparison_file) as _paths:
            for _run, _label, _style in [(0, "Previous domain", "--"), (1, "Wider domain", "-")]:
                _time = _paths[f"run_{_run}_time"]
                _states = _paths[f"run_{_run}_states"][:, 256 + _probe]
                _controls = _paths[f"run_{_run}_controls"][:, 256 + _probe]
                _comparison_axes[0].plot(_time, np.exp(_states[:, 1]), _style, label=_label)
                _comparison_axes[1].plot(_time[:-1], _controls[:, 1], _style, label=_label)
                _comparison_axes[2].plot(_time[:-1], _controls[:, 0], _style, label=_label)
        with np.load(benchmark_directory / f"{_reference_name}.npz") as _direct:
            for _axis, _field in zip(_comparison_axes, ["K", "h", "c"], strict=True):
                _axis.plot(_direct["time"], _direct[_field], color="black", ls=":", label="Direct reference")
    for _axis, _title in zip(_comparison_axes, ["Human capital", "Active time", "Consumption"], strict=True):
        _axis.set(title=_title, xlabel="Model age")
        _axis.grid(alpha=0.2)
    _comparison_axes[0].legend(fontsize=8)
    plt.close(_comparison_fig)
    mo.vstack([
        mo.md("**Lifecycle check against independent direct solutions.** The earlier human-capital floor forced some retired households to train. The wider grid covers their falling human capital. These comparisons are part of ongoing convergence checks."),
        _comparison_fig,
    ])
    return


@app.cell(hide_code=True)
def continuum_intro(mo):
    mo.md(r"""
    ## Continuum distribution and calibration pilot

    Quadrature integrates a specified initial distribution using deterministic,
    weighted nodes. Forward transport moves probability mass across the state grid
    as each cohort ages. Both use the same feasible policies and lifecycle dynamics.

    The initial distribution and pilot targets are **synthetic**. Assets at the
    numerical floor have a separate mass component; nearby interior mass is tracked
    separately. Conservation and agreement under refinement are separate checks.
    """)
    return


if __name__ == "__main__":
    app.run()
