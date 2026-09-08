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
    return Path, benchmark_directory, json, np, plt


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
        ("Refined assets (121 nodes)", "bicubic_asset121_fine/policies.npz", "-"),
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


@app.cell
def population_imports(Path, json, np):
    import sys

    _project = Path("/marimo/blinder_weiss_1976")
    if str(_project / "code/jax") not in sys.path:
        sys.path.insert(0, str(_project / "code/jax"))

    from blinder_weiss import (
        BellmanConfig, BellmanSolution, ModelParams,
        InitialAtom, SyntheticInitialDistribution, initial_quadrature,
        simulate_population, AgeMomentTarget, CalibrationTargets,
        MOMENT_UNITS, weighted_age_moment_loss,
    )


    def load_population_policy(folder):
        """Read one completed policy checkpoint with its exact numerical settings."""
        import jax
        report = json.loads((folder / "report.json").read_text())
        config = BellmanConfig(**report["config"])
        device = jax.devices(config.compute_platform)[config.device_index]
        with np.load(folder / "policies.npz") as arrays:
            return BellmanSolution(
                params=ModelParams(**report["params"]), config=config,
                time=arrays["time"], asset_grid=arrays["A"],
                log_human_capital_grid=arrays["y"], values=arrays["values"],
                consumption_policy=arrays["c"], hours_policy=arrays["h"],
                training_time_policy=arrays["q"], solve_seconds=0.0,
                backend=device.platform,
                device=f"{device.platform}:{device.id} ({device.device_kind})",
            )


    return (
        SyntheticInitialDistribution,
        initial_quadrature,
        load_population_policy,
        simulate_population,
    )


@app.cell
def population_controls(mo):
    population_settings = mo.ui.dictionary({
        "backend": mo.ui.dropdown(
            options={"Quadrature reference": "quadrature", "Conservative transport": "transport"},
            value="Quadrature reference", label="Population method",
        ),
        "quadrature_order": mo.ui.dropdown(
            options={"8 × 8": 8, "16 × 16": 16, "32 × 32": 32, "64 × 64": 64},
            value="16 × 16", label="Initial quadrature",
        ),
        "transport_resolution": mo.ui.dropdown(
            options={"31 × 31": 31, "61 × 61": 61, "121 × 121": 121},
            value="61 × 61", label="Forward grid (transport only)",
        ),
        "correlation": mo.ui.slider(-0.3, 0.3, step=0.05, value=0.25,
                                    label="Initial Corr(A, log K), interior component"),
        "floor_mass": mo.ui.slider(0.0, 0.2, step=0.01, value=0.05,
                                   label="Initial probability at numerical asset floor"),
    }).form(submit_button_label="Simulate population", show_clear_button=False)
    population_settings
    return (population_settings,)


@app.cell
def run_population(
    SyntheticInitialDistribution,
    benchmark_directory,
    initial_quadrature,
    load_population_policy,
    mo,
    np,
    population_settings,
    simulate_population,
):
    mo.stop(population_settings.value is None, mo.md("Choose settings and submit to simulate a cohort's mass through its full lifecycle."))
    from blinder_weiss.distribution import DistributionGrid

    _population_inputs = population_settings.value
    population_solution = load_population_policy(benchmark_directory / "bicubic_asset121_fine")
    _population_config = population_solution.config
    population_law = SyntheticInitialDistribution(
        correlation=_population_inputs["correlation"],
        asset_floor_mass=_population_inputs["floor_mass"],
    )
    population_initial_nodes = initial_quadrature(
        population_law, nodes_per_dimension=_population_inputs["quadrature_order"],
        asset_floor=_population_config.asset_minimum,
    )
    _population_n = _population_inputs["transport_resolution"]
    population_grid = DistributionGrid(
        _population_config.asset_minimum,
        _population_config.asset_minimum
        + (_population_config.asset_maximum - _population_config.asset_minimum)
          * np.linspace(0.0, 1.0, _population_n)[1:] ** _population_config.asset_grid_curvature,
        np.linspace(_population_config.log_human_capital_minimum,
                    _population_config.log_human_capital_maximum, _population_n),
    )
    population_result = simulate_population(
        population_solution, population_initial_nodes, backend=_population_inputs["backend"],
        participation_hours_threshold=0.02, distribution_grid=population_grid,
        store_snapshots=True,
    )
    mo.md(f"**Completed {population_result.backend}:** "
          f"{len(population_result.state_moments.time) - 1} decision periods; "
          f"maximum mass drift {population_result.diagnostics['maximum_mass_drift']:.2e}. "
          "The initial point atom at (A, K) = (5, 1) has probability 0.05; "
          "the remaining probability is continuous after accounting for the selected floor mass.")
    return (population_result,)


@app.cell(hide_code=True)
def population_age_control(mo):
    population_age = mo.ui.slider(0, 70, step=0.5, value=25,
                                  label="Model age for distribution view")
    population_age
    return (population_age,)


@app.cell(hide_code=True)
def population_distribution_view(
    mo,
    np,
    plt,
    population_age,
    population_result,
):
    _period = int(np.argmin(np.abs(population_result.state_moments.time - population_age.value)))
    _figure, _axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    if population_result.backend == "transport":
        _mass = population_result.simulation.masses[_period]
        _grid = population_result.simulation.grid
        _log_probability = np.log10(np.maximum(_mass[1:], 1e-16))
        _mesh = _axes[0].pcolormesh(
            _grid.log_human_capital_nodes, _grid.asset_interior_nodes,
            np.ma.masked_where(_mass[1:] <= 0, _log_probability), shading="nearest", cmap="viridis",
            vmin=-8, vmax=0,
        )
        _figure.colorbar(_mesh, ax=_axes[0], label="log10 probability per node")
        _axes[0].set(xlabel="Log human capital", ylabel="Assets", title="Interior probability mass")
        _axes[0].set_yscale("symlog", linthresh=0.1)
        _axes[1].plot(_grid.log_human_capital_nodes, _mass[0], color="tab:orange")
        _axes[1].set(xlabel="Log human capital", ylabel="Probability per face node",
                     title=f"Floor face: {_mass[0].sum():.2%} of population")
        _axes[2].step(_grid.asset_nodes, np.cumsum(_mass.sum(axis=1)), where="post")
    else:
        _states = population_result.simulation.states[_period]
        _weights = population_result.simulation.weights
        _positive = _weights > 0
        _scatter = _axes[0].scatter(
            _states[_positive, 1], _states[_positive, 0],
            c=np.log10(_weights[_positive]), s=9, cmap="viridis",
        )
        _figure.colorbar(_scatter, ax=_axes[0], label="log10 node probability")
        _axes[0].set(xlabel="Log human capital", ylabel="Assets", title="Quadrature probability nodes")
        _axes[0].set_yscale("symlog", linthresh=0.1)
        _sort_k = np.argsort(_states[:, 1])
        _axes[1].step(_states[_sort_k, 1], np.cumsum(_weights[_sort_k]), where="post")
        _axes[1].set(xlabel="Log human capital", ylabel="Cumulative probability", title="Human-capital CDF")
        _sort_a = np.argsort(_states[:, 0])
        _axes[2].step(_states[_sort_a, 0], np.cumsum(_weights[_sort_a]), where="post")
    _axes[2].set(xlabel="Assets", ylabel="Cumulative probability", title="Asset CDF", ylim=(0, 1.02))
    for _axis in _axes:
        _axis.grid(alpha=0.15)
    plt.close(_figure)
    mo.vstack([
        mo.md(f"**Distribution at model age {population_result.state_moments.time[_period]:g}.** "
              "Colors and the floor-face curve show probability masses, not densities. "
              "Grid size affects mass per node, so use CDFs and aggregate moments for refinement comparisons."),
        _figure,
    ])
    return


@app.cell(hide_code=True)
def population_moment_view(mo, plt, population_result):
    _moment_figure, _moment_axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for _axis, _field, _title in zip(
        _moment_axes.ravel()[:5],
        ["consumption", "hours", "training_time", "earnings", "participation"],
        ["Mean consumption", "Mean active time", "Mean training time", "Mean earnings", "Participation (h > 0.02)"],
        strict=True,
    ):
        _axis.plot(population_result.moments.time, getattr(population_result.moments, _field))
        _axis.set(title=_title, xlabel="Model age")
    _moment_axes.ravel()[5].plot(population_result.state_moments.time,
                                 population_result.state_moments.asset_floor_mass)
    _moment_axes.ravel()[5].set(title="Probability at numerical asset floor", xlabel="Model age")
    for _axis in _moment_axes.ravel():
        _axis.grid(alpha=0.2)
    plt.close(_moment_figure)
    mo.vstack([mo.md("**Unconditional cohort moments.** All initial probability components enter "
                     "the means. These are model units; model age is not a calendar-age estimate."), _moment_figure])
    return


@app.cell(hide_code=True)
def continuum_benchmark_table(
    benchmark_directory,
    github_sync_refresh,
    json,
    mo,
):
    github_sync_refresh.value
    continuum_benchmark_path = benchmark_directory / "continuum_gpu"
    continuum_benchmark_report = (
        json.loads((continuum_benchmark_path / "report.json").read_text())
        if (continuum_benchmark_path / "report.json").exists() else None
    )
    _rows = []
    if continuum_benchmark_report is not None:
        for _name, _result in continuum_benchmark_report.get("backends", {}).items():
            _difference = _result.get("maximum_absolute_difference_from_refined_quadrature", {})
            _rows.append({
                "Method / resolution": _name,
                "Completed": _result.get("accepted", False),
                "Warm population seconds": _result.get("warm_population", {}).get("wall_seconds"),
                "Max hours difference": _difference.get("hours"),
                "Max participation difference": _difference.get("participation"),
                "Max floor-mass difference": _difference.get("asset_floor_mass"),
                "Mass drift": _result.get("diagnostics", {}).get("maximum_mass_drift"),
            })
    mo.vstack([
        mo.md("**Population resolution comparison.** Differences use the highest quadrature order "
              "in the saved run. Completion means the numerical checks passed; transport remains "
              "an optional approximation pending moment convergence."),
        mo.ui.table(_rows, selection=None) if _rows else mo.md("The GPU comparison is being prepared."),
    ])
    return


if __name__ == "__main__":
    app.run()
