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


@app.cell
def empirical_profile_controls(mo):
    survey_group = mo.ui.dropdown(options=["all", "men", "women"], value="all", label="Survey population")
    survey_group
    return (survey_group,)


@app.cell
def empirical_profile_view(
    Path,
    github_sync_refresh,
    json,
    mo,
    np,
    plt,
    survey_group,
):
    github_sync_refresh
    empirical_profiles = json.loads(Path("/marimo/blinder_weiss_1976/output/calibration/empirical_age_profiles_2024.json").read_text())
    _acs = [_row for _row in empirical_profiles["acs"]["profiles"] if _row["sex"] == survey_group.value]
    _atus = [_row for _row in empirical_profiles["atus"]["profiles"] if _row["sex"] == survey_group.value]
    _labels = [_row["age_band"] for _row in _acs]
    _x = np.arange(len(_labels))
    _survey_fig, _survey_axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    _survey_axes[0].plot(_x, [_row["employment_rate"] for _row in _acs], "o-", label="ACS reference week")
    _survey_axes[0].plot(_x, [_row["employment_rate"] for _row in _atus], "o-", label="ATUS employment status")
    _survey_axes[0].set(title="Employment rate", ylabel="Population fraction", ylim=(0, 1))
    _survey_axes[0].legend(fontsize=8)
    _survey_axes[1].plot(_x, [_row["usual_weekly_hours_among_reporters"] for _row in _acs], "o-")
    _survey_axes[1].set(title="ACS usual weekly hours", ylabel="Hours, among valid reporters")
    _survey_axes[2].plot(_x, [_row["working_minutes_per_day"] for _row in _atus], "o-", label="Working (0501xx)")
    _survey_axes[2].plot(_x, [_row["education_minutes_per_day"] for _row in _atus], "o-", label="Education (06xxxx)")
    _survey_axes[2].set(title="ATUS unconditional diary time", ylabel="Minutes per day")
    _survey_axes[2].legend(fontsize=8)
    for _axis in _survey_axes:
        _axis.set_xticks(_x, _labels, rotation=45)
        _axis.set_xlabel("Calendar-age band")
        _axis.grid(alpha=0.2)
    plt.close(_survey_fig)
    mo.vstack([
        mo.md("**Empirical measurement preparation — 2024.** These weighted descriptive profiles retain survey units. "
              "ACS excludes institutional group quarters; ATUS applies each diary weight once. "
              "Employment includes employed people with no work on the diary day. ACS hours are conditional on a "
              "valid usual-hours report, while diary means include zero time. The 80–84 band respects ATUS age grouping. "
              "Model time units, survey uncertainty, and the mapping from education to training remain to be specified; "
              "these profiles have not been used as calibration targets."),
        _survey_fig,
    ])
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
def policy_jitter_comparison(
    benchmark_directory,
    mo,
    np,
    plt,
    reference_household,
):
    _probe = ["A = 5, K = 1", "A = 2, K = 0.8", "A = 8, K = 1.2"].index(reference_household.value)
    _jitter_figure, _jitter_axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    with np.load(benchmark_directory / "asset_refinement_smoothness.npz") as _data:
        for _folder, _label, _style in [
            ("bicubic_padded_fine", "61 asset nodes", "--"),
            ("bicubic_asset121_fine", "121 asset nodes", "-"),
        ]:
            _time = _data[f"{_folder}_time"]
            _controls = _data[f"{_folder}_controls"][:, _probe]
            _retired = _controls[:, 1] < 1e-6
            _jitter_axes[0].plot(_time[:-1], np.where(_retired, _controls[:, 0], np.nan), _style, label=_label)
            _eligible = _data[f"{_folder}_euler_eligible"][:, _probe]
            _residual = _data[f"{_folder}_euler_residuals"][:, _probe]
            _jitter_axes[1].plot(_time[1:-1], np.where(_eligible, _residual, np.nan), _style, label=_label)
    _jitter_axes[0].set(title="Consumption during retirement", xlabel="Model age", ylabel="Consumption")
    _jitter_axes[1].axhline(0, color="black", lw=0.8)
    _jitter_axes[1].set(title="Interior retirement Euler residual", xlabel="Model age", ylabel="Log-growth error per year")
    for _axis in _jitter_axes:
        _axis.grid(alpha=0.2)
        _axis.legend(fontsize=8)
    plt.close(_jitter_figure)
    mo.vstack([
        mo.md("**Measured jitter reduction.** The denser asset grid and larger propagation budget reduce "
              "retirement Euler RMS by 53.7%, 52.3%, and 36.7% for the baseline, low, and high reference types. "
              "Residuals exclude the borrowing boundary. Full mesh convergence remains unverified."),
        _jitter_figure,
    ])
    return


@app.cell
def boundary_refinement_view(
    benchmark_directory,
    github_sync_refresh,
    json,
    mo,
    np,
    plt,
):
    github_sync_refresh
    _refinement_folder = benchmark_directory / "boundary_time_refinement"
    _state_refinement = benchmark_directory / "boundary_state_refinement"
    if (_state_refinement / "report.json").exists():
        _candidate_report = json.loads((_state_refinement / "report.json").read_text())
        if len(_candidate_report.get("runs", [])) >= 4:
            _refinement_folder = _state_refinement
    boundary_refinement_report = json.loads((_refinement_folder / "report.json").read_text())
    _refinement_rows = []
    for _run in boundary_refinement_report["runs"]:
        _audit = _run["continuous_asset_path_audit"]
        _refinement_rows.append({
            "Method": _run["config"]["asset_feasibility"],
            "Periods": _run["config"]["periods"],
            "Asset nodes": _run["config"]["asset_nodes"],
            "Maximum numerical-floor violation": _audit["maximum_numerical_floor_violation"],
            "Retirement consumption Euler RMS / year": _run["weighted_retirement_euler_rms_per_year"],
            "Maximum value vs realized-utility gap": _run["maximum_absolute_value_gap"],
        })
    _refinement_figure, _refinement_axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    with np.load(_refinement_folder / "paths.npz") as _refinement_arrays:
        _atom = int(np.flatnonzero(_refinement_arrays["initial_component"] == "point_atom")[0])
        for _i, _run in enumerate(boundary_refinement_report["runs"]):
            _label = f"{_run['config']['asset_feasibility']}, {_run['config']['periods']} periods, {_run['config']['asset_nodes']} assets"
            _time = _refinement_arrays[f"run_{_i}_time"]
            _controls = _refinement_arrays[f"run_{_i}_controls"]
            _refinement_axes[0].plot(_time[:-1], _controls[:, _atom, 0], label=_label, linewidth=1.2)
            _refinement_axes[1].plot(_time, _refinement_arrays[f"run_{_i}_native_floor_mass"], label=_label)
    _refinement_axes[0].set(title="Consumption: reference type A=5, K=1", xlabel="Model time", xlim=(45, 70))
    _refinement_axes[1].set(title="Endpoint floor mass: fixed initial-law stress test", xlabel="Model time", xlim=(0, 25))
    for _axis in _refinement_axes:
        _axis.grid(alpha=0.2)
        _axis.legend(fontsize=7)
    plt.close(_refinement_figure)
    mo.vstack([
        mo.md("**Constraint and refinement checks.** These comparisons keep the same initial distribution, including its 5% point atom, "
              "and compare moments at shared physical ages. Exact full-period feasibility removes the missed dips below the numerical floor. "
              "Refining time alone increases consumption jitter on the fixed asset grid; the denser asset-grid experiment is reported separately. "
              "A smaller floor violation is a feasibility result, while smoothness and calibration stability require their own checks."),
        mo.ui.table(_refinement_rows, selection=None),
        _refinement_figure,
    ])
    return


@app.cell(hide_code=True)
def continuum_intro(mo):
    mo.md(r"""
    ## Continuum distribution and calibration pilot

    Quadrature integrates a specified initial distribution using deterministic,
    weighted nodes. Forward transport moves probability mass across the state grid
    as each cohort ages. Both use the same feasible policies and lifecycle dynamics.

    The initial distribution and pilot targets are **synthetic**. The working law has continuous initial heterogeneity and a selectable floor component; the interior point atom is an explicit stress-test option. Quadrature is
    the current calibration reference. Forward transport is experimental because
    its moment and boundary-mass errors remain material under refinement. Assets at the
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
        BellmanConfig, BellmanSolution, ModelParams, InitialAtom,
        SyntheticInitialDistribution, initial_quadrature, simulate_population,
    )


    def load_population_policy(folder):
        """Read one completed policy checkpoint with its exact numerical settings."""
        import jax
        report = json.loads((folder / "report.json").read_text())
        config = BellmanConfig(**{"asset_feasibility": "checkpoints", **report["config"]})
        platform = None if config.compute_platform == "auto" else config.compute_platform
        device = jax.devices(platform)[config.device_index]
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
        InitialAtom,
        SyntheticInitialDistribution,
        initial_quadrature,
        load_population_policy,
        simulate_population,
    )


@app.cell
def population_controls(mo):
    population_settings = mo.ui.dictionary({
        "solution": mo.ui.dropdown(
            options={"Continuous constraint: 140 periods": "continuous_140",
                     "Continuous constraint: 280 periods": "continuous_280",
                     "Historical checkpoints: 140 periods": "checkpoints_140"},
            value="Continuous constraint: 140 periods", label="Policy solution",
        ),
        "backend": mo.ui.dropdown(
            options={"Quadrature reference": "quadrature", "Conservative transport": "transport"},
            value="Quadrature reference", label="Population method",
        ),
        "quadrature_order": mo.ui.dropdown(
            options={"8 × 8": 8, "16 × 16": 16, "32 × 32": 32, "64 × 64": 64, "128 × 128": 128},
            value="64 × 64", label="Initial quadrature",
        ),
        "transport_resolution": mo.ui.dropdown(
            options={"31 × 31": 31, "61 × 61": 61, "121 × 121": 121},
            value="61 × 61", label="Forward grid (transport only)",
        ),
        "correlation": mo.ui.slider(-0.3, 0.3, step=0.05, value=0.25,
                                    label="Initial Corr(A, log K), interior component"),
        "floor_mass": mo.ui.slider(0.0, 0.2, step=0.01, value=0.05,
                                   label="Initial probability at numerical asset floor"),
        "point_mass": mo.ui.slider(0.0, 0.1, step=0.01, value=0.0,
                                   label="Point-atom stress test: probability at (A, K) = (5, 1)"),
        "near_floor_width": mo.ui.slider(0.002, 0.05, step=0.002, value=0.01,
                                         label="Near-floor band width (model assets)"),
    }).form(submit_button_label="Simulate population", show_clear_button=False)
    population_settings
    return (population_settings,)


@app.cell
def run_population(
    InitialAtom,
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
    _policy_folder = (benchmark_directory / "bicubic_asset121_fine"
                      if _population_inputs["solution"] == "checkpoints_140"
                      else benchmark_directory / "boundary_time_refinement" / _population_inputs["solution"])
    population_solution = load_population_policy(_policy_folder)
    _population_config = population_solution.config
    population_law = SyntheticInitialDistribution(
        correlation=_population_inputs["correlation"],
        asset_floor_mass=_population_inputs["floor_mass"],
        atoms=(InitialAtom(5.0, 1.0, _population_inputs["point_mass"]),)
              if _population_inputs["point_mass"] > 0 else (),
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
        store_snapshots=True, near_asset_floor_width=_population_inputs["near_floor_width"],
    )
    mo.md(f"**Completed {population_result.backend}:** "
          f"{len(population_result.state_moments.time) - 1} decision periods; "
          f"maximum mass drift {population_result.diagnostics['maximum_mass_drift']:.2e}; "
          f"maximum full-period numerical-floor violation "
          f"{population_result.diagnostics['maximum_full_period_floor_violation']:.2e}. "
          f"The selected initial point atom has probability {_population_inputs['point_mass']:.0%}. "
          "The remaining probability is continuous after accounting for the selected floor mass.")
    return population_initial_nodes, population_result


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
                                 population_result.state_moments.asset_floor_mass, label="Exact endpoint floor contact")
    _moment_axes.ravel()[5].plot(population_result.state_moments.time,
        population_result.state_moments.near_asset_floor_mass, "--",
        label=f"Within {population_result.state_moments.near_asset_floor_width:g} assets of floor")
    _moment_axes.ravel()[5].set(title="Floor contact and near-floor share", xlabel="Model age")
    _moment_axes.ravel()[5].legend(fontsize=7)
    for _axis in _moment_axes.ravel():
        _axis.grid(alpha=0.2)
    plt.close(_moment_figure)
    mo.vstack([mo.md("**Unconditional cohort moments.** All initial probability components enter "
                     "the means. Exact floor mass counts period endpoints. The fixed-width band also includes nearby interior states. These are model units; model age is not a calendar-age estimate."), _moment_figure])
    return


@app.cell
def floor_spike_attribution(
    mo,
    np,
    plt,
    population_initial_nodes,
    population_result,
):
    if population_result.backend == "transport":
        _floor_explanation = mo.md("Select quadrature to attribute endpoint floor mass to the initial population components.")
    else:
        from blinder_weiss.continuum import _cached_floor_contacts as _classify_floor_contacts

        _sim = population_result.simulation
        _contacts = np.asarray(_classify_floor_contacts(_sim.solution.config)(
            _sim.solution.params, _sim.states, _sim.controls,
        ))
        _components = population_initial_nodes.component
        _component_names = ["interior", "asset_floor", "point_atom"]
        _component_labels = ["Initially continuous population", "Initial floor component", "Initial point atom"]
        _contributions = [
            _contacts @ np.where(_components == _name, _sim.weights, 0.0)
            for _name in _component_names
        ]
        _point_probability = float(_sim.weights[_components == "point_atom"].sum())
        _interior_probability = float(_sim.weights[_components == "interior"].sum())
        _alternative_weights = np.where(_components == "point_atom", 0.0, _sim.weights)
        if _interior_probability > 0:
            _alternative_weights += np.where(
                _components == "interior",
                _sim.weights * _point_probability / _interior_probability, 0.0,
            )
        _alternative_floor = _contacts @ _alternative_weights
        _attribution_fig, _attribution_axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
        _attribution_axes[0].stackplot(_sim.time, *_contributions, labels=_component_labels, alpha=0.8)
        _attribution_axes[0].set(title="Where endpoint floor mass comes from", xlabel="Model age", ylabel="Population probability")
        _attribution_axes[0].legend(fontsize=7)
        _attribution_axes[1].plot(_sim.time, population_result.state_moments.asset_floor_mass,
                                  label="Current synthetic mixture")
        _attribution_axes[1].plot(_sim.time, _alternative_floor, "--",
                                  label="Point-atom weight moved to continuous component")
        _attribution_axes[1].set(title="Sensitivity to the imposed initial atom", xlabel="Model age", ylabel="Population probability")
        _attribution_axes[1].legend(fontsize=7)
        for _axis in _attribution_axes:
            _axis.set_xlim(0, 25)
            _axis.grid(alpha=0.2)
        plt.close(_attribution_fig)
        _floor_explanation = mo.vstack([
            mo.md(f"**Sensitivity to the initial distribution.** The selected point atom has mass {_point_probability:.0%}. "
                  "People starting at exactly the same state follow one identical path and can reach the floor together. "
                  "The historical checkpoint benchmark with a 5% atom produced a five-percentage-point spike at model time 13.5. "
                  "Continuous feasibility changes when contact occurs; contact inside a period can leave positive endpoint assets. "
                  "The dashed comparison moves any point-atom weight into the continuous component and retains the initial floor share. "
                  "Both profiles use the same policies and unsmoothed trajectories. With no point atom selected, the curves coincide."),
            _attribution_fig,
        ])
    _floor_explanation
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
                "Completed": _result.get("completed", _result.get("accepted", False)),
                "Warm population seconds": _result.get("warm_population", {}).get("wall_seconds"),
                "Max hours difference": _difference.get("hours"),
                "Max participation difference": _difference.get("participation"),
                "Max floor-mass difference": _difference.get("asset_floor_mass"),
                "Mass drift": _result.get("diagnostics", {}).get("maximum_mass_drift"),
            })
    mo.vstack([
        mo.md("**Historical population resolution comparison.** This saved run uses the 140-period, "
              "121-asset checkpoint policy and an explicit 5% interior point atom. Differences use the highest quadrature order "
              "in the saved run. Completion means the numerical checks passed; transport remains "
              "an optional approximation pending moment convergence."),
        mo.ui.table(_rows, selection=None) if _rows else mo.md("The GPU comparison is being prepared."),
    ])
    return (continuum_benchmark_report,)


@app.cell(hide_code=True)
def synthetic_calibration_view(continuum_benchmark_report, mo, plt):
    _fit = (
        continuum_benchmark_report.get("synthetic_fit")
        if continuum_benchmark_report is not None else None
    )
    if _fit is None:
        _calibration_display = mo.md("The synthetic fitting experiment will appear after the benchmark completes.")
    else:
        _fit_figure, _fit_axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        _evaluations = sorted(_fit["evaluations"], key=lambda _item: _item["leisure_weight"])
        _fit_axis.plot([_item["leisure_weight"] for _item in _evaluations],
                       [_item["loss"] for _item in _evaluations], "o-")
        _fit_axis.axvline(_fit["known_leisure_weight"], color="black", ls=":", label="Known synthetic value")
        _fit_axis.set(xlabel="Leisure weight", ylabel="Scaled moment loss", title="Historical synthetic parameter recovery")
        _fit_axis.grid(alpha=0.2)
        _fit_axis.legend()
        plt.close(_fit_figure)
        _calibration_display = mo.vstack([
            mo.md(f"**Historical synthetic calibration pilot:** known leisure weight {_fit['known_leisure_weight']:.6f}; "
                  f"fitted {_fit['fitted_leisure_weight']:.6f}; "
                  f"absolute error {_fit['absolute_parameter_error']:.2e}. "
                  f"Optimizer success: {_fit['optimizer_success']}. "
                  "This saved result uses the 140-period, 121-asset checkpoint policy, a 5% interior point atom, "
                  "and the earlier optimizer. The current fitter also evaluates the supplied parameter and retains it "
                  "when its loss is lower. Targets come from the same model with refined quadrature. This checks the fitting pipeline; "
                  "it does not estimate survey parameters or establish identification."),
            _fit_figure,
        ])
    _calibration_display
    return


@app.cell
def calibration_stability_view(
    benchmark_directory,
    github_sync_refresh,
    json,
    mo,
):
    github_sync_refresh
    _stability_path = benchmark_directory / "calibration_stability_gpu" / "report.json"
    mo.stop(not _stability_path.exists(), mo.md("The production calibration stability run is preparing its fixed target."))
    calibration_stability_report = json.loads(_stability_path.read_text())
    _calibration_rows = []
    for _name, _run in calibration_stability_report.get("resolutions", {}).items():
        _trace = _run.get("fit_trace", [])
        _calibration_rows.append({
            "Resolution": _name,
            "Asset nodes": _run.get("asset_nodes"),
            "Status": "completed" if "optimizer_success" in _run else "running",
            "Parameter evaluations": len(_trace),
            "Initial-parameter loss": _run.get("baseline_loss_against_fixed_target"),
            "Best fitted parameter": _run.get("fitted_parameter"),
            "Fitted loss": _run.get("fitted_loss"),
            "Candidate selected from": _run.get("selection_source"),
        })
    _acceptance_rows = []
    for _name, _check in calibration_stability_report.get("comparisons", {}).items():
        _acceptance_rows.append({
            "Comparison with finest reference": _name,
            "Within all tolerances": _check["passed"],
            "Largest moment change / target scale": _check["maximum_standardized_moment_difference"],
            "Loss change": _check["absolute_loss_difference"],
            "Fitted parameter change": _check["absolute_fitted_parameter_difference"],
        })
    _initial_fit = calibration_stability_report.get("initial_law_fit")
    _initial_note = (f"Initial-floor-share fit: {_initial_fit['fitted_value']:.6f}; "
                     f"known synthetic share {_initial_fit['known_value']:.6f}."
                     if _initial_fit else "Initial-distribution fitting follows the structural comparisons.")
    mo.vstack([
        mo.md("**Calibration stability under refinement.** Every resolution uses one fixed synthetic target from "
              "the finest time/state grid and quadrature rule. The supplied parameter is retained if it beats the "
              "bounded search; its selection is reported explicitly. Acceptance checks compare moments, loss, fitted "
              "parameters, and full-period feasibility. The tolerances are numerical choices and are not survey standard errors."),
        mo.ui.table(_calibration_rows, selection=None),
        mo.ui.table(_acceptance_rows, selection=None) if _acceptance_rows else mo.md("Resolution acceptance checks follow the completed fits."),
        mo.md(_initial_note),
    ])
    return


if __name__ == "__main__":
    app.run()
