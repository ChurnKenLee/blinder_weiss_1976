import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Blinder–Weiss (1976): baseline lifecycle

    This notebook solves one deterministic lifecycle as a bounded nonlinear
    program. JAX differentiates a Hermite–Simpson transcription; the
    optimizer chooses the state and control paths jointly. Human-capital
    investment is represented internally by training time $q=hx$, with
    $0\le q\le h$, so retirement has the unique value $h=q=0$.

    The displayed result is accepted only after feasibility, KKT, and an
    independent Diffrax integration check.
    """)
    return


@app.cell
def _():
    import jax
    import matplotlib.pyplot as plt
    from blinder_weiss import (
        SolverConfig,
        benchmark_params,
        diagnose_path,
        regime_labels,
        solve_lifecycle,
    )

    return (
        SolverConfig,
        benchmark_params,
        diagnose_path,
        jax,
        plt,
        regime_labels,
        solve_lifecycle,
    )


@app.cell
def _(mo):
    interval_input = mo.ui.dropdown(
        options={"12 (quick)": 12, "24 (recommended)": 24, "48 (fine)": 48},
        value="24 (recommended)",
        label="Time intervals",
    )
    scheme_input = mo.ui.dropdown(
        options={"Hermite–Simpson": "hermite-simpson", "Trapezoid": "trapezoid"},
        value="Hermite–Simpson",
        label="Transcription",
    )
    optimizer_input = mo.ui.dropdown(
        options={"SLSQP (recommended)": "SLSQP", "IPOPT (advanced)": "ipopt"},
        value="SLSQP (recommended)",
        label="Optimizer",
    )
    productivity_input = mo.ui.number(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.22,
        label="Human-capital productivity a",
    )
    assets_input = mo.ui.number(
        start=0.0,
        stop=100.0,
        step=0.5,
        value=5.0,
        label="Initial assets A₀",
    )
    discount_input = mo.ui.number(
        start=0.0,
        stop=0.2,
        step=0.005,
        value=0.03,
        label="Discount rate ρ",
    )
    solve_button = mo.ui.run_button(label="Solve lifecycle")
    control_panel = mo.vstack(
        [
            mo.hstack([interval_input, scheme_input, optimizer_input]),
            mo.hstack([productivity_input, assets_input, discount_input]),
            solve_button,
        ]
    )
    control_panel
    return (
        assets_input,
        discount_input,
        interval_input,
        optimizer_input,
        productivity_input,
        scheme_input,
        solve_button,
    )


@app.cell
def _(
    SolverConfig,
    assets_input,
    benchmark_params,
    diagnose_path,
    discount_input,
    interval_input,
    mo,
    optimizer_input,
    productivity_input,
    scheme_input,
    solve_button,
    solve_lifecycle,
):
    mo.stop(
        not solve_button.value,
        mo.callout("Choose settings and press **Solve lifecycle**.", kind="info"),
    )
    selected_params = benchmark_params(
        human_capital_productivity=productivity_input.value,
        initial_assets=assets_input.value,
        rho=discount_input.value,
    )
    selected_config = SolverConfig(
        intervals=interval_input.value,
        scheme=scheme_input.value,
        optimizer=optimizer_input.value,
        display=False,
    )
    baseline_result = solve_lifecycle(selected_params, selected_config)
    baseline_diagnostics = diagnose_path(baseline_result)
    return baseline_diagnostics, baseline_result


@app.cell
def _(baseline_diagnostics, baseline_result, jax, mo):
    acceptance_kind = "success" if baseline_diagnostics.accepted_success else "danger"
    acceptance_text = "accepted" if baseline_diagnostics.accepted_success else "not accepted"
    minimum_integrated_assets = baseline_diagnostics.minimum_independently_integrated_assets
    status_report = mo.callout(
        mo.md(
            f"""
            ### Solution {acceptance_text}

            - Backend: `{jax.default_backend()}` with 64-bit enabled: `{jax.config.jax_enable_x64}`
            - Optimizer: `{baseline_result.config.optimizer}` — {baseline_result.message}
            - Iterations: `{baseline_result.iterations}`
            - Lifetime utility: `{baseline_result.lifetime_utility:.8g}`
            - Maximum constraint violation: `{baseline_result.max_constraint_violation:.3e}`
            - Projected KKT residual: `{baseline_diagnostics.projected_kkt_residual:.3e}`
            - Independent asset-path error: `{baseline_diagnostics.independent_asset_error:.3e}`
            - Minimum independently integrated assets: `{minimum_integrated_assets:.3e}`
            - Regimes: **{baseline_diagnostics.regime_sequence}**
            """
        ),
        kind=acceptance_kind,
    )
    status_report
    return


@app.cell
def _(baseline_result, plt, regime_labels):
    lifecycle_figure, lifecycle_axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    lifecycle_time = baseline_result.time
    lifecycle_axes[0, 0].plot(lifecycle_time, baseline_result.consumption)
    lifecycle_axes[0, 0].set_ylabel("Consumption c")
    lifecycle_axes[0, 1].plot(lifecycle_time, baseline_result.hours)
    lifecycle_axes[0, 1].set_ylabel("Labor hours h")
    lifecycle_axes[1, 0].plot(lifecycle_time, baseline_result.investment_share)
    lifecycle_axes[1, 0].set_ylabel("Investment share x")
    lifecycle_axes[1, 1].plot(lifecycle_time, baseline_result.human_capital)
    lifecycle_axes[1, 1].set_ylabel("Human capital K")
    lifecycle_axes[2, 0].plot(lifecycle_time, baseline_result.assets)
    lifecycle_axes[2, 0].axhline(
        baseline_result.params.asset_floor, color="black", linewidth=0.8, linestyle="--"
    )
    lifecycle_axes[2, 0].set_ylabel("Assets A")
    lifecycle_axes[2, 1].plot(lifecycle_time, baseline_result.wage)
    lifecycle_axes[2, 1].set_ylabel("Potential wage K g(x)")
    for lifecycle_axis in lifecycle_axes[-1, :]:
        lifecycle_axis.set_xlabel("Age / model time")
    for lifecycle_axis in lifecycle_axes.ravel():
        lifecycle_axis.grid(alpha=0.2)
    lifecycle_regimes = regime_labels(baseline_result.hours, baseline_result.investment_share)
    lifecycle_figure.suptitle(
        "Optimal lifecycle — " + " → ".join(dict.fromkeys(lifecycle_regimes)),
        y=1.01,
    )
    lifecycle_figure.tight_layout()
    lifecycle_figure
    return


@app.cell
def _(baseline_diagnostics, mo):
    diagnostic_values = baseline_diagnostics.as_dict()
    diagnostic_lines = "\n".join(
        f"| {name.replace('_', ' ')} | `{value}` |" for name, value in diagnostic_values.items()
    )
    mo.md("| Diagnostic | Value |\n|---|---:|\n" + diagnostic_lines)
    return


if __name__ == "__main__":
    app.run()
