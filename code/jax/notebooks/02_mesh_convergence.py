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
    # Mesh convergence and independent integration

    A small collocation defect only shows that the finite-dimensional NLP
    was solved. This notebook performs the stronger check: interpolate the
    optimized controls, integrate the original continuous dynamics with an
    adaptive Diffrax solver, and verify that the discrepancy shrinks as the
    time mesh is refined.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from blinder_weiss import SolverConfig, diagnose_path, solve_mesh_sequence

    return SolverConfig, diagnose_path, np, plt, solve_mesh_sequence


@app.cell
def _(mo):
    finest_mesh_input = mo.ui.dropdown(
        options={"24 intervals": 24, "48 intervals": 48},
        value="24 intervals",
        label="Finest mesh",
    )
    convergence_button = mo.ui.run_button(label="Run convergence study")
    mo.hstack([finest_mesh_input, convergence_button])
    return convergence_button, finest_mesh_input


@app.cell
def _(
    SolverConfig,
    convergence_button,
    diagnose_path,
    finest_mesh_input,
    mo,
    solve_mesh_sequence,
):
    mo.stop(
        not convergence_button.value,
        mo.callout("Press **Run convergence study** to solve all meshes.", kind="info"),
    )
    convergence_meshes = [8, 12, 24] if finest_mesh_input.value == 24 else [8, 12, 24, 48]
    convergence_config = SolverConfig(
        scheme="hermite-simpson",
        optimizer="SLSQP",
        max_iterations=2_500,
    )
    convergence_results = solve_mesh_sequence(convergence_meshes, config=convergence_config)
    convergence_diagnostics = [diagnose_path(result) for result in convergence_results]
    return convergence_diagnostics, convergence_results


@app.cell
def _(convergence_diagnostics, convergence_results, mo):
    convergence_rows = []
    for convergence_result, convergence_diagnostic in zip(
        convergence_results, convergence_diagnostics, strict=True
    ):
        convergence_rows.append(
            "| "
            + " | ".join(
                [
                    str(convergence_result.config.intervals),
                    str(convergence_diagnostic.accepted_success),
                    f"{convergence_result.lifetime_utility:.9g}",
                    f"{convergence_result.max_constraint_violation:.2e}",
                    f"{convergence_diagnostic.projected_kkt_residual:.2e}",
                    f"{convergence_diagnostic.independent_asset_error:.2e}",
                ]
            )
            + " |"
        )
    convergence_table = "\n".join(
        [
            "| Intervals | Accepted | Utility | Constraint | KKT | ODE asset error |",
            "|---:|:---:|---:|---:|---:|---:|",
            *convergence_rows,
        ]
    )
    mo.md(convergence_table)
    return


@app.cell
def _(convergence_diagnostics, convergence_results, np, plt):
    convergence_figure, convergence_axes = plt.subplots(1, 3, figsize=(16, 4.5))
    mesh_sizes = np.asarray([result.config.intervals for result in convergence_results])
    mesh_steps = np.asarray(
        [result.params.horizon / result.config.intervals for result in convergence_results]
    )
    utility_values = np.asarray([result.lifetime_utility for result in convergence_results])
    integration_errors = np.asarray(
        [diagnostic.independent_asset_error for diagnostic in convergence_diagnostics]
    )

    convergence_axes[0].plot(mesh_sizes, utility_values, marker="o")
    convergence_axes[0].set_xlabel("Intervals")
    convergence_axes[0].set_ylabel("Lifetime utility")
    convergence_axes[1].loglog(mesh_steps, integration_errors, marker="o")
    convergence_axes[1].invert_xaxis()
    convergence_axes[1].set_xlabel("Time step Δt")
    convergence_axes[1].set_ylabel("Independent asset error")
    for path_result in convergence_results:
        convergence_axes[2].plot(
            path_result.time,
            path_result.investment_share,
            label=f"N={path_result.config.intervals}",
        )
    convergence_axes[2].set_xlabel("Age / model time")
    convergence_axes[2].set_ylabel("Investment share x")
    convergence_axes[2].legend()
    for convergence_axis in convergence_axes:
        convergence_axis.grid(alpha=0.2)
    convergence_figure.tight_layout()
    convergence_figure
    return


if __name__ == "__main__":
    app.run()
