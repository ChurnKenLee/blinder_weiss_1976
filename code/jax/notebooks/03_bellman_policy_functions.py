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
    # Feedback policies from the Bellman equation

    This notebook solves the deterministic lifecycle problem backward on an
    asset–human-capital state domain. It does **not** take finite differences
    of the value function. Instead, a semi-Lagrangian step holds controls
    constant for one period, integrates the state equations exactly, and
    evaluates the next-period value by monotone bilinear interpolation.

    The maximizers at every state node are the feedback policies

    \[
    \pi_n(A,K)=\left(c_n^*(A,K),h_n^*(A,K),q_n^*(A,K)\right).
    \]

    The direct-collocation path is solved independently below. Agreement of
    the two methods along the benchmark lifecycle is a more meaningful check
    than a small Bellman residual alone. The notebook automatically expands
    the computational domain and jointly refines time, state, and control
    resolution until the research stopping rules pass or the level cap is
    reached.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from blinder_weiss import (
        BellmanConfig,
        BellmanConvergenceConfig,
        SolverConfig,
        diagnose_bellman,
        diagnose_path,
        simulate_policy,
        solve_bellman_converged,
        solve_mesh_sequence,
    )

    return (
        BellmanConfig,
        BellmanConvergenceConfig,
        SolverConfig,
        diagnose_bellman,
        diagnose_path,
        np,
        plt,
        simulate_policy,
        solve_bellman_converged,
        solve_mesh_sequence,
    )


@app.cell
def _(mo):
    bellman_resolution_input = mo.ui.dropdown(
        options={
            "Quick: 21 × 17 states": "quick",
            "Recommended: 31 × 25 states": "recommended",
            "Fine: 41 × 31 states": "fine",
        },
        value="Recommended: 31 × 25 states",
        label="State and control resolution",
    )
    bellman_periods_input = mo.ui.dropdown(
        options={
            "Annual controls: 70 periods": 70,
            "Half-year controls: 140 periods": 140,
        },
        value="Annual controls: 70 periods",
        label="Time resolution",
    )
    bellman_levels_input = mo.ui.dropdown(
        options={
            "Two-level audit": 2,
            "Three refinement levels": 3,
            "Research: up to four levels": 4,
        },
        value="Research: up to four levels",
        label="Automatic convergence",
    )
    bellman_platform_input = mo.ui.dropdown(
        options={
            "Automatic (GPU when available)": "auto",
            "Require NVIDIA GPU": "gpu",
            "CPU": "cpu",
        },
        value="Automatic (GPU when available)",
        label="JAX compute platform",
    )
    bellman_run_button = mo.ui.run_button(label="Solve Bellman and comparison path")
    mo.vstack(
        [
            mo.hstack(
                [
                    bellman_resolution_input,
                    bellman_periods_input,
                    bellman_levels_input,
                    bellman_platform_input,
                ]
            ),
            bellman_run_button,
        ]
    )
    return (
        bellman_levels_input,
        bellman_periods_input,
        bellman_platform_input,
        bellman_resolution_input,
        bellman_run_button,
    )


@app.cell
def _(
    BellmanConfig,
    BellmanConvergenceConfig,
    SolverConfig,
    bellman_levels_input,
    bellman_periods_input,
    bellman_platform_input,
    bellman_resolution_input,
    bellman_run_button,
    diagnose_bellman,
    diagnose_path,
    mo,
    simulate_policy,
    solve_bellman_converged,
    solve_mesh_sequence,
):
    mo.stop(
        not bellman_run_button.value,
        mo.callout(
            "Choose a resolution and press **Solve Bellman and comparison path**.",
            kind="info",
        ),
    )
    bellman_resolution_settings = {
        "quick": dict(
            asset_nodes=21,
            human_capital_nodes=17,
            hours_nodes=9,
            investment_nodes=9,
            consumption_nodes=11,
            refinement_steps=16,
        ),
        "recommended": dict(
            asset_nodes=31,
            human_capital_nodes=25,
            hours_nodes=11,
            investment_nodes=11,
            consumption_nodes=15,
            refinement_steps=24,
        ),
        "fine": dict(
            asset_nodes=41,
            human_capital_nodes=31,
            hours_nodes=13,
            investment_nodes=13,
            consumption_nodes=17,
            refinement_steps=32,
        ),
    }
    selected_target_config = BellmanConfig(
        periods=bellman_periods_input.value,
        compute_platform=bellman_platform_input.value,
        **bellman_resolution_settings[bellman_resolution_input.value],
    )
    selected_convergence_config = BellmanConvergenceConfig(
        max_levels=bellman_levels_input.value,
        required_consecutive_passes=min(2, bellman_levels_input.value - 1),
    )
    bellman_convergence = solve_bellman_converged(
        config=selected_target_config,
        convergence_config=selected_convergence_config,
    )
    bellman_solution = bellman_convergence.solution
    bellman_simulation = simulate_policy(bellman_solution, policy_method="greedy")
    bellman_interpolated_simulation = simulate_policy(bellman_solution, policy_method="interpolate")
    bellman_diagnostics = diagnose_bellman(bellman_solution, bellman_simulation)
    comparison_sequence = solve_mesh_sequence(
        [24, 48, 96],
        config=SolverConfig(
            scheme="hermite-simpson",
            optimizer="SLSQP",
            max_iterations=5_000,
        ),
    )
    comparison_solution = comparison_sequence[-1]
    comparison_diagnostics = diagnose_path(comparison_solution, independent_integration=False)
    return (
        bellman_convergence,
        bellman_diagnostics,
        bellman_interpolated_simulation,
        bellman_simulation,
        bellman_solution,
        comparison_diagnostics,
        comparison_sequence,
        comparison_solution,
    )


@app.cell
def _(
    bellman_convergence,
    bellman_diagnostics,
    bellman_interpolated_simulation,
    bellman_simulation,
    bellman_solution,
    comparison_diagnostics,
    comparison_solution,
    mo,
):
    bellman_status_kind = (
        "success"
        if (
            bellman_convergence.converged
            and bellman_diagnostics.accepted_node_solution
            and comparison_diagnostics.accepted_success
        )
        else "warn"
    )
    bellman_utility_difference = (
        bellman_simulation.lifetime_utility - comparison_solution.lifetime_utility
    )
    interpolated_utility_difference = (
        bellman_interpolated_simulation.lifetime_utility - comparison_solution.lifetime_utility
    )
    bellman_state_nodes = (
        bellman_solution.config.asset_nodes * bellman_solution.config.human_capital_nodes
    )
    bellman_residual = bellman_diagnostics.maximum_node_bellman_residual
    bellman_capacity_slack = bellman_diagnostics.minimum_consumption_capacity_slack
    bellman_monotonicity = bellman_diagnostics.maximum_value_monotonicity_violation
    greedy_value_gap = bellman_simulation.value_gap
    interpolated_value_gap = bellman_interpolated_simulation.value_gap
    greedy_utility = bellman_simulation.lifetime_utility
    interpolated_utility = bellman_interpolated_simulation.lifetime_utility
    final_convergence_level = bellman_convergence.levels[-1]
    final_value_change = final_convergence_level.normalized_value_change
    final_policy_regret = final_convergence_level.maximum_policy_regret
    final_interpolated_regret = (
        final_convergence_level.maximum_interpolated_policy_regret
    )
    mo.callout(
        mo.md(
            f"""
            ### Backward solution and independent path comparison

            - Automatic convergence reached: `{bellman_convergence.converged}`
            - Refinement stop: `{bellman_convergence.stop_reason}`
            - Refinement levels solved: `{len(bellman_convergence.levels)}`
            - Final normalized value change: `{final_value_change}`
            - Final maximum previous-policy regret: `{final_policy_regret}`
            - Final interpolated-policy regret: `{final_interpolated_regret:.3e}`
            - JAX backend: `{bellman_solution.backend}`
            - Compute device: `{bellman_solution.device}`
            - Bellman solve time: `{bellman_solution.solve_seconds:.3f}` seconds
            - State nodes per age: `{bellman_state_nodes:,}`
            - Stored state-age values: `{bellman_solution.values.size:,}`
            - Node solution accepted: `{bellman_diagnostics.accepted_node_solution}`
            - Maximum node Bellman residual: `{bellman_residual:.3e}`
            - Minimum consumption-capacity slack: `{bellman_capacity_slack:.3e}`
            - Maximum value-monotonicity violation: `{bellman_monotonicity:.3e}`
            - Greedy lifecycle stayed in domain: `{bellman_simulation.stayed_in_domain}`
            - Greedy-recovery gap, $V_0-J(\\pi_\\mathrm{{greedy}})$: `{greedy_value_gap:.6g}`
            - Interpolated gap, $V_0-J(\\pi_\\mathrm{{interp}})$: `{interpolated_value_gap:.6g}`
            - Greedy-recovery lifecycle utility: `{greedy_utility:.9g}`
            - Interpolated-policy lifecycle utility: `{interpolated_utility:.9g}`
            - Direct-collocation utility: `{comparison_solution.lifetime_utility:.9g}`
            - Greedy minus collocation utility: `{bellman_utility_difference:.6g}`
            - Interpolated policy minus collocation: `{interpolated_utility_difference:.6g}`

            Greedy recovery re-maximizes the local Bellman objective and keeps
            the interpolated node policy as a fallback candidate. A smaller
            consistency gap alone does not establish a better economic policy;
            both rollouts must converge toward the independent benchmark as the
            state and time approximations are refined.
            """
        ),
        kind=bellman_status_kind,
    )
    return


@app.cell
def _(bellman_convergence, mo):
    convergence_rows = []
    for level in bellman_convergence.levels:
        value_change = (
            "—"
            if level.normalized_value_change is None
            else f"{level.normalized_value_change:.3e}"
        )
        policy_regret = (
            "—"
            if level.maximum_policy_regret is None
            else f"{level.maximum_policy_regret:.3e}"
        )
        convergence_rows.append(
            "| "
            + " | ".join(
                [
                    str(level.level),
                    str(level.solution.config.periods),
                    (
                        f"{level.solution.config.asset_nodes} × "
                        f"{level.solution.config.human_capital_nodes}"
                    ),
                    f"{level.solution.solve_seconds:.2f}",
                    value_change,
                    policy_regret,
                    f"{level.maximum_optimizer_shortfall:.3e}",
                    f"{level.normalized_monotonicity_violation:.3e}",
                    str(level.passed),
                ]
            )
            + " |"
        )
    convergence_table = "\n".join(
        [
            (
                "| Level | Periods | State grid | Seconds | Value change | "
                "Policy regret | Optimizer shortfall | Monotonicity | Pass |"
            ),
            "|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
            *convergence_rows,
        ]
    )
    mo.md("## Automatic convergence history\n\n" + convergence_table)
    return


@app.cell
def _(
    bellman_interpolated_simulation,
    bellman_simulation,
    comparison_solution,
    np,
    plt,
):
    comparison_figure, comparison_axes = plt.subplots(3, 2, figsize=(13, 11))
    bellman_control_time = bellman_simulation.time[:-1]
    comparison_axes[0, 0].plot(
        bellman_control_time,
        bellman_simulation.consumption,
        label="Bellman greedy recovery",
    )
    comparison_axes[0, 0].plot(
        bellman_control_time,
        bellman_interpolated_simulation.consumption,
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[0, 0].plot(
        comparison_solution.time,
        comparison_solution.consumption,
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[0, 0].set_ylabel("Consumption c")
    comparison_axes[0, 1].plot(
        bellman_control_time,
        bellman_simulation.hours,
        label="Bellman greedy recovery",
    )
    comparison_axes[0, 1].plot(
        bellman_control_time,
        bellman_interpolated_simulation.hours,
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[0, 1].plot(
        comparison_solution.time,
        comparison_solution.hours,
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[0, 1].set_ylabel("Active time h")
    minimum_identified_hours = 0.02
    comparison_axes[1, 0].plot(
        bellman_control_time,
        np.where(
            bellman_simulation.hours > minimum_identified_hours,
            bellman_simulation.investment_share,
            np.nan,
        ),
        label="Bellman greedy recovery",
    )
    comparison_axes[1, 0].plot(
        bellman_control_time,
        np.where(
            bellman_interpolated_simulation.hours > minimum_identified_hours,
            bellman_interpolated_simulation.investment_share,
            np.nan,
        ),
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[1, 0].plot(
        comparison_solution.time,
        np.where(
            comparison_solution.hours > minimum_identified_hours,
            comparison_solution.investment_share,
            np.nan,
        ),
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[1, 0].set_ylabel("Investment share x (h > 0.02)")
    comparison_axes[1, 1].plot(
        bellman_simulation.time,
        bellman_simulation.human_capital,
        label="Bellman greedy recovery",
    )
    comparison_axes[1, 1].plot(
        bellman_interpolated_simulation.time,
        bellman_interpolated_simulation.human_capital,
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[1, 1].plot(
        comparison_solution.time,
        comparison_solution.human_capital,
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[1, 1].set_ylabel("Human capital K")
    comparison_axes[2, 0].plot(
        bellman_simulation.time,
        bellman_simulation.assets,
        label="Bellman greedy recovery",
    )
    comparison_axes[2, 0].plot(
        bellman_interpolated_simulation.time,
        bellman_interpolated_simulation.assets,
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[2, 0].plot(
        comparison_solution.time,
        comparison_solution.assets,
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[2, 0].axhline(
        bellman_simulation.solution.config.asset_minimum,
        color="black",
        linewidth=0.8,
        linestyle=":",
    )
    comparison_axes[2, 0].set_ylabel("Assets A")
    comparison_axes[2, 1].plot(
        bellman_control_time,
        bellman_simulation.training_time,
        label="Bellman greedy recovery",
    )
    comparison_axes[2, 1].plot(
        bellman_control_time,
        bellman_interpolated_simulation.training_time,
        linestyle=":",
        label="Node-policy interpolation",
    )
    comparison_axes[2, 1].plot(
        comparison_solution.time,
        comparison_solution.training_time,
        linestyle="--",
        label="Direct collocation",
    )
    comparison_axes[2, 1].set_ylabel("Training time q")
    for comparison_axis in comparison_axes.ravel():
        comparison_axis.set_xlabel("Age / model time")
        comparison_axis.grid(alpha=0.2)
    comparison_axes[0, 0].legend()
    comparison_figure.suptitle("Bellman feedback rollout versus open-loop collocation", y=1.01)
    comparison_figure.tight_layout()
    comparison_figure
    return


@app.cell
def _(mo):
    policy_age_input = mo.ui.dropdown(
        options={f"t = {age}": age for age in range(0, 70, 10)},
        value="t = 20",
        label="Policy slice age",
    )
    policy_age_input
    return (policy_age_input,)


@app.cell
def _(bellman_solution, np, plt, policy_age_input):
    policy_period = min(
        int(
            policy_age_input.value
            / bellman_solution.params.horizon
            * bellman_solution.config.periods
        ),
        bellman_solution.config.periods - 1,
    )
    policy_asset_mesh, policy_human_capital_mesh = np.meshgrid(
        bellman_solution.asset_grid,
        bellman_solution.human_capital_grid,
        indexing="ij",
    )
    policy_hours = bellman_solution.hours_policy[policy_period]
    policy_investment_share = np.ma.masked_where(
        policy_hours <= 0.02,
        bellman_solution.investment_share_policy[policy_period],
    )
    policy_figure, policy_axes = plt.subplots(2, 3, figsize=(17, 9))
    policy_arrays = (
        bellman_solution.values[policy_period],
        bellman_solution.consumption_policy[policy_period],
        policy_hours,
        bellman_solution.training_time_policy[policy_period],
        policy_investment_share,
    )
    policy_titles = (
        "Value V(A,K)",
        "Consumption c*(A,K)",
        "Active time h*(A,K)",
        "Training time q*(A,K)",
        "Investment share x*(A,K), shown for h > 0.02",
    )
    for policy_axis, policy_array, policy_title in zip(
        policy_axes.ravel(), policy_arrays, policy_titles, strict=True
    ):
        policy_surface = policy_axis.pcolormesh(
            policy_asset_mesh,
            policy_human_capital_mesh,
            policy_array,
            shading="auto",
        )
        policy_axis.set_xlabel("Assets A")
        policy_axis.set_ylabel("Human capital K")
        policy_axis.set_yscale("log")
        policy_axis.set_title(policy_title)
        policy_figure.colorbar(policy_surface, ax=policy_axis)
    policy_axes.ravel()[-1].axis("off")
    policy_figure.suptitle(
        f"Feedback policies near model time {bellman_solution.time[policy_period]:.1f}",
        y=1.02,
    )
    policy_figure.tight_layout()
    policy_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and remaining convergence work

    A tiny node Bellman residual establishes internal consistency for the
    stored approximation; it does not establish that the state domain and
    control grids are fine enough. The automatic refinement table therefore
    checks:

    1. the simulated state path remains strictly inside the chosen domain;
    2. held-out values and the Bellman regret of the preceding policy stabilize;
    3. optimizer shortfall, monotonicity violations, and policy-interpolation regret remain small;
    4. both Bellman rollouts approach the independently computed collocation path.

    The plotted arrays are not smoothed after solving. Persistent jumps are
    retained as economic regime boundaries; isolated numerical misses are
    addressed by multi-start optimization and adjacent-state candidate
    propagation.

    The Bellman solution is the correct starting point for stochastic shocks:
    the next-period term can be replaced by a quadrature-weighted expectation
    without changing the state-constraint treatment.
    """)
    return


if __name__ == "__main__":
    app.run()
