"""Compare saved solver runs with independent paths and value-shape checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from blinder_weiss import (
    BellmanConfig,
    BellmanSolution,
    ModelParams,
    diagnose_bellman,
    simulate_policy,
)
from blinder_weiss.bellman import _interpolate_value_jax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_solution(folder: Path) -> tuple[BellmanSolution, dict]:
    report = json.loads((folder / "report.json").read_text())
    with np.load(folder / "policies.npz") as data:
        solution = BellmanSolution(
            params=ModelParams(**report["params"]),
            config=BellmanConfig(**{"asset_feasibility": "checkpoints", **report["config"]}),
            time=data["time"],
            asset_grid=data["A"],
            log_human_capital_grid=data["y"],
            values=data["values"],
            consumption_policy=data["c"],
            hours_policy=data["h"],
            training_time_policy=data["q"],
            solve_seconds=report["results"][-1]["solve_seconds"],
            backend="gpu",
            device=report["device"],
        )
    return solution, report


def make_derivatives(asset_grid, log_grid, config, queries):
    @jax.jit
    def derivatives(values):
        return jax.vmap(
            jax.grad(
                lambda z: _interpolate_value_jax(values, asset_grid, log_grid, z[0], z[1], config)
            )
        )(queries)

    return derivatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/solver_benchmarks"))
    parser.add_argument(
        "--folders", nargs="+", default=["optimized_default", "pchip_default", "pchip_fine"]
    )
    args = parser.parse_args()
    reference_file = args.root / "direct_reference.npz"
    reference = dict(np.load(reference_file)) if reference_file.exists() else None
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    queries = jnp.asarray(
        np.stack(
            np.meshgrid(np.linspace(0.05, 25, 101), np.linspace(-0.5, 1, 51), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
    )
    for name in args.folders:
        folder = args.root / name
        solution, report = load_solution(folder)
        simulation = simulate_policy(solution)
        report["diagnostics"] = diagnose_bellman(solution, simulation).as_dict()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        asset_grid = jnp.asarray(solution.asset_grid)
        log_grid = jnp.asarray(solution.log_human_capital_grid)
        config = solution.config

        derivatives = make_derivatives(asset_grid, log_grid, config, queries)
        metrics = []
        for age in [0, 10, 25, 40, 55, 65]:
            index = int(age / solution.params.horizon * config.periods)
            gradient = np.asarray(derivatives(jnp.asarray(solution.values[index])))
            metrics.append(
                dict(
                    age=age,
                    min_dVdA=float(gradient[:, 0].min()),
                    min_dVdy=float(gradient[:, 1].min()),
                    negative_A_count=int((gradient[:, 0] < -1e-8).sum()),
                    negative_y_count=int((gradient[:, 1] < -1e-8).sum()),
                )
            )
        step = solution.params.horizon / config.periods
        retirement = (
            (simulation.hours[1:] < 1e-6)
            & (simulation.hours[:-1] < 1e-6)
            & (simulation.assets[1:-1] > 0.001)
        )
        params = solution.params
        euler = np.log(simulation.consumption[1:] / simulation.consumption[:-1]) / step - (
            params.interest_rate - params.rho
        ) / (1.0 - params.consumption_power)
        row = dict(
            initial_value=solution.initial_value,
            lifetime_utility=simulation.lifetime_utility,
            value_gap=simulation.value_gap,
            derivative_checks=metrics,
            retirement_consumption_euler=dict(
                count=int(retirement.sum()),
                rms=float(np.sqrt(np.mean(euler[retirement] ** 2))),
                maximum=float(np.max(np.abs(euler[retirement]))),
            ),
        )
        if reference is not None:
            midpoints = 0.5 * (simulation.time[:-1] + simulation.time[1:])
            row["direct_control_rmse_at_period_midpoints"] = {
                key: float(
                    np.sqrt(
                        np.mean(
                            (
                                getattr(simulation, field)
                                - np.interp(midpoints, reference["time"], reference[key])
                            )
                            ** 2
                        )
                    )
                )
                for key, field in [("c", "consumption"), ("h", "hours"), ("q", "training_time")]
            }
        summary[name] = row
        with (folder / "trajectory.npz.tmp").open("wb") as output:
            np.savez_compressed(
                output,
                t=simulation.time,
                A=simulation.assets,
                K=simulation.human_capital,
                c=simulation.consumption,
                h=simulation.hours,
                q=simulation.training_time,
            )
        (folder / "trajectory.npz.tmp").replace(folder / "trajectory.npz")
        fields = [
            "consumption",
            "hours",
            "training_time",
            "assets",
            "human_capital",
            "investment_share",
        ]
        titles = [
            "Consumption",
            "Active time",
            "Training time",
            "Assets",
            "Human capital",
            "Training share",
        ]
        for ax, field, title in zip(axes.flat, fields, titles, strict=True):
            values = getattr(simulation, field)
            ax.plot(simulation.time[: len(values)], values, label=name)
            ax.set(title=title, xlabel="Model age")
            ax.grid(alpha=0.2)
        print(name, json.dumps(row), flush=True)
    if reference is not None:
        reference["x"] = np.divide(
            reference["q"],
            reference["h"],
            out=np.zeros_like(reference["q"]),
            where=reference["h"] > 1e-6,
        )
        for ax, key in zip(axes.flat, ["c", "h", "q", "A", "K", "x"], strict=True):
            ax.plot(
                reference["time"],
                reference[key],
                color="black",
                ls="--",
                label="Direct collocation",
            )
    axes[0, 0].legend(fontsize=7)
    fig.savefig(args.root / "lifecycle_comparison.png", dpi=170)
    (args.root / "accuracy_summary.json.tmp").write_text(json.dumps(summary, indent=2) + "\n")
    (args.root / "accuracy_summary.json.tmp").replace(args.root / "accuracy_summary.json")


if __name__ == "__main__":
    main()
