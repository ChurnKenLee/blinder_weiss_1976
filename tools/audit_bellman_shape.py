"""Audit saved cubic solutions throughout every state cell and model age.

Reports measured derivatives and algebraic reconstruction constraints; this
is a shape check, not a control-search or mesh-convergence certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_bellman import load_solution
from blinder_weiss.bellman_bicubic import (
    bicubic_constraint_violations,
    monotone_bicubic_interpolate,
    prepare_monotone_bicubic,
)


def audit(solution, subdivisions: int) -> dict:
    if solution.config.value_interpolation != "monotone_bicubic":
        raise ValueError("This audit requires the monotone_bicubic representation")
    grids = [solution.asset_grid, solution.log_human_capital_grid]
    dense_grids = [
        np.concatenate(
            (
                (grid[:-1, None] + np.diff(grid)[:, None] * np.arange(subdivisions)
                 / subdivisions).ravel(),
                grid[-1:],
            )
        )
        for grid in grids
    ]
    queries = jnp.asarray(
        np.stack(np.meshgrid(*dense_grids, indexing="ij"), axis=-1).reshape(-1, 2)
    )
    assets, log_k = map(jnp.asarray, grids)

    @jax.jit
    def age_metrics(values):
        packed = prepare_monotone_bicubic(values, assets, log_k)

        def value(state):
            return monotone_bicubic_interpolate(
                packed, assets, log_k, state[0], state[1],
                asset_grid_curvature=solution.config.asset_grid_curvature,
                uniform_log_grid=True,
            )

        gradients = jax.vmap(jax.grad(value))(queries)
        constraints = bicubic_constraint_violations(packed, assets, log_k)
        return (
            jnp.min(gradients, axis=0),
            jnp.sum(gradients < -1e-8, axis=0),
            jnp.all(jnp.isfinite(gradients)),
            constraints,
        )

    ages = []
    for time, values in zip(solution.time, solution.values, strict=True):
        minimum, negative_count, finite, constraints = age_metrics(jnp.asarray(values))
        ages.append({
            "model_age": float(time),
            "minimum_dVdA": float(minimum[0]),
            "minimum_dVdlogK": float(minimum[1]),
            "negative_asset_derivative_count": int(negative_count[0]),
            "negative_logK_derivative_count": int(negative_count[1]),
            "all_derivatives_finite": bool(finite),
            "constraint_residuals": {key: float(value) for key, value in constraints.items()},
        })
    return {
        "subdivisions_per_cell_axis": subdivisions,
        "queries_per_age": int(queries.shape[0]),
        "negative_derivative_threshold": -1e-8,
        "domain": [[float(grid[0]), float(grid[-1])] for grid in grids],
        "minimum_dVdA": min(row["minimum_dVdA"] for row in ages),
        "minimum_dVdlogK": min(row["minimum_dVdlogK"] for row in ages),
        "negative_asset_derivative_count": sum(row["negative_asset_derivative_count"] for row in ages),
        "negative_logK_derivative_count": sum(row["negative_logK_derivative_count"] for row in ages),
        "all_derivatives_finite": all(row["all_derivatives_finite"] for row in ages),
        "ages": ages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/solver_benchmarks"))
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("--subdivisions must be positive")
    result = {}
    for folder in args.folders:
        solution, _ = load_solution(args.root / folder)
        result[folder] = audit(solution, args.subdivisions)
        print(folder, json.dumps({key: value for key, value in result[folder].items()
                                  if key != "ages"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
