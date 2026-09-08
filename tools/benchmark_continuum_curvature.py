"""Isolate forward-grid curvature with an unchanged saved Bellman solution.

Use after benchmark_continuum.py. Compares the same initial probability law
and saved quadrature target, reusing the standard curvature-two result and
running only the supplemental forward population calculation. No new fit or
Bellman solve is performed, and transport does not become a default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from blinder_weiss.calibration import (
    AgeMomentTarget,
    CalibrationTargets,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    InitialAtom,
    SyntheticInitialDistribution,
    initial_quadrature,
    simulate_population,
)
from blinder_weiss.distribution import DistributionDomainError, DistributionGrid


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark_tool = _load_tool("benchmark_continuum")
validation_tool = _load_tool("validate_calibration_grid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--size", type=int, default=121)
    parser.add_argument("--curvature", type=float, default=3.0)
    args = parser.parse_args()
    if args.size < 3 or not np.isfinite(args.curvature) or args.curvature <= 0:
        parser.error("size>=3 and finite positive curvature required")
    baseline = json.loads((args.baseline / "report.json").read_text())
    with np.load(args.baseline / "profiles.npz") as saved:
        reference = {name: saved[name] for name in saved.files}
    solution, _ = validation_tool.load_solution(args.solution, args.platform)
    if (
        asdict(solution.config) != baseline["config"]
        or solution.params._asdict() != baseline["params"]
    ):
        raise ValueError(
            "saved solution must use exactly the baseline model and Bellman configuration"
        )
    law_data = baseline["initial_law"]
    law = SyntheticInitialDistribution(
        **{**law_data, "atoms": tuple(InitialAtom(**atom) for atom in law_data["atoms"])}
    )
    reference_order = max(
        int(name.split("_")[-1]) for name in baseline["backends"] if name.startswith("quadrature_")
    )
    reference_label = f"quadrature_{reference_order}"
    standard_label = f"transport_{args.size}"
    targets_data = baseline["targets"]
    targets = CalibrationTargets(
        **{
            **targets_data,
            "profiles": tuple(AgeMomentTarget(**profile) for profile in targets_data["profiles"]),
        }
    )
    nodes = initial_quadrature(
        law, nodes_per_dimension=reference_order, asset_floor=solution.config.asset_minimum
    )
    config = solution.config
    grid = DistributionGrid(
        config.asset_minimum,
        config.asset_minimum
        + (config.asset_maximum - config.asset_minimum)
        * np.linspace(0.0, 1.0, args.size)[1:] ** args.curvature,
        np.linspace(config.log_human_capital_minimum, config.log_human_capital_maximum, args.size),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "supplemental forward-grid curvature sensitivity; no Bellman change or new fit",
        "accepted_for_calibration": False,
        "baseline_source": str(args.baseline),
        "saved_solution_source": str(args.solution),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.baseline / "report.json",
                args.baseline / "profiles.npz",
                args.solution / "report.json",
                args.solution / "policies.npz",
            )
        },
        "model_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss").glob(
                "*.py"
            )
        },
        "config": asdict(config),
        "params": solution.params._asdict(),
        "initial_law": asdict(law),
        "targets": targets_data,
        "distribution_asset_curvature": args.curvature,
        "distribution_shape": grid.shape,
        "first_interior_gap": float(grid.asset_interior_nodes[0] - grid.asset_floor),
        "standard_curvature": baseline.get("distribution_asset_curvature", 2.0),
        "standard_result": baseline["backends"][standard_label],
    }

    def save_report():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(
            json.dumps(benchmark_tool.jsonable(report), indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(args.output / "report.json")

    timer = benchmark_tool.CompilationTimer()

    def population_call():
        return simulate_population(
            solution,
            nodes,
            backend="transport",
            distribution_grid=grid,
            participation_hours_threshold=targets.participation_hours_threshold,
            store_snapshots=True,
        )

    try:
        result, report["first_population"] = timer.measure(population_call)
        result, report["warm_population"] = timer.measure(population_call)
    except DistributionDomainError as error:
        report.update(completed=False, error=str(error), diagnostics=error.diagnostics)
        save_report()
        raise
    report.update(
        completed=True,
        diagnostics=result.diagnostics,
        synthetic_loss=weighted_age_moment_loss(result, targets).loss,
    )
    arrays = {"time": result.moments.time, "state_time": result.state_moments.time}
    differences = {}
    fields = (
        "hours",
        "participation",
        "training_time",
        "consumption",
        "earnings",
        "assets",
        "human_capital",
        "log_human_capital",
        "asset_floor_mass",
    )
    for name in fields:
        owner = result.moments if hasattr(result.moments, name) else result.state_moments
        values = getattr(owner, name)
        arrays[f"curvature_{args.curvature:g}_{name}"] = values
        arrays[f"standard_{name}"] = reference[f"{standard_label}_{name}"]
        arrays[f"reference_{name}"] = reference[f"{reference_label}_{name}"]
        differences[name] = float(np.max(np.abs(values - reference[f"{reference_label}_{name}"])))
    report["maximum_absolute_difference_from_refined_quadrature"] = differences
    indices = np.unique(np.linspace(0, config.periods, 7, dtype=int))
    arrays["snapshot_periods"] = indices
    arrays["snapshots"] = result.simulation.masses[indices]
    arrays["asset_nodes"] = grid.asset_nodes
    arrays["log_human_capital_nodes"] = grid.log_human_capital_nodes
    arrays["initial_assets"] = nodes.assets
    arrays["initial_human_capital"] = nodes.human_capital
    arrays["initial_weights"] = nodes.weights
    temporary = args.output / "profiles.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    temporary.replace(args.output / "profiles.npz")
    save_report()
    print(
        json.dumps(
            benchmark_tool.jsonable(
                {
                    key: report[key]
                    for key in (
                        "completed",
                        "distribution_asset_curvature",
                        "first_interior_gap",
                        "synthetic_loss",
                        "maximum_absolute_difference_from_refined_quadrature",
                        "warm_population",
                    )
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
