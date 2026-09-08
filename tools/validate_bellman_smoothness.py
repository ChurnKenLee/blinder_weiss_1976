"""Compare saved Bellman policies on three fixed lifecycle reference types.

This is numerical evidence, not mesh convergence or an empirical calibration.
Use --reuse-paths to recompute diagnostics from the paired NPZ without a solve.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from blinder_weiss import ModelParams, simulate_cohort
from blinder_weiss.bellman_smoothness import retirement_euler_diagnostics
from validate_calibration_grid import lifetime_utilities, load_solution

_INITIAL_ASSETS = np.array([5.0, 2.0, 8.0])
_INITIAL_CAPITAL = np.array([1.0, 0.8, 1.2])


def summarize(solution, states, controls, direct_utilities):
    diagnostic = retirement_euler_diagnostics(
        solution.time,
        controls[..., 0],
        controls[..., 1],
        states[..., 0],
        solution.params,
        asset_minimum=solution.config.asset_minimum,
    )
    utility = lifetime_utilities(solution.time, states, controls, solution.params)
    summary = {
        "config": asdict(solution.config),
        "lifetime_utility": utility.tolist(),
        "direct_reference_utility_shortfall": (direct_utilities - utility).tolist(),
        "retired_pair_count": diagnostic.count.tolist(),
        "retirement_euler_rms_per_year": diagnostic.root_mean_square.tolist(),
        "retirement_euler_maximum_absolute_per_year": diagnostic.maximum_absolute.tolist(),
        "minimum_assets": np.min(states[..., 0], axis=0).tolist(),
        "maximum_assets": np.max(states[..., 0], axis=0).tolist(),
        "minimum_log_human_capital": np.min(states[..., 1], axis=0).tolist(),
        "terminal_hours": controls[-1, :, 1].tolist(),
        "terminal_training_time": controls[-1, :, 2].tolist(),
    }
    return summary, diagnostic


def summarize_population_validation(validation_path: Path, output: Path):
    """Diagnose weighted retirement jitter from an existing paired cohort NPZ."""

    validation = json.loads(validation_path.read_text())
    weights = np.asarray(validation["cohort"]["weights"])
    people = weights.size
    report = {
        "source_validation": str(validation_path),
        "people": people,
        "metric": "Euler log-consumption growth residual per model year",
        "aggregation": "squared residuals weighted by initial type mass and eligible time pairs",
        "runs": [],
    }
    diagnostics = []
    times = []
    with np.load(validation_path.with_suffix(".npz")) as arrays:
        for run in validation["runs"]:
            if run["status"] != "evaluated":
                raise ValueError("all population runs must have evaluated successfully")
            prefix = run["array_prefix"]
            controls = arrays[f"{prefix}_controls"][:, :people]
            states = arrays[f"{prefix}_states"][:, :people]
            time = arrays[f"{prefix}_time"]
            result = retirement_euler_diagnostics(
                time,
                controls[..., 0],
                controls[..., 1],
                states[..., 0],
                ModelParams(**run["params"]),
                asset_minimum=run["config"]["asset_minimum"],
            )
            diagnostics.append(result)
            times.append(time)
            mass = float(result.count @ weights)
            rms = (
                float(
                    np.sqrt(
                        np.sum(np.where(result.eligible, result.residuals**2, 0) @ weights) / mass
                    )
                )
                if mass > 0
                else None
            )
            valid = result.count > 0
            report["runs"].append(
                {
                    "folder": run["folder"],
                    "weighted_eligible_pairs": mass,
                    "type_mass_with_eligible_pairs": float(weights[valid].sum()),
                    "weighted_pooled_euler_rms": rms,
                    "maximum_absolute_residual": (
                        float(np.max(result.maximum_absolute[valid])) if np.any(valid) else None
                    ),
                    "maximum_absolute_residual_person": (
                        int(np.nanargmax(result.maximum_absolute)) if np.any(valid) else None
                    ),
                }
            )
    if len(diagnostics) == 2 and np.array_equal(times[0], times[1]):
        common = diagnostics[0].eligible & diagnostics[1].eligible
        mass = float(np.sum(common @ weights))
        if mass > 0:
            rms = [
                float(np.sqrt(np.sum(np.where(common, d.residuals**2, 0) @ weights) / mass))
                for d in diagnostics
            ]
            report["common_retired_pairs"] = {
                "weighted_pair_count": mass,
                "weighted_pooled_euler_rms": rms,
                "rms_reduction_fraction": 1.0 - rms[1] / rms[0] if rms[0] > 0 else None,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/solver_benchmarks"))
    parser.add_argument(
        "--folders", nargs=2, default=["bicubic_padded_fine", "bicubic_asset121_fine"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--reuse-paths", action="store_true")
    parser.add_argument("--population-validation", type=Path)
    args = parser.parse_args()
    if args.population_validation is not None:
        report = summarize_population_validation(args.population_validation, args.output)
        print(json.dumps(report), flush=True)
        return
    archive = args.output.with_suffix(".npz")
    arrays = {
        "initial_assets": _INITIAL_ASSETS,
        "initial_human_capital": _INITIAL_CAPITAL,
    }
    if args.reuse_paths:
        with np.load(archive) as data:
            arrays.update({name: data[name] for name in data.files})
    references = []
    for name in ("direct_reference", "direct_type_low", "direct_type_high"):
        reference = json.loads((args.root / f"{name}.json").read_text())
        if not reference["diagnostics"]["accepted_success"]:
            raise ValueError(f"Independent reference {name} is not accepted")
        references.append(float(reference["lifetime_utility"]))
    direct_utilities = np.asarray(references)
    report = {
        "type_names": ["baseline", "low", "high"],
        "initial_assets": _INITIAL_ASSETS.tolist(),
        "initial_human_capital": _INITIAL_CAPITAL.tolist(),
        "direct_reference_utilities": direct_utilities.tolist(),
        "reference_note": "Accepted 48-interval independent paths, not mesh-converged truth.",
        "metric_definition": (
            "Adjacent h<1e-6 retired periods, all three asset endpoints above numerical floor"
            "+1e-6; residual dlog(c)/dt-(r-rho)/(1-consumption_power), per model year."
        ),
        "runs": {},
    }
    for folder in args.folders:
        solution, saved_report = load_solution(args.root / folder, args.platform)
        if not args.reuse_paths:
            simulation = simulate_cohort(solution, _INITIAL_ASSETS, _INITIAL_CAPITAL)
            arrays[f"{folder}_states"] = simulation.states
            arrays[f"{folder}_controls"] = simulation.controls
            arrays[f"{folder}_simulation_platform"] = np.asarray(simulation.solution.backend)
        arrays[f"{folder}_time"] = solution.time
        summary, diagnostic = summarize(
            solution,
            arrays[f"{folder}_states"],
            arrays[f"{folder}_controls"],
            direct_utilities,
        )
        summary["config"] = saved_report["config"]
        summary["simulation_platform"] = str(arrays[f"{folder}_simulation_platform"])
        report["runs"][folder] = summary
        arrays[f"{folder}_euler_residuals"] = diagnostic.residuals
        arrays[f"{folder}_euler_eligible"] = diagnostic.eligible
        print(folder, json.dumps(summary), flush=True)
    baseline, refined = (report["runs"][folder] for folder in args.folders)
    report["retirement_euler_rms_reduction_fraction"] = (
        1.0
        - np.asarray(refined["retirement_euler_rms_per_year"])
        / np.asarray(baseline["retirement_euler_rms_per_year"])
    ).tolist()
    report["realized_utility_change"] = (
        np.asarray(refined["lifetime_utility"]) - np.asarray(baseline["lifetime_utility"])
    ).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    with archive.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    archive.with_suffix(".npz.tmp").replace(archive)


if __name__ == "__main__":
    main()
