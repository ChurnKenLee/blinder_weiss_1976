"""Compare one synthetic F0 using IID cohorts, quadrature and mass transport.

Small CPU smoke run:
  PYTHONPATH=code/jax JAX_PLATFORMS=cpu python tools/benchmark_continuum.py --output DIR
GPU experiment using a recorded Bellman configuration (re-solves current code):
  PYTHONPATH=code/jax python tools/benchmark_continuum.py --platform gpu \
    --config-report output/solver_benchmarks/bicubic_padded_fine/report.json \
    --quadrature 8 16 32 --distribution 31 61 121 --output DIR
Targets and F0 are synthetic. This is a numerical calibration pilot, not an
estimate from ACS/ATUS or a proof of transport/grid convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax
import numpy as np
from blinder_weiss.bellman import BellmanConfig, solve_bellman
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationTargets,
    evaluate_calibration,
    fit_scalar_calibration,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    InitialAtom,
    SyntheticInitialDistribution,
    initial_quadrature,
    sample_initial_population,
    simulate_population,
)
from blinder_weiss.distribution import DistributionDomainError, DistributionGrid
from blinder_weiss.model import benchmark_params


class CompilationTimer:
    """Installed JAX's monitoring events, reported separately from wall time.

    Trace events may nest; durations are not subtracted from wall time. The
    backend-compile event covers compilation/cache lookup, not policy execution.
    First calls and warm calls are both timed; no cold-minus-warm estimate is
    mislabeled as exact compilation time.
    """

    def __init__(self):
        self.events: dict[str, float] = {}
        jax.monitoring.register_event_duration_secs_listener(self.record)

    def record(self, event: str, duration_secs: float, **_metadata: str | int):
        name, duration = event, duration_secs
        if name.startswith("/jax/core/compile/"):
            self.events[name.rsplit("/", 1)[-1]] = (
                self.events.get(name.rsplit("/", 1)[-1], 0.0) + duration
            )

    def measure(self, function):
        previous = self.events.copy()
        started = perf_counter()
        result = function()
        timing = {
            "wall_seconds": perf_counter() - started,
            "compilation_events_seconds": {
                key: value - previous.get(key, 0.0) for key, value in self.events.items()
            },
        }
        return result, timing


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--config-report", type=Path)
    parser.add_argument(
        "--initial-law-report",
        type=Path,
        help="Reuse the explicit initial_law in a previous population report",
    )
    parser.add_argument("--quadrature", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--distribution", type=int, nargs="*", default=[9, 17, 33])
    parser.add_argument("--people", type=int, default=256)
    parser.add_argument(
        "--distribution-asset-curvature",
        type=float,
        default=2.0,
        help="Independent forward asset-grid curvature; Bellman grid is unchanged",
    )
    parser.add_argument("--fit-evaluations", type=int, default=8)
    parser.add_argument("--skip-perturbations", action="store_true")
    args = parser.parse_args()
    if min(args.quadrature) < 2 or args.people < 1 or args.fit_evaluations < 0:
        parser.error("positive population sizes (quadrature >=2) and fit-evaluations >=0 required")
    if not np.isfinite(args.distribution_asset_curvature) or args.distribution_asset_curvature <= 0:
        parser.error("distribution asset curvature must be finite and positive")
    if any(size < 3 for size in args.distribution):
        parser.error("distribution grids require at least three nodes per dimension")
    if args.config_report:
        previous = json.loads(args.config_report.read_text())
        config = BellmanConfig(**{**previous["config"], "compute_platform": args.platform})
        params = benchmark_params(**previous["params"])
    else:
        config = BellmanConfig(
            periods=6,
            asset_nodes=9,
            human_capital_nodes=9,
            asset_minimum=1e-3,
            asset_maximum=12.0,
            log_human_capital_minimum=-1.5,
            log_human_capital_maximum=1.0,
            hours_nodes=5,
            investment_nodes=5,
            consumption_nodes=7,
            refinement_steps=4,
            control_batch_size=32,
            neighbor_policy_sweeps=2,
            compute_platform=args.platform,
        )
        params = benchmark_params(horizon=6.0)
    if args.initial_law_report is not None:
        law_data = json.loads(args.initial_law_report.read_text())["initial_law"]
        law = SyntheticInitialDistribution(
            **{**law_data, "atoms": tuple(InitialAtom(**atom) for atom in law_data["atoms"])}
        )
    else:
        law = SyntheticInitialDistribution()
    threshold = 0.02
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss"
    report: dict[str, Any] = {
        "purpose": "synthetic continuum and one-parameter numerical calibration pilot",
        "empirical_calibration": False,
        "transport_accepted_as_default": False,
        "jax_version": jax.__version__,
        "config": asdict(config),
        "params": params._asdict(),
        "initial_law": asdict(law),
        "initial_law_source": str(args.initial_law_report)
        if args.initial_law_report
        else "current synthetic default",
        "distribution_asset_curvature": args.distribution_asset_curvature,
        "seed": 125,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_root.glob("*.py")
        },
        "timing": "synchronous complete calls including policy recovery and host transfer; "
        "JAX compilation events recorded independently; trace durations may nest",
        "backends": {},
        "perturbations": [],
    }
    arrays: dict[str, np.ndarray] = {}

    def checkpoint():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "report.json")
        temporary = args.output / "profiles.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, allow_pickle=False, **arrays)
        temporary.replace(args.output / "profiles.npz")

    timer = CompilationTimer()
    print(json.dumps({"phase": "baseline policy solve"}), flush=True)
    solution, report["first_solve"] = timer.measure(lambda: solve_bellman(params, config))
    solution, report["warm_solve"] = timer.measure(lambda: solve_bellman(params, config))
    report["device"] = solution.device
    arrays["time"] = solution.time[:-1]
    arrays["state_time"] = solution.time
    arrays["cdf_assets"] = np.linspace(config.asset_minimum, config.asset_maximum, 201)
    arrays["cdf_log_human_capital"] = np.linspace(
        config.log_human_capital_minimum, config.log_human_capital_maximum, 201
    )
    results = {}
    configurations = {}
    specifications: list[tuple[Literal["cohort", "quadrature", "transport"], int]] = [
        ("cohort", args.people)
    ]
    for order in sorted(set(args.quadrature)):
        specifications.append(("quadrature", order))
    for size in sorted(set(args.distribution)):
        specifications.append(("transport", size))
    reference_nodes = initial_quadrature(
        law, nodes_per_dimension=max(args.quadrature), asset_floor=config.asset_minimum
    )
    for backend, size in specifications:
        label = f"{backend}_{size}"
        nodes = (
            sample_initial_population(law, people=size, asset_floor=config.asset_minimum)
            if backend == "cohort"
            else initial_quadrature(law, nodes_per_dimension=size, asset_floor=config.asset_minimum)
            if backend == "quadrature"
            else reference_nodes
        )
        grid = None
        if backend == "transport":
            grid = DistributionGrid(
                config.asset_minimum,
                config.asset_minimum
                + (config.asset_maximum - config.asset_minimum)
                * np.linspace(0.0, 1.0, size)[1:] ** args.distribution_asset_curvature,
                np.linspace(
                    config.log_human_capital_minimum, config.log_human_capital_maximum, size
                ),
            )
        configurations[label] = (nodes, backend, grid)
        print(json.dumps({"phase": "population", "backend": label}), flush=True)

        def population_call(
            nodes=nodes,
            backend: Literal["quadrature", "cohort", "transport"] = backend,
            grid=grid,
        ):
            return simulate_population(
                solution,
                nodes,
                backend=backend,
                distribution_grid=grid,
                participation_hours_threshold=threshold,
                store_snapshots=True,
            )

        try:
            result, first = timer.measure(population_call)
            result, warm = timer.measure(population_call)
        except DistributionDomainError as error:
            report["backends"][label] = {
                "completed": False,
                "accepted_for_calibration": False,
                "error": str(error),
                "diagnostics": error.diagnostics,
            }
            checkpoint()
            continue
        results[label] = result
        report["backends"][label] = {
            "completed": True,
            "accepted_for_calibration": False,
            "first_population": first,
            "warm_population": warm,
            "diagnostics": result.diagnostics,
        }
        for field in ("hours", "participation", "training_time", "consumption", "earnings"):
            arrays[f"{label}_{field}"] = getattr(result.moments, field)
        for field in ("assets", "human_capital", "log_human_capital", "asset_floor_mass"):
            arrays[f"{label}_{field}"] = getattr(result.state_moments, field)
        if backend == "transport":
            assert grid is not None
            weights = result.simulation.terminal_mass.ravel()
            state = grid.states
            indices = np.unique(np.linspace(0, config.periods, 7, dtype=int))
            arrays[f"{label}_snapshot_periods"] = indices
            arrays[f"{label}_snapshots"] = result.simulation.masses[indices]
            arrays[f"{label}_asset_nodes"] = grid.asset_nodes
            arrays[f"{label}_log_human_capital_nodes"] = grid.log_human_capital_nodes
        else:
            weights = result.simulation.weights
            state = result.simulation.states[-1]
        for field, component in (("assets", 0), ("log_human_capital", 1)):
            arrays[f"{label}_terminal_cdf_{field}"] = (
                state[:, component, None] <= arrays[f"cdf_{field}"][None, :]
            ).T @ weights
        checkpoint()
    reference_label = f"quadrature_{max(args.quadrature)}"
    reference = results[reference_label]
    target_times = np.linspace(0.0, reference.moments.time[-1], 7)
    scales = {
        "hours": 0.02,
        "participation": 0.02,
        "training_time": 0.02,
        "consumption": 0.1,
        "earnings": 0.1,
    }
    targets = CalibrationTargets(
        tuple(
            AgeMomentTarget(
                field,
                20.0 + target_times,
                np.interp(target_times, reference.moments.time, getattr(reference.moments, field)),
                scale,
                1.0,
                MOMENT_UNITS[field],
            )
            for field, scale in scales.items()
        ),
        20.0,
        threshold,
        f"synthetic same-policy target at leisure_weight={params.leisure_weight}; "
        f"initial quadrature order={max(args.quadrature)}; no survey data",
    )
    report["targets"] = asdict(targets)
    for label, result in results.items():
        report["backends"][label]["synthetic_loss"] = weighted_age_moment_loss(result, targets).loss
        report["backends"][label]["maximum_absolute_difference_from_refined_quadrature"] = {
            field: float(
                np.max(np.abs(arrays[f"{label}_{field}"] - arrays[f"{reference_label}_{field}"]))
            )
            for field in (
                *scales,
                "assets",
                "human_capital",
                "log_human_capital",
                "asset_floor_mass",
            )
        }
        report["backends"][label]["terminal_cdf_sup_difference"] = {
            field: float(
                np.max(
                    np.abs(
                        arrays[f"{label}_terminal_cdf_{field}"]
                        - arrays[f"{reference_label}_terminal_cdf_{field}"]
                    )
                )
            )
            for field in ("assets", "log_human_capital")
        }
    if not args.skip_perturbations:
        active = [f"cohort_{args.people}", reference_label]
        active += [f"transport_{max(args.distribution)}"] if args.distribution else []
        for label in active:
            if label not in results:
                continue
            nodes, backend, grid = configurations[label]
            for leisure_weight in [
                0.98 * params.leisure_weight,
                params.leisure_weight,
                1.02 * params.leisure_weight,
            ]:
                print(
                    json.dumps(
                        {
                            "phase": "solve and loss",
                            "backend": label,
                            "leisure_weight": leisure_weight,
                        }
                    ),
                    flush=True,
                )
                try:
                    evaluation, timing = timer.measure(
                        partial(
                            evaluate_calibration,
                            params._replace(leisure_weight=leisure_weight),
                            config,
                            nodes,
                            targets,
                            backend=backend,
                            distribution_grid=grid,
                            store_snapshots=True,
                        )
                    )
                except DistributionDomainError as error:
                    report["perturbations"].append(
                        {
                            "backend": label,
                            "leisure_weight": leisure_weight,
                            "completed": False,
                            "error": str(error),
                            "diagnostics": error.diagnostics,
                        }
                    )
                    checkpoint()
                    continue
                report["perturbations"].append(
                    {
                        "backend": label,
                        "leisure_weight": leisure_weight,
                        "completed": True,
                        "loss": evaluation.objective.loss,
                        "timing": timing,
                        "solve_seconds": evaluation.solve_seconds,
                        "population_and_loss_seconds": evaluation.population_and_loss_seconds,
                    }
                )
                checkpoint()
    if args.fit_evaluations:
        print(json.dumps({"phase": "synthetic scalar fit"}), flush=True)
        fit, timing = timer.measure(
            lambda: fit_scalar_calibration(
                "leisure_weight",
                (0.9 * params.leisure_weight, 1.1 * params.leisure_weight),
                params=params,
                config=config,
                initial_nodes=reference_nodes,
                targets=targets,
                parameter_tolerance=1e-3,
                max_evaluations=args.fit_evaluations,
            )
        )
        report["synthetic_fit"] = {
            "known_leisure_weight": params.leisure_weight,
            "fitted_leisure_weight": fit.x,
            "absolute_parameter_error": abs(fit.x - params.leisure_weight),
            "loss": fit.fun,
            "optimizer_success": bool(fit.success),
            "message": fit.message,
            "evaluations": [
                {
                    "leisure_weight": item.params.leisure_weight,
                    "loss": item.objective.loss,
                    "seconds": item.total_seconds,
                }
                for item in fit.evaluations
            ],
            "timing": timing,
        }
    checkpoint()
    print(json.dumps({"phase": "complete", "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
