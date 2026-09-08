"""Compare checkpoint feasibility, continuous feasibility, and time refinement.

The asset-minimum audit is independent NumPy algebra; it never calls the
production feasibility solver. Every comparison uses the same physical ages
and initial quadrature, including the numerical-floor component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from math import lcm
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from blinder_weiss import diagnose_bellman, solve_bellman
from blinder_weiss.bellman_smoothness import retirement_euler_diagnostics
from blinder_weiss.calibration import (
    MOMENT_UNITS,
    AgeMomentTarget,
    CalibrationTargets,
    weighted_age_moment_loss,
)
from blinder_weiss.continuum import (
    InitialAtom,
    PopulationResult,
    PopulationStateMoments,
    SyntheticInitialDistribution,
    initial_quadrature,
    simulate_population,
)
from blinder_weiss.model import ModelParams
from blinder_weiss.population import CohortMoments
from scipy.special import exprel
from validate_calibration_grid import (
    error_metrics,
    lifetime_utilities,
    load_solution,
    sample_trajectory,
)


def earnings_share(hours, training):
    """NumPy evaluation of h*g(q/h) on feasible controls, including retirement."""
    slope = np.sqrt(1.25) - 0.5
    hours, training = np.broadcast_arrays(hours, training)
    ratio = np.divide(training, hours, out=np.zeros_like(training), where=hours > 0)
    return hours - slope * training - slope**2 * training * ratio


def within_period_asset_minima(states, controls, params: ModelParams, step):
    """Return exact minima, minimizer times, and endpoint assets for fixed controls.

    For Y=h*g(q/h)*K, g=a*q-delta, and D=A'(s)*exp(-r*s),
    D(s)=r*A0+Y-c+Y*g*s*exprel((g-r)*s). Its derivative has the
    constant sign of Y*g, so an interior minimum exists precisely when D
    crosses from negative to positive. Both endpoints and that sole minimum
    suffice. The stationary time uses log1p with its removable zero limit.
    Leading state/control dimensions are batched; step must broadcast to them.
    """
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if states.shape[:-1] != controls.shape[:-1] or states.shape[-1] != 2 or controls.shape[-1] != 3:
        raise ValueError("states and controls must have matching batches and final dimensions 2/3")
    step = np.broadcast_to(np.asarray(step, dtype=float), states.shape[:-1])
    if not all(np.all(np.isfinite(x)) for x in (states, controls, step)) or np.any(step <= 0):
        raise ValueError("states, controls, and positive step must be finite")
    consumption, hours, training = np.moveaxis(controls, -1, 0)
    if (
        np.any(consumption <= 0)
        or np.any(hours < 0)
        or np.any(hours > 1)
        or np.any(training < 0)
        or np.any(training > hours + 1e-12)
    ):
        raise ValueError("controls must satisfy c>0 and 0<=q<=h<=1")
    assets = states[..., 0]
    income = earnings_share(hours, training) * np.exp(states[..., 1])
    growth = params.human_capital_productivity * training - params.human_capital_depreciation
    rate = params.interest_rate
    difference = growth - rate
    derivative0 = rate * assets + income - consumption
    derivative1 = derivative0 + income * growth * step * exprel(difference * step)
    interior = (derivative0 < 0) & (derivative1 > 0)
    denominator = np.where(interior, income * growth, 1.0)
    linear_root = np.where(interior, -derivative0 / denominator, 0.0)
    argument = np.where(interior, difference * linear_root, 0.0)
    small = np.abs(argument) < 1e-5
    safe_argument = np.where(small, 1.0, argument)
    ratio = np.where(
        small,
        1 - argument / 2 + argument**2 / 3 - argument**3 / 4 + argument**4 / 5,
        np.log1p(safe_argument) / safe_argument,
    )
    stationary_time = np.clip(linear_root * ratio, 0.0, step)

    def assets_at(duration):
        return (
            assets * np.exp(rate * duration)
            + income * np.exp(rate * duration) * duration * exprel(difference * duration)
            - consumption * duration * exprel(rate * duration)
        )

    endpoint = assets_at(step)
    stationary_assets = np.where(interior, assets_at(stationary_time), np.inf)
    candidates = np.stack((assets, endpoint, stationary_assets))
    minimum_index = np.argmin(candidates, axis=0)
    times = np.stack((np.zeros_like(step), step, stationary_time))
    minimum = np.take_along_axis(candidates, minimum_index[None], axis=0)[0]
    minimum_time = np.take_along_axis(times, minimum_index[None], axis=0)[0]
    if not all(np.all(np.isfinite(x)) for x in (minimum, minimum_time, endpoint)):
        raise ValueError("asset trajectories are not representable in float64")
    return minimum, minimum_time, endpoint


def path_audit(time, states, controls, params, floor, weights, tolerance):
    step = np.diff(time)[:, None]
    minimum, when, endpoint = within_period_asset_minima(states[:-1], controls, params, step)
    below = minimum < floor - tolerance
    people = np.any(below, axis=0)
    worst = np.unravel_index(np.argmin(minimum), minimum.shape)
    total_duration = float(time[-1] - time[0])
    result = {
        "minimum_assets": float(minimum.min()),
        "maximum_numerical_floor_violation": float(max(0.0, floor - minimum.min())),
        "maximum_economic_floor_violation": float(max(0.0, params.asset_floor - minimum.min())),
        "violating_interval_count": int(below.sum()),
        "weight_of_paths_with_any_numerical_floor_violation": float(weights[people].sum()),
        "weighted_fraction_of_intervals_with_violation": float(
            np.sum(step[:, 0] * (below @ weights)) / total_duration
        ),
        "maximum_transition_reproduction_error": float(np.max(np.abs(endpoint - states[1:, :, 0]))),
        "worst_period": int(worst[0]),
        "worst_type": int(worst[1]),
        "worst_physical_age": float(time[worst[0]] + when[worst]),
        "worst_control": controls[worst].tolist(),
        "worst_initial_state_A_logK": states[:-1][worst].tolist(),
        "feasibility_tolerance": tolerance,
        "interval_fraction_note": (
            "fraction of interval duration flagged; not time spent below floor"
        ),
    }
    return result, minimum, when


def common_profiles(time, states, controls, params, weights, state_time, moment_time, threshold):
    """Exact fixed-control states and step controls at shared physical times."""
    boundary_states, _ = sample_trajectory(time, states, controls, params, state_time)
    midpoint_states, midpoint_controls = sample_trajectory(
        time, states, controls, params, moment_time
    )
    consumption, hours, training = np.moveaxis(midpoint_controls, -1, 0)
    profiles = {
        "consumption": consumption @ weights,
        "hours": hours @ weights,
        "training_time": training @ weights,
        "participation": (hours > threshold) @ weights,
        "earnings": (earnings_share(hours, training) * np.exp(midpoint_states[..., 1])) @ weights,
        "assets": boundary_states[..., 0] @ weights,
        "human_capital": np.exp(boundary_states[..., 1]) @ weights,
        "log_human_capital": boundary_states[..., 1] @ weights,
    }
    return profiles, boundary_states, midpoint_controls


def sampled_population(native, profiles, state_time, moment_time):
    moments = CohortMoments(
        time=moment_time,
        hours=profiles["hours"],
        participation=profiles["participation"],
        training_time=profiles["training_time"],
        consumption=profiles["consumption"],
        earnings=profiles["earnings"],
        participation_hours_threshold=native.moments.participation_hours_threshold,
    )
    state_moments = PopulationStateMoments(
        time=state_time,
        assets=profiles["assets"],
        human_capital=profiles["human_capital"],
        log_human_capital=profiles["log_human_capital"],
        # Floor mass is an atom; compare only shared native boundaries below.
        asset_floor_mass=np.full(state_time.shape, np.nan),
    )
    return PopulationResult(
        native.backend, moments, state_moments, native.diagnostics, native.simulation
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False, default=lambda value: value.item()) + "\n"
    )
    temporary.replace(path)


def _write_arrays(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
    temporary.replace(path)


def save_solution(solution, folder, timings):
    folder.mkdir(parents=True, exist_ok=True)
    _write_arrays(
        folder / "policies.npz",
        {
            "values": solution.values,
            "c": solution.consumption_policy,
            "h": solution.hours_policy,
            "q": solution.training_time_policy,
            "A": solution.asset_grid,
            "y": solution.log_human_capital_grid,
            "time": solution.time,
        },
    )
    _write_json(
        folder / "report.json",
        {
            "config": asdict(solution.config),
            "params": solution.params._asdict(),
            "device": solution.device,
            "results": timings,
        },
    )


def audit_nodes(solution, tolerance):
    asset_mesh, capital_mesh = np.meshgrid(
        solution.asset_grid, solution.log_human_capital_grid, indexing="ij"
    )
    states = np.stack((asset_mesh, capital_mesh), axis=-1)
    minimum, count = np.inf, 0
    for period in range(solution.config.periods):
        controls = np.stack(
            (
                solution.consumption_policy[period],
                solution.hours_policy[period],
                solution.training_time_policy[period],
            ),
            axis=-1,
        )
        minima, _, _ = within_period_asset_minima(
            states, controls, solution.params, solution.params.horizon / solution.config.periods
        )
        minimum = min(minimum, float(minima.min()))
        count += int(np.sum(minima < solution.config.asset_minimum - tolerance))
    return {
        "state_age_controls_checked": int(solution.consumption_policy.size),
        "minimum_assets": minimum,
        "maximum_numerical_floor_violation": max(0.0, solution.config.asset_minimum - minimum),
        "maximum_economic_floor_violation": max(0.0, solution.params.asset_floor - minimum),
        "violating_state_age_controls": count,
        "tolerance": tolerance,
    }


def population_from_saved(solution, nodes, saved_run, stored_arrays, source_path):
    """Reconstruct the measurement interface from a verified prior rollout."""
    prefix = saved_run["array_prefix"]
    states, controls, time, values = (
        stored_arrays[f"{prefix}_{name}"]
        for name in ("states", "controls", "time", "policy_values")
    )
    if not np.array_equal(time, solution.time):
        raise ValueError("cached population time grid differs from the requested policy")
    weights = nodes.weights
    if not np.array_equal(states[0, :, 0], nodes.assets) or not np.allclose(
        np.exp(states[0, :, 1]), nodes.human_capital, rtol=2e-15, atol=0
    ):
        raise ValueError("cached population initial states differ from quadrature")
    consumption, hours, training = np.moveaxis(controls, -1, 0)
    moments = CohortMoments(
        time=time[:-1],
        consumption=consumption @ weights,
        hours=hours @ weights,
        training_time=training @ weights,
        participation=(hours > 0.02) @ weights,
        earnings=(earnings_share(hours, training) * np.exp(states[:-1, :, 1])) @ weights,
        participation_hours_threshold=0.02,
    )
    state_moments = PopulationStateMoments(
        time=time,
        assets=states[..., 0] @ weights,
        human_capital=np.exp(states[..., 1]) @ weights,
        log_human_capital=states[..., 1] @ weights,
        asset_floor_mass=stored_arrays[f"{prefix}_native_floor_mass"],
    )
    simulation = SimpleNamespace(
        time=time,
        states=states,
        controls=controls,
        policy_values=values,
        consumption=consumption,
        hours=hours,
        assets=states[..., 0],
        log_human_capital=states[..., 1],
    )
    return PopulationResult(
        "quadrature",
        moments,
        state_moments,
        {"saved_population_source": str(source_path)},
        simulation,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Saved checkpoint solution folder"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--periods", nargs="+", type=int, default=[140, 280])
    parser.add_argument(
        "--candidate-folders",
        nargs="*",
        type=Path,
        default=[],
        help="One folder per candidate; use - to solve that candidate",
    )
    parser.add_argument("--asset-nodes", nargs="+", type=int)
    parser.add_argument(
        "--reuse-populations",
        type=Path,
        help="Prior report with matching policy hashes and quadrature paths",
    )
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--common-periods", type=int)
    parser.add_argument("--warm-solves", type=int, default=1)
    parser.add_argument("--node-diagnostics", action="store_true")
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if min(args.periods) < 2 or args.quadrature < 2 or args.warm_solves < 0:
        parser.error("periods/quadrature must be at least two and warm-solves nonnegative")
    if not np.isfinite(args.feasibility_tolerance) or args.feasibility_tolerance < 0:
        parser.error("feasibility tolerance must be finite and nonnegative")
    if args.candidate_folders and len(args.candidate_folders) != len(args.periods):
        parser.error("provide exactly one candidate folder per requested period count")
    baseline, original = load_solution(args.baseline, args.platform)
    if original["config"].get("asset_feasibility", "checkpoints") != "checkpoints":
        parser.error("baseline must use checkpoint feasibility")
    baseline = replace(baseline, config=replace(baseline.config, asset_feasibility="checkpoints"))
    asset_counts = args.asset_nodes or [baseline.config.asset_nodes] * len(args.periods)
    if len(asset_counts) != len(args.periods) or min(asset_counts) < 2:
        parser.error("asset-nodes must specify one size >=2 per candidate")
    if args.periods[0] != baseline.config.periods or asset_counts[0] != baseline.config.asset_nodes:
        parser.error("first continuous candidate must retain baseline time and asset grids")
    if np.any(np.diff(args.periods) < 0) or np.any(np.diff(asset_counts) < 0):
        parser.error("time and asset grids must be nondecreasing")
    if any(
        a == b and c == d
        for a, b, c, d in zip(
            args.periods[:-1], args.periods[1:], asset_counts[:-1], asset_counts[1:], strict=True
        )
    ):
        parser.error("successive continuous candidates must differ in time or asset grid")
    common_periods = args.common_periods or lcm(baseline.config.periods, *args.periods)
    if common_periods < 1:
        parser.error("common-periods must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    law = SyntheticInitialDistribution(asset_floor_mass=0.05, atoms=(InitialAtom(5.0, 1.0, 0.05),))
    nodes = initial_quadrature(
        law, nodes_per_dimension=args.quadrature, asset_floor=baseline.config.asset_minimum
    )
    state_time = np.linspace(0.0, baseline.params.horizon, common_periods + 1)
    moment_time = (state_time[:-1] + state_time[1:]) / 2
    report: dict[str, Any] = {
        "purpose": (
            "boundary-feasibility correction and separate fixed-domain time-policy refinement"
        ),
        "convergence_claimed": False,
        "baseline_source": {
            "folder": str(args.baseline),
            "policy_sha256": _hash(args.baseline / "policies.npz"),
            "report_sha256": _hash(args.baseline / "report.json"),
            "recovery": "current code with explicit legacy checkpoint method",
        },
        "source_sha256": {
            p.name: _hash(p)
            for p in (Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss").glob("*.py")
        },
        "audit_source_sha256": _hash(__file__),
        "initial_law": asdict(law),
        "quadrature_order": args.quadrature,
        "common_state_time": state_time.tolist(),
        "common_moment_time": moment_time.tolist(),
        "alignment": (
            "analytic fixed-control states at common boundaries; "
            "step controls and exact K at common midpoints"
        ),
        "runs": [],
        "comparisons": [],
    }
    arrays = {
        "weights": nodes.weights,
        "initial_assets": nodes.assets,
        "initial_human_capital": nodes.human_capital,
        "initial_component": nodes.component,
        "common_state_time": state_time,
        "common_moment_time": moment_time,
    }
    cached_report, cached_arrays = None, None
    if args.reuse_populations:
        cached_report = json.loads(args.reuse_populations.read_text())
        if (
            cached_report["initial_law"] != json.loads(json.dumps(asdict(law)))
            or cached_report["quadrature_order"] != args.quadrature
        ):
            raise ValueError(
                "cached populations require exactly the same initial law and quadrature"
            )
        with np.load(args.reuse_populations.parent / "paths.npz") as archive:
            cached_arrays = {name: archive[name] for name in archive.files}
        if not np.array_equal(cached_arrays["weights"], nodes.weights):
            raise ValueError("cached population weights differ")
        report["population_reuse_source"] = {
            "report": str(args.reuse_populations),
            "report_sha256": _hash(args.reuse_populations),
            "paths_sha256": _hash(args.reuse_populations.parent / "paths.npz"),
        }
    solutions = [baseline]
    solution_folders = [args.baseline]
    for index, (periods, asset_count) in enumerate(zip(args.periods, asset_counts, strict=True)):
        expected = replace(
            baseline.config,
            periods=periods,
            asset_nodes=asset_count,
            asset_feasibility="continuous",
        )
        folder = args.candidate_folders[index] if args.candidate_folders else Path("-")
        if folder != Path("-"):
            solution, _ = load_solution(folder, args.platform)
            if solution.config != expected or solution.params != baseline.params:
                raise ValueError(
                    "candidate differs beyond declared feasibility, time, and asset settings"
                )
        else:
            config = expected
            timings = []
            for repetition in range(1 + args.warm_solves):
                started = perf_counter()
                solution = solve_bellman(baseline.params, config)
                timings.append(
                    {
                        "call": repetition,
                        "first_call_in_process": repetition == 0,
                        "wall_seconds": perf_counter() - started,
                        "solve_seconds": solution.solve_seconds,
                    }
                )
                print(json.dumps({"phase": "solve", "periods": periods, **timings[-1]}), flush=True)
            suffix = f"_asset{asset_count}" if asset_count != baseline.config.asset_nodes else ""
            folder = args.output / f"continuous_{periods}{suffix}"
            save_solution(solution, folder, timings)
        solutions.append(solution)
        solution_folders.append(folder)
    populations, profiles_by_run, raw_states, raw_controls = [], [], [], []
    for index, solution in enumerate(solutions):
        label = f"{solution.config.asset_feasibility}_{solution.config.periods}"
        if solution.config.asset_nodes != baseline.config.asset_nodes:
            label += f"_asset{solution.config.asset_nodes}"
        policy_hash = _hash(solution_folders[index] / "policies.npz")
        saved_run = None
        if cached_report is not None:
            saved_run = next(
                (
                    row
                    for row in cached_report["runs"]
                    if row.get("solution_policy_sha256") == policy_hash
                    and row["config"] == asdict(solution.config)
                    and row["params"] == solution.params._asdict()
                ),
                None,
            )
        if saved_run is not None:
            population = population_from_saved(
                solution, nodes, saved_run, cached_arrays, args.reuse_populations
            )
            seconds = None
        else:
            started = perf_counter()
            population = simulate_population(
                solution, nodes, backend="quadrature", participation_hours_threshold=0.02
            )
            seconds = perf_counter() - started
        sim = population.simulation
        audit, minima, minimum_times = path_audit(
            sim.time,
            sim.states,
            sim.controls,
            solution.params,
            solution.config.asset_minimum,
            nodes.weights,
            args.feasibility_tolerance,
        )
        profiles, states, controls = common_profiles(
            sim.time,
            sim.states,
            sim.controls,
            solution.params,
            nodes.weights,
            state_time,
            moment_time,
            0.02,
        )
        sampled = sampled_population(population, profiles, state_time, moment_time)
        euler = retirement_euler_diagnostics(
            sim.time,
            sim.consumption,
            sim.hours,
            sim.assets,
            solution.params,
            asset_minimum=solution.config.asset_minimum,
        )
        mass = float(euler.count @ nodes.weights)
        pooled_rms = (
            float(
                np.sqrt(
                    np.sum(np.where(euler.eligible, euler.residuals**2, 0) @ nodes.weights) / mass
                )
            )
            if mass > 0
            else None
        )
        utility = lifetime_utilities(sim.time, sim.states, sim.controls, solution.params)
        row = {
            "label": label,
            "array_prefix": f"run_{index}",
            "solution_folder": str(solution_folders[index]),
            "solution_policy_sha256": policy_hash,
            "population_reused_from": str(args.reuse_populations)
            if saved_run is not None
            else None,
            "config": asdict(solution.config),
            "params": solution.params._asdict(),
            "population_seconds_including_first_compile": seconds,
            "continuous_asset_path_audit": audit,
            "node_asset_path_audit": audit_nodes(solution, args.feasibility_tolerance),
            "weighted_realized_utility": float(utility @ nodes.weights),
            "maximum_absolute_value_gap": float(np.max(np.abs(sim.policy_values[0] - utility))),
            "weighted_retirement_euler_rms_per_year": pooled_rms,
            "weighted_eligible_retired_pairs": mass,
            "weighted_eligible_retired_pair_years": (
                mass * solution.params.horizon / solution.config.periods
            ),
            "minimum_log_human_capital": float(sim.log_human_capital.min()),
            "maximum_log_human_capital": float(sim.log_human_capital.max()),
        }
        if args.node_diagnostics:
            row["node_diagnostics"] = (
                saved_run["node_diagnostics"]
                if saved_run is not None and "node_diagnostics" in saved_run
                else diagnose_bellman(solution).as_dict()
            )
        report["runs"].append(row)
        for key, value in {
            "time": sim.time,
            "states": sim.states,
            "controls": sim.controls,
            "utilities": utility,
            "policy_values": sim.policy_values,
            "analytic_asset_minima": minima,
            "analytic_minimum_times": minimum_times,
            "native_floor_mass": population.state_moments.asset_floor_mass,
        }.items():
            arrays[f"run_{index}_{key}"] = value
        for key, value in profiles.items():
            arrays[f"run_{index}_common_{key}"] = value
        populations.append(sampled)
        profiles_by_run.append(profiles)
        raw_states.append(states)
        raw_controls.append(controls)
        print(json.dumps({"phase": "population", "label": label, "audit": audit}), flush=True)
        _write_json(args.output / "report.json", report)
        _write_arrays(args.output / "paths.npz", arrays)
    scales = {
        "consumption": 0.1,
        "hours": 0.02,
        "training_time": 0.02,
        "participation": 0.02,
        "earnings": 0.1,
    }
    target_indices = np.unique(np.linspace(0, moment_time.size - 1, 7, dtype=int))
    targets = CalibrationTargets(
        tuple(
            AgeMomentTarget(
                name,
                20 + moment_time[target_indices],
                profiles_by_run[-1][name][target_indices],
                scale,
                1.0,
                MOMENT_UNITS[name],
            )
            for name, scale in scales.items()
        ),
        20.0,
        0.02,
        "synthetic final continuous-candidate quadrature reference; no empirical data",
    )
    report["loss_targets"] = {
        "source": targets.source,
        "age_origin": 20,
        "model_ages": moment_time[target_indices].tolist(),
        "scales": scales,
        "values": {name: profiles_by_run[-1][name][target_indices].tolist() for name in scales},
    }
    for row, population in zip(report["runs"], populations, strict=True):
        row["standardized_common_age_loss"] = weighted_age_moment_loss(population, targets).loss
    for index in range(1, len(solutions)):
        earlier, later = solutions[index - 1], solutions[index]
        first, second = report["runs"][index - 1], report["runs"][index]
        first_time, second_time = earlier.time, later.time
        # Native-boundary intersection: never interpolate probability in an atom.
        floor_time = np.linspace(
            0, earlier.params.horizon, np.gcd(earlier.config.periods, later.config.periods) + 1
        )
        first_indices = np.rint(
            floor_time / (earlier.params.horizon / earlier.config.periods)
        ).astype(int)
        second_indices = np.rint(floor_time / (later.params.horizon / later.config.periods)).astype(
            int
        )
        if not np.allclose(first_time[first_indices], floor_time) or not np.allclose(
            second_time[second_indices], floor_time
        ):
            raise ValueError("floor-mass comparison requires shared native physical-age boundaries")
        comparison = {
            "from": first["label"],
            "to": second["label"],
            "kind": (
                "feasibility_method"
                if index == 1
                else "joint_time_asset_refinement"
                if earlier.config.periods != later.config.periods
                and earlier.config.asset_nodes != later.config.asset_nodes
                else "time_refinement"
                if earlier.config.periods != later.config.periods
                else "asset_refinement"
            ),
            "configuration_changes": {
                key: {"from": first["config"][key], "to": value}
                for key, value in second["config"].items()
                if value != first["config"][key]
            },
            "moment_errors": {
                name: error_metrics(profiles_by_run[index][name], profiles_by_run[index - 1][name])
                for name in profiles_by_run[index]
            },
            "cohort_state_errors": {
                name: error_metrics(
                    raw_states[index][..., axis], raw_states[index - 1][..., axis], nodes.weights
                )
                for axis, name in enumerate(("assets", "log_human_capital"))
            },
            "cohort_control_errors": {
                name: error_metrics(
                    raw_controls[index][..., axis],
                    raw_controls[index - 1][..., axis],
                    nodes.weights,
                )
                for axis, name in enumerate(("consumption", "hours", "training_time"))
            },
            "floor_mass_errors_at_common_native_boundaries": error_metrics(
                arrays[f"run_{index}_native_floor_mass"][second_indices],
                arrays[f"run_{index - 1}_native_floor_mass"][first_indices],
            ),
            "floor_mass_comparison_times": floor_time.tolist(),
            "weighted_realized_utility_change": second["weighted_realized_utility"]
            - first["weighted_realized_utility"],
        }
        report["comparisons"].append(comparison)
    report["limitations"] = [
        "Checkpoint policies may earn higher utility by violating the continuous constraint.",
        "The finest time grid is a numerical reference, not converged truth.",
        (
            "Only explicitly declared time/asset settings change; "
            "control search, quadrature, and domain remain fixed."
        ),
        "Standardized loss scales are numerical choices, not survey uncertainty estimates.",
    ]
    _write_json(args.output / "report.json", report)
    _write_arrays(args.output / "paths.npz", arrays)
    print(json.dumps({"phase": "complete", "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
