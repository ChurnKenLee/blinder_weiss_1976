"""Compare saved Bellman grids on fixed heterogeneous paths and model moments.

This reports numerical evidence, not a calibration or convergence certificate.
Run only after the desired solver implementation and saved grids are finalized.
Saved policies are recovered using the current implementation on --platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import jax
import numpy as np
from blinder_weiss import (
    BellmanConfig,
    BellmanConvergenceConfig,
    BellmanSolution,
    ModelParams,
    simulate_cohort,
)
from scipy.integrate import solve_ivp

_REFERENCE_NAMES = ("direct_reference", "direct_type_low", "direct_type_high")
_REFERENCE_STATES = ((5.0, 1.0), (2.0, 0.8), (8.0, 1.2))


def fixed_cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the 256-type seed-125 cohort in benchmark_calibration.py."""

    random = np.random.default_rng(125)
    assets = random.uniform(2.0, 8.0, 256)
    capital = np.exp(random.uniform(-0.25, 0.25, 256))
    weights = random.uniform(0.5, 1.5, 256)
    return assets, capital, weights / weights.sum()


def _exprel(value: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(value) > 1e-7, value, 1.0)
    series = 1.0 + value / 2.0 + value**2 / 6.0 + value**3 / 24.0
    return np.where(np.abs(value) > 1e-7, np.expm1(value) / safe, series)


def _earnings_share(hours: np.ndarray, training: np.ndarray) -> np.ndarray:
    slope = np.sqrt(1.25) - 0.5
    quadratic = training <= hours
    safe_hours = np.where((hours > 0.0) & quadratic, hours, 1.0)
    loss = np.where(
        quadratic, slope**2 * training**2 / safe_hours, slope**2 * (2.0 * training - hours)
    )
    return hours - slope * training - loss


def sample_trajectory(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    params: ModelParams,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample exact within-period states and right-continuous step controls.

    At the terminal time the returned control is from the final interval;
    control comparisons in this tool use interior common-interval midpoints.
    """

    time = np.asarray(time)
    sample_times = np.asarray(sample_times)
    if (
        np.any(np.diff(time) <= 0.0)
        or np.any(sample_times < time[0])
        or np.any(sample_times > time[-1])
    ):
        raise ValueError("sample times must lie inside a strictly increasing trajectory horizon")
    periods = np.clip(np.searchsorted(time, sample_times, side="right") - 1, 0, len(time) - 2)
    duration = (sample_times - time[periods])[:, None]
    initial = states[periods]
    sampled_controls = controls[periods]
    consumption, hours, training = np.moveaxis(sampled_controls, -1, 0)
    growth = params.human_capital_productivity * training - params.human_capital_depreciation
    asset_factor = np.exp(params.interest_rate * duration)
    earnings = _earnings_share(hours, training) * np.exp(initial[..., 1])
    assets = (
        asset_factor * initial[..., 0]
        + asset_factor * duration * _exprel((growth - params.interest_rate) * duration) * earnings
        - consumption * duration * _exprel(params.interest_rate * duration)
    )
    log_capital = initial[..., 1] + growth * duration
    return np.stack((assets, log_capital), axis=-1), sampled_controls


def lifetime_utilities(
    time: np.ndarray, states: np.ndarray, controls: np.ndarray, params: ModelParams
) -> np.ndarray:
    """Realized discounted utility under the simulated constant controls."""

    duration = np.diff(time)
    consumption, hours = controls[..., 0], controls[..., 1]
    flow = (
        params.consumption_weight * consumption**params.consumption_power / params.consumption_power
        + params.leisure_weight * (1.0 - hours) ** params.leisure_power / params.leisure_power
    )
    discount_integral = np.exp(-params.rho * time[:-1]) * duration * _exprel(-params.rho * duration)
    terminal = (
        params.bequest_weight * states[-1, :, 0] ** params.bequest_power / params.bequest_power
    )
    return (
        np.sum(discount_integral[:, None] * flow, axis=0)
        + np.exp(-params.rho * time[-1]) * terminal
    )


def error_metrics(
    actual: np.ndarray, reference: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    """Absolute and signed errors; optional person weights leave time uniform."""

    difference = np.asarray(actual) - np.asarray(reference)
    if difference.size == 0 or not np.all(np.isfinite(difference)):
        raise ValueError("comparison arrays must be nonempty and finite")
    if weights is None:
        mean_square = np.mean(difference**2)
        mean_absolute = np.mean(np.abs(difference))
        mean_signed = np.mean(difference)
    else:
        weights = np.asarray(weights)
        if difference.ndim != 2 or weights.shape != (difference.shape[1],):
            raise ValueError("person weights require comparison arrays of shape (time, people)")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)) or not weights.sum() > 0.0:
            raise ValueError("person weights must be finite, nonnegative, and have positive mass")
        weights = weights / weights.sum()
        mean_square = np.mean((difference**2) @ weights)
        mean_absolute = np.mean(np.abs(difference) @ weights)
        mean_signed = np.mean(difference @ weights)
    return {
        "rmse": float(np.sqrt(mean_square)),
        "mean_absolute": float(mean_absolute),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "mean_signed": float(mean_signed),
        "maximum_index": list(np.unravel_index(np.argmax(np.abs(difference)), difference.shape)),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_solution(folder: Path, platform: str) -> tuple[BellmanSolution, dict[str, Any]]:
    report = json.loads((folder / "report.json").read_text())
    config = replace(BellmanConfig(**report["config"]), compute_platform=platform, device_index=0)
    devices = jax.devices(platform)
    device = devices[0]
    with np.load(folder / "policies.npz") as arrays:
        solution = BellmanSolution(
            params=ModelParams(**report["params"]),
            config=config,
            time=arrays["time"],
            asset_grid=arrays["A"],
            log_human_capital_grid=arrays["y"],
            values=arrays["values"],
            consumption_policy=arrays["c"],
            hours_policy=arrays["h"],
            training_time_policy=arrays["q"],
            solve_seconds=report["results"][-1]["solve_seconds"],
            backend=device.platform,
            device=f"{device.platform}:{device.id} ({device.device_kind})",
        )
    expected = (config.periods, config.asset_nodes, config.human_capital_nodes)
    if solution.values.shape != (config.periods + 1, *expected[1:]) or any(
        array.shape != expected
        for array in (
            solution.consumption_policy,
            solution.hours_policy,
            solution.training_time_policy,
        )
    ):
        raise ValueError("saved policy/value shapes do not match report configuration")
    return solution, report


def _economics(params: ModelParams) -> dict[str, float]:
    return {
        key: value
        for key, value in params._asdict().items()
        if key not in {"initial_assets", "initial_human_capital"}
    }


def _boundary_metrics(solution: BellmanSolution, states: np.ndarray, weights: np.ndarray) -> dict:
    result = {}
    for axis, name, grid in (
        (0, "assets", solution.asset_grid),
        (1, "log_human_capital", solution.log_human_capital_grid),
    ):
        values = states[..., axis]
        result[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "minimum_lower_boundary_slack": float(np.min(values - grid[0])),
            "minimum_upper_boundary_slack": float(np.min(grid[-1] - values)),
            "weighted_lower_edge_cell_fraction": float(np.mean((values <= grid[1]) @ weights)),
            "weighted_upper_edge_cell_fraction": float(np.mean((values >= grid[-2]) @ weights)),
            "lower_edge_cell": grid[:2].tolist(),
            "upper_edge_cell": grid[-2:].tolist(),
        }
    return result


def _reference_comparison(
    root: Path,
    name: str,
    solution: BellmanSolution,
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    realized_utility: float,
) -> dict[str, Any]:
    metadata_path, arrays_path = root / f"{name}.json", root / f"{name}.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        return {"status": "missing_reference"}
    metadata = json.loads(metadata_path.read_text())
    if (
        not metadata.get("reference_accepted", False)
        or not metadata["diagnostics"]["accepted_success"]
        or not metadata.get("comparison_usable", True)
    ):
        return {"status": "reference_not_accepted", "diagnostics": metadata.get("diagnostics")}
    if _economics(solution.params) != _economics(ModelParams(**metadata["params"])):
        return {"status": "economic_parameters_differ"}
    with np.load(arrays_path) as source:
        reference = dict(source)
    if not np.allclose(
        [states[0, 0], np.exp(states[0, 1])],
        [reference["A"][0], reference["K"][0]],
        rtol=0.0,
        atol=1e-10,
    ):
        return {"status": "initial_states_differ"}
    # Direct controls are linear between collocation nodes. Integrate them
    # independently instead of linearly interpolating curved state paths.
    reference_params = ModelParams(**metadata["params"])

    def direct_dynamics(age, state):
        consumption, hours, training = [
            np.interp(age, reference["time"], reference[key]) for key in ("c", "h", "q")
        ]
        earnings = _earnings_share(np.asarray(hours), np.asarray(training)) * np.exp(state[1])
        return [
            reference_params.interest_rate * state[0] + earnings - consumption,
            reference_params.human_capital_productivity * training
            - reference_params.human_capital_depreciation,
        ]

    integrated = solve_ivp(
        direct_dynamics,
        (float(reference["time"][0]), float(reference["time"][-1])),
        [reference["A"][0], np.log(reference["K"][0])],
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        t_eval=time,
    )
    if not integrated.success or not np.all(np.isfinite(integrated.y)):
        return {"status": "reference_integration_failed", "message": integrated.message}
    midpoints = (time[:-1] + time[1:]) / 2.0
    errors = {}
    for field, actual, sample_times in (
        ("A", states[:, 0], time),
        ("K", np.exp(states[:, 1]), time),
        ("c", controls[:, 0], midpoints),
        ("h", controls[:, 1], midpoints),
        ("q", controls[:, 2], midpoints),
    ):
        expected = (
            integrated.y[0]
            if field == "A"
            else np.exp(integrated.y[1])
            if field == "K"
            else np.interp(sample_times, reference["time"], reference[field])
        )
        errors[field] = error_metrics(actual, expected)
    reference_utility = metadata["lifetime_utility"]
    return {
        "status": "compared",
        "reference_accepted": True,
        "reference_mesh_convergence_verified": metadata.get("mesh_convergence_verified", False),
        "reference_diagnostics": metadata["diagnostics"],
        "reference_sha256": {"json": _hash(metadata_path), "npz": _hash(arrays_path)},
        "path_errors": errors,
        "realized_utility": realized_utility,
        "reference_utility": reference_utility,
        "reference_minus_realized_utility": reference_utility - realized_utility,
        "state_alignment": (
            "independent DOP853 integration of linear direct controls; rtol1e-10, atol1e-12"
        ),
        "control_alignment": "linear direct controls evaluated at Bellman interval midpoints",
    }


def validate_folders(
    folders: list[Path], root: Path, output: Path, platform: str, participation_threshold: float
) -> dict[str, Any]:
    assets, capital, weights = fixed_cohort()
    probe_assets, probe_capital = np.asarray(_REFERENCE_STATES).T
    initial_assets = np.concatenate((assets, probe_assets))
    initial_capital = np.concatenate((capital, probe_capital))
    simulation_weights = np.concatenate((weights, np.zeros(3)))
    report: dict[str, Any] = {
        "purpose": "grid/domain refinement evidence; no automatic accuracy certificate",
        "cohort": {
            "seed": 125,
            "people": 256,
            "assets": assets.tolist(),
            "human_capital": capital.tolist(),
            "weights": weights.tolist(),
        },
        "participation_definition": f"hours > {participation_threshold} in model time units",
        "reference_probes": dict(zip(_REFERENCE_NAMES, _REFERENCE_STATES, strict=True)),
        "existing_convergence_settings": asdict(BellmanConvergenceConfig()),
        "scope": (
            "fixed benchmark economics and initial-state distribution; "
            "model moments, not survey units"
        ),
        "recovery": "current solver implementation applied to each saved value/policy grid",
        "runs": [],
        "successive_grid_comparisons": [],
    }
    project = Path(__file__).resolve().parents[1]
    report["current_source_sha256"] = {
        str(path.relative_to(project)): _hash(path)
        for path in (project / "code/jax/blinder_weiss").glob("*.py")
    }
    arrays: dict[str, np.ndarray] = {}
    previous = None
    economics = None
    for index, folder in enumerate(folders):
        row: dict[str, Any] = {"folder": str(folder), "status": "pending"}
        report["runs"].append(row)
        try:
            solution, source = load_solution(folder, platform)
            if economics is not None and _economics(solution.params) != economics:
                raise ValueError("cross-grid comparison requires identical economic parameters")
            economics = _economics(solution.params)
            row.update(
                config=source["config"],
                params=source["params"],
                device=solution.device,
                source_sha256={
                    "report": _hash(folder / "report.json"),
                    "policies": _hash(folder / "policies.npz"),
                },
                saved_node_diagnostics=source.get("diagnostics"),
            )
            started = perf_counter()
            cohort = simulate_cohort(
                solution, initial_assets, initial_capital, weights=simulation_weights
            )
            row["rollout_wall_seconds_including_first_compile"] = perf_counter() - started
            common_time = np.linspace(0.0, solution.params.horizon, 281)
            common_midpoints = (common_time[:-1] + common_time[1:]) / 2.0
            states, _ = sample_trajectory(
                cohort.time, cohort.states, cohort.controls, solution.params, common_time
            )
            middle_states, controls = sample_trajectory(
                cohort.time, cohort.states, cohort.controls, solution.params, common_midpoints
            )
            utilities = lifetime_utilities(
                cohort.time, cohort.states, cohort.controls, solution.params
            )
            path_fields = {
                "A": states[:, :256, 0],
                "K": np.exp(states[:, :256, 1]),
                "c": controls[:, :256, 0],
                "h": controls[:, :256, 1],
                "q": controls[:, :256, 2],
            }
            moments = {key: path_fields[key] @ weights for key in ("c", "h", "q")}
            moments["participation"] = (path_fields["h"] > participation_threshold) @ weights
            moments["earnings"] = (
                _earnings_share(path_fields["h"], path_fields["q"])
                * np.exp(middle_states[:, :256, 1])
            ) @ weights
            value_gap = cohort.policy_values[0, :256] - utilities[:256]
            row["cohort"] = {
                "weighted_realized_utility": float(utilities[:256] @ weights),
                "recovered_initial_value_minus_realized_utility": {
                    "weighted_mean": float(value_gap @ weights),
                    "minimum": float(value_gap.min()),
                    "maximum": float(value_gap.max()),
                    "maximum_absolute": float(np.max(np.abs(value_gap))),
                    "minimum_index": int(np.argmin(value_gap)),
                    "maximum_index": int(np.argmax(value_gap)),
                    "maximum_absolute_index": int(np.argmax(np.abs(value_gap))),
                },
                "minimum_consumption_capacity_slack": cohort.minimum_consumption_capacity_slack,
                "boundary_visitation": _boundary_metrics(solution, states[:, :256], weights),
                "moments": {key: value.tolist() for key, value in moments.items()},
            }
            row["direct_references"] = {
                name: _reference_comparison(
                    root,
                    name,
                    solution,
                    cohort.time,
                    cohort.states[:, 256 + probe],
                    cohort.controls[:, 256 + probe],
                    float(utilities[256 + probe]),
                )
                for probe, name in enumerate(_REFERENCE_NAMES)
            }
            prefix = f"run_{index}"
            row["array_prefix"] = prefix
            arrays.update(
                {
                    f"{prefix}_time": cohort.time,
                    f"{prefix}_states": cohort.states,
                    f"{prefix}_controls": cohort.controls,
                    f"{prefix}_utilities": utilities,
                    f"{prefix}_policy_values": cohort.policy_values,
                }
            )
            arrays["common_state_time"] = common_time
            arrays["common_moment_time"] = common_midpoints
            if previous is not None:
                (
                    previous_name,
                    previous_paths,
                    previous_moments,
                    previous_utilities,
                    previous_config,
                ) = previous
                report["successive_grid_comparisons"].append(
                    {
                        "from": previous_name,
                        "to": str(folder),
                        "configuration_changes": {
                            key: {"from": previous_config.get(key), "to": value}
                            for key, value in source["config"].items()
                            if previous_config.get(key) != value
                        },
                        "cohort_path_errors": {
                            key: error_metrics(path_fields[key], previous_paths[key], weights)
                            for key in path_fields
                        },
                        "moment_errors": {
                            key: error_metrics(moments[key], previous_moments[key])
                            for key in moments
                        },
                        "realized_utility_change": error_metrics(
                            utilities[:256][None, :], previous_utilities[None, :], weights
                        ),
                    }
                )
            previous = (str(folder), path_fields, moments, utilities[:256], source["config"])
            row["status"] = "evaluated"
            print(
                json.dumps(
                    {
                        "folder": str(folder),
                        "status": row["status"],
                        "weighted_utility": row["cohort"]["weighted_realized_utility"],
                    }
                ),
                flush=True,
            )
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            row.update(status="evaluation_failed", error=f"{type(error).__name__}: {error}")
            print(
                json.dumps({"folder": str(folder), "status": row["status"], "error": row["error"]}),
                flush=True,
            )
    report["common_state_time"] = arrays.get("common_state_time", np.asarray([])).tolist()
    report["common_moment_time"] = arrays.get("common_moment_time", np.asarray([])).tolist()
    report["moment_alignment"] = (
        "uniform common-interval midpoints, piecewise-constant controls and exact human capital"
    )
    report["path_alignment"] = (
        "exact constant-control states at common boundaries; step controls at common midpoints"
    )
    report["direct_reference_limit"] = (
        "three validated paths are not a population-wide error bound or mesh-converged truth"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".npz")
    temp = sidecar.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
    temp.replace(sidecar)
    report["trajectory_artifact"] = str(sidecar)
    temporary_json = output.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=lambda value: value.item()) + "\n"
    )
    temporary_json.replace(output)
    return report


def recovery_gap_decomposition(
    solution: BellmanSolution,
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    policy_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose a saved rollout's value gap without reoptimizing its controls.

    Q0 - realized utility = sum(k=1..N-1) exp(-rho*t[k]) (V[k] - Q[k]).
    Q[k] is independently reevaluated from the saved controls and next state.
    Signed interpolation/optimization defects are retained, including gains.
    """

    from blinder_weiss.bellman import _interpolate_value_jax

    if (
        states.shape != (len(time), *controls.shape[1:-1], 2)
        or policy_values.shape != controls.shape[:-1]
    ):
        raise ValueError("rollout state, control, and policy-value shapes must agree")
    if not np.array_equal(time, solution.time):
        raise ValueError("rollout and saved value table must use the same times")
    params = solution.params
    grid_a, grid_y = (
        jax.numpy.asarray(solution.asset_grid),
        jax.numpy.asarray(solution.log_human_capital_grid),
    )
    interpolate = jax.jit(
        lambda table, query: _interpolate_value_jax(
            table, grid_a, grid_y, query[..., 0], query[..., 1], solution.config
        )
    )
    represented = np.stack(
        [
            np.asarray(interpolate(solution.values[k], states[k]))
            for k in range(solution.config.periods)
        ]
    )
    terminal = (
        params.bequest_weight * states[-1, :, 0] ** params.bequest_power / params.bequest_power
    )
    all_values = np.concatenate((represented, terminal[None]), axis=0)
    duration = np.diff(time)
    consumption, hours = controls[..., 0], controls[..., 1]
    flow = (
        params.consumption_weight * consumption**params.consumption_power / params.consumption_power
        + params.leisure_weight * (1.0 - hours) ** params.leisure_power / params.leisure_power
    )
    reevaluated_q = (duration * _exprel(-params.rho * duration))[:, None] * flow + np.exp(
        -params.rho * duration
    )[:, None] * all_values[1:]
    discounted_defects = np.exp(-params.rho * time[:-1])[:, None] * (represented - reevaluated_q)
    realized = lifetime_utilities(time, states, controls, params)
    initial_gap = reevaluated_q[0] - realized
    return {
        "represented_values": represented,
        "reevaluated_policy_values": reevaluated_q,
        "saved_policy_value_error": policy_values - reevaluated_q,
        "discounted_defects": discounted_defects,
        "initial_value_gap": initial_gap,
        "telescoping_error": initial_gap - discounted_defects[1:].sum(axis=0),
    }


def audit_saved_recovery(
    validation_path: Path,
    output: Path,
    platform: str = "cpu",
    run_index: int = -1,
    top_types: int = 3,
    states_per_type: int = 4,
    refinement_starts: int = 4,
    refinement_steps: int = 192,
) -> dict[str, Any]:
    """Test stronger recovery at saved states, keeping continuation tables fixed.

    This is a lower bound on search regret at selected states, not a new rollout
    or an upper error bound. A negative V-Q defect cannot be caused by too little
    maximization alone: an improved feasible Q makes that defect more negative.
    """

    from blinder_weiss import greedy_policy_value_at

    if min(top_types, states_per_type, refinement_starts, refinement_steps) < 1:
        raise ValueError("audit counts and refinement settings must be positive")
    validation = json.loads(validation_path.read_text())
    run = validation["runs"][run_index]
    folder = Path(run["folder"])
    if _hash(folder / "policies.npz") != run["source_sha256"]["policies"]:
        raise ValueError("saved value/policy table differs from the validation source")
    solution, source = load_solution(folder, platform)
    stronger = replace(
        solution,
        config=replace(
            solution.config,
            refinement_starts=refinement_starts,
            refinement_steps=refinement_steps,
        ),
    )
    sidecar = validation_path.with_suffix(".npz")
    prefix = run["array_prefix"]
    with np.load(sidecar) as arrays:
        time, states, controls, policy_values = (
            arrays[f"{prefix}_{key}"] for key in ("time", "states", "controls", "policy_values")
        )
    decomposition = recovery_gap_decomposition(solution, time, states, controls, policy_values)
    people = validation["cohort"]["people"]
    gaps = decomposition["initial_value_gap"][:people]
    selected = list(
        dict.fromkeys(
            [
                int(np.argmin(gaps)),
                int(np.argmax(gaps)),
                *np.argsort(np.abs(gaps))[::-1][:top_types].tolist(),
            ]
        )
    )
    report: dict[str, Any] = {
        "purpose": "same-table search-regret lower bounds and signed along-path value defects",
        "source_validation": str(validation_path),
        "source_sha256": {
            "validation": _hash(validation_path),
            "trajectory": _hash(sidecar),
            "policies": _hash(folder / "policies.npz"),
        },
        "folder": str(folder),
        "device": solution.device,
        "current_source_sha256": {
            path.name: _hash(path)
            for path in (Path(__file__).resolve().parents[1] / "code/jax/blinder_weiss").glob(
                "*.py"
            )
        },
        "saved_config": source["config"],
        "recovery_config_changes": {
            "refinement_starts": refinement_starts,
            "refinement_steps": refinement_steps,
        },
        "identity": "Q0-U = sum(k=1..N-1) exp(-rho*t[k])*(V[k]-Q[k])",
        "maximum_saved_policy_value_recomputation_error": float(
            np.max(np.abs(decomposition["saved_policy_value_error"]))
        ),
        "maximum_telescoping_error": float(np.max(np.abs(decomposition["telescoping_error"]))),
        "selection": (
            "minimum/maximum signed gap and top absolute gaps; initial state plus "
            "largest absolute discounted defects after period zero"
        ),
        "types": [],
        "limitations": (
            "Fixed paths and continuation values; sampled feasible improvements are "
            "lower bounds on optimizer regret, not global optima or population convergence."
        ),
    }
    for person in selected:
        defects = decomposition["discounted_defects"][:, person]
        periods = [0, *(np.argsort(np.abs(defects[1:]))[::-1][:states_per_type] + 1).tolist()]
        type_report: dict[str, Any] = {
            "person": person,
            "initial_assets": float(states[0, person, 0]),
            "initial_human_capital": float(np.exp(states[0, person, 1])),
            "value_gap": float(gaps[person]),
            "positive_discounted_defect_sum": float(np.maximum(defects[1:], 0).sum()),
            "negative_discounted_defect_sum": float(np.minimum(defects[1:], 0).sum()),
            "states": [],
        }
        for period in periods:
            exact = states[period, person]
            # The simulator accepts 1e-10 boundary roundoff; public recovery
            # requires exact containment. Record this sub-tolerance adjustment.
            query = np.clip(
                exact,
                [solution.asset_grid[0], solution.log_human_capital_grid[0]],
                [solution.asset_grid[-1], solution.log_human_capital_grid[-1]],
            )
            if np.max(np.abs(query - exact)) > 1e-10:
                raise ValueError("saved audit state is outside its computational domain")
            results = []
            for candidate in (solution, stronger):
                recovered = greedy_policy_value_at(
                    candidate, int(period), query[0:1], np.exp(query[1:2])
                )
                results.append([float(array[0]) for array in recovered])
            baseline, strong = results
            original_q = float(decomposition["reevaluated_policy_values"][period, person])
            difference = strong[0] - original_q
            row = {
                "period": int(period),
                "model_age": float(time[period]),
                "state_A_logK": exact.tolist(),
                "query_roundoff_adjustment": (query - exact).tolist(),
                "saved_control_c_h_q": controls[period, person].tolist(),
                "represented_value": float(decomposition["represented_values"][period, person]),
                "saved_policy_objective": original_q,
                "baseline_recovered_objective": baseline[0],
                "baseline_minus_saved_objective": baseline[0] - original_q,
                "stronger_recovered_objective": strong[0],
                "stronger_control_c_h_q": strong[1:],
                "stronger_minus_saved_objective": difference,
                "observed_policy_regret_lower_bound": max(0.0, difference),
                "saved_value_defect": float(
                    decomposition["represented_values"][period, person] - original_q
                ),
                "stronger_value_defect": float(
                    decomposition["represented_values"][period, person] - strong[0]
                ),
                "discounted_saved_value_defect": float(defects[period]),
            }
            type_report["states"].append(row)
            print(
                json.dumps(
                    {
                        "person": person,
                        "period": int(period),
                        "regret_lower_bound": max(0.0, difference),
                        "saved_value_defect": row["saved_value_defect"],
                    }
                ),
                flush=True,
            )
        worst_period = periods[1]
        worst_state = states[worst_period, person]
        lower_indices = [
            int(np.clip(np.searchsorted(grid, coordinate, side="right") - 1, 0, len(grid) - 2))
            for grid, coordinate in zip(
                (solution.asset_grid, solution.log_human_capital_grid), worst_state, strict=True
            )
        ]
        corner_indices = [
            (lower_indices[0] + i, lower_indices[1] + j) for i in (0, 1) for j in (0, 1)
        ]
        corner_states = np.asarray(
            [
                (solution.asset_grid[i], solution.log_human_capital_grid[j])
                for i, j in corner_indices
            ]
        )
        corner_recovery = greedy_policy_value_at(
            stronger, int(worst_period), corner_states[:, 0], np.exp(corner_states[:, 1])
        )
        type_report["worst_defect_cell_corners"] = [
            {
                "period": int(worst_period),
                "grid_index": list(index),
                "state_A_logK": corner_states[k].tolist(),
                "stored_node_value": float(solution.values[worst_period, *index]),
                "stronger_recovered_objective": float(corner_recovery[0][k]),
                "stronger_minus_stored_node_value": float(
                    corner_recovery[0][k] - solution.values[worst_period, *index]
                ),
            }
            for k, index in enumerate(corner_indices)
        ]
        report["types"].append(type_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/solver_benchmarks"))
    parser.add_argument("--folders", nargs="+", type=Path)
    parser.add_argument(
        "--recovery-audit", type=Path, help="audit a saved cohort JSON instead of simulating"
    )
    parser.add_argument("--audit-run-index", type=int, default=-1)
    parser.add_argument("--audit-top-types", type=int, default=3)
    parser.add_argument("--audit-states-per-type", type=int, default=4)
    parser.add_argument("--audit-refinement-starts", type=int, default=4)
    parser.add_argument("--audit-refinement-steps", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--participation-threshold", type=float, default=0.02)
    args = parser.parse_args()
    if args.recovery_audit is not None:
        if args.folders:
            parser.error("--folders and --recovery-audit are separate validation modes")
        audit_saved_recovery(
            args.recovery_audit,
            args.output,
            args.platform,
            args.audit_run_index,
            args.audit_top_types,
            args.audit_states_per_type,
            args.audit_refinement_starts,
            args.audit_refinement_steps,
        )
        return
    if not args.folders:
        parser.error("provide --folders or --recovery-audit")
    if (
        not np.isfinite(args.participation_threshold)
        or not 0.0 <= args.participation_threshold < 1.0
    ):
        parser.error("participation threshold must lie in [0, 1)")
    folders = [
        path if path.is_absolute() or path.exists() else args.root / path for path in args.folders
    ]
    report = validate_folders(
        folders, args.root, args.output, args.platform, args.participation_threshold
    )
    if any(row["status"] == "evaluation_failed" for row in report["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
