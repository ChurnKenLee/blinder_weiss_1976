"""Direct transcription of the continuous-time lifecycle problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from .model import (
    ModelParams,
    batched_dynamics,
    bequest_utility,
    earnings_tradeoff,
    flow_utility,
)

CollocationScheme = Literal["trapezoid", "hermite-simpson"]


@dataclass(frozen=True)
class DecisionLayout:
    """Locations of state and control paths in a flat NLP vector."""

    intervals: int

    @property
    def nodes(self) -> int:
        return self.intervals + 1

    @property
    def state_size(self) -> int:
        return 2 * self.nodes

    @property
    def control_size(self) -> int:
        return 3 * self.nodes

    @property
    def size(self) -> int:
        return self.state_size + self.control_size

    def unpack(self, decision: Array) -> tuple[Array, Array]:
        states = decision[: self.state_size].reshape((self.nodes, 2))
        controls = decision[self.state_size :].reshape((self.nodes, 3))
        return states, controls

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        if states.shape != (self.nodes, 2):
            raise ValueError(f"states must have shape {(self.nodes, 2)}, got {states.shape}")
        if controls.shape != (self.nodes, 3):
            raise ValueError(f"controls must have shape {(self.nodes, 3)}, got {controls.shape}")
        return np.concatenate((states.ravel(), controls.ravel())).astype(np.float64)


def time_grid(params: ModelParams, intervals: int) -> Array:
    """Equally spaced collocation nodes on ``[0, T]``."""

    return jnp.linspace(0.0, params.horizon, intervals + 1)


def midpoint_values(
    states: Array, controls: Array, params: ModelParams, step: float
) -> tuple[Array, Array]:
    """Hermite--Simpson midpoint states and linearly interpolated controls."""

    dynamics_at_nodes = batched_dynamics(states, controls, params)
    midpoint_states = 0.5 * (states[:-1] + states[1:]) + step / 8.0 * (
        dynamics_at_nodes[:-1] - dynamics_at_nodes[1:]
    )
    midpoint_controls = 0.5 * (controls[:-1] + controls[1:])
    return midpoint_states, midpoint_controls


def collocation_defects(
    decision: Array,
    params: ModelParams,
    intervals: int,
    scheme: CollocationScheme,
) -> Array:
    """Unscaled local defects in assets and log human capital."""

    layout = DecisionLayout(intervals)
    states, controls = layout.unpack(decision)
    step = params.horizon / intervals
    dynamics_at_nodes = batched_dynamics(states, controls, params)

    if scheme == "trapezoid":
        integrated_dynamics = 0.5 * step * (dynamics_at_nodes[:-1] + dynamics_at_nodes[1:])
    elif scheme == "hermite-simpson":
        midpoint_states, midpoint_controls = midpoint_values(states, controls, params, step)
        midpoint_dynamics = batched_dynamics(midpoint_states, midpoint_controls, params)
        integrated_dynamics = (
            step / 6.0 * (dynamics_at_nodes[:-1] + 4.0 * midpoint_dynamics + dynamics_at_nodes[1:])
        )
    else:
        raise ValueError(f"Unknown collocation scheme: {scheme}")

    return states[1:] - states[:-1] - integrated_dynamics


def equality_constraints(
    decision: Array,
    params: ModelParams,
    intervals: int,
    scheme: CollocationScheme,
) -> Array:
    """Scaled initial-condition and local-dynamics equality constraints."""

    layout = DecisionLayout(intervals)
    states, _ = layout.unpack(decision)
    initial_target = jnp.array([params.initial_assets, jnp.log(params.initial_human_capital)])
    state_scale = jnp.array(
        [max(1.0, abs(params.initial_assets), params.initial_human_capital), 1.0]
    )
    initial_residual = (states[0] - initial_target) / state_scale
    defects = collocation_defects(decision, params, intervals, scheme) / state_scale
    return jnp.concatenate((initial_residual, defects.ravel()))


def training_time_slack(decision: Array, intervals: int) -> Array:
    """Nonnegative slack for the economically exact constraint ``q <= h``."""

    layout = DecisionLayout(intervals)
    _, controls = layout.unpack(decision)
    return controls[:, 1] - controls[:, 2]


def asset_path_slack(
    decision: Array,
    params: ModelParams,
    intervals: int,
    scheme: CollocationScheme,
) -> Array:
    """Assets above the borrowing limit at every interval check point.

    Node assets are handled by box bounds. Hermite--Simpson can nevertheless
    produce a negative cubic state between two nonnegative nodes, so its
    collocation midpoint is constrained explicitly. For trapezoid, the linear
    midpoint is returned for a uniform optimizer interface.
    """

    layout = DecisionLayout(intervals)
    states, controls = layout.unpack(decision)
    if scheme == "hermite-simpson":
        step = params.horizon / intervals
        path_states, _ = midpoint_values(states, controls, params, step)
    elif scheme == "trapezoid":
        path_states = 0.5 * (states[:-1] + states[1:])
    else:
        raise ValueError(f"Unknown collocation scheme: {scheme}")
    return path_states[:, 0] - params.asset_floor


def lifetime_utility(
    decision: Array,
    params: ModelParams,
    intervals: int,
    scheme: CollocationScheme,
) -> Array:
    """Discounted flow utility plus discounted terminal bequest."""

    layout = DecisionLayout(intervals)
    states, controls = layout.unpack(decision)
    times = time_grid(params, intervals)
    step = params.horizon / intervals

    consumption = controls[:, 0]
    hours = controls[:, 1]
    discounted_flow = jnp.exp(-params.rho * times) * flow_utility(consumption, hours, params)

    if scheme == "trapezoid":
        flow_value = step * (
            0.5 * discounted_flow[0] + jnp.sum(discounted_flow[1:-1]) + 0.5 * discounted_flow[-1]
        )
    elif scheme == "hermite-simpson":
        midpoint_states, midpoint_controls = midpoint_values(states, controls, params, step)
        del midpoint_states  # Flow utility does not depend directly on either state.
        midpoint_times = 0.5 * (times[:-1] + times[1:])
        midpoint_flow = jnp.exp(-params.rho * midpoint_times) * flow_utility(
            midpoint_controls[:, 0], midpoint_controls[:, 1], params
        )
        flow_value = (
            step / 6.0 * jnp.sum(discounted_flow[:-1] + 4.0 * midpoint_flow + discounted_flow[1:])
        )
    else:
        raise ValueError(f"Unknown collocation scheme: {scheme}")

    discounted_bequest = jnp.exp(-params.rho * params.horizon) * bequest_utility(
        states[-1, 0], params
    )
    return flow_value + discounted_bequest


def scaled_negative_objective(
    decision: Array,
    params: ModelParams,
    intervals: int,
    scheme: CollocationScheme,
) -> Array:
    """Minimization objective scaled to roughly one period of utility."""

    return -lifetime_utility(decision, params, intervals, scheme) / params.horizon


def make_initial_guess(
    params: ModelParams,
    intervals: int,
    *,
    hours: float = 0.35,
    investment_share: float = 0.10,
) -> np.ndarray:
    """Construct a positive, exactly trapezoid-feasible constant-asset path."""

    if not 0.0 <= hours < 1.0:
        raise ValueError("hours must lie in [0, 1)")
    if not 0.0 <= investment_share <= 1.0:
        raise ValueError("investment_share must lie in [0, 1]")

    layout = DecisionLayout(intervals)
    times = np.linspace(0.0, params.horizon, layout.nodes)
    log_initial_human_capital = np.log(params.initial_human_capital)
    log_growth = (
        params.human_capital_productivity * investment_share * hours
        - params.human_capital_depreciation
    )
    log_human_capital = log_initial_human_capital + log_growth * times
    human_capital = np.exp(log_human_capital)
    assets = np.full(layout.nodes, params.initial_assets)
    tradeoff = float(earnings_tradeoff(jnp.asarray(investment_share)))
    consumption = params.interest_rate * assets + hours * tradeoff * human_capital

    states = np.column_stack((assets, log_human_capital))
    training_time = hours * investment_share
    controls = np.column_stack(
        (
            consumption,
            np.full(layout.nodes, hours),
            np.full(layout.nodes, training_time),
        )
    )
    return layout.pack(states, controls)


def decision_bounds(
    params: ModelParams,
    intervals: int,
    *,
    consumption_floor: float,
    leisure_floor: float,
    terminal_asset_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit state and control bounds, preserving economically relevant corners."""

    layout = DecisionLayout(intervals)
    state_lower = np.tile([params.asset_floor, -80.0], (layout.nodes, 1))
    state_upper = np.tile([np.inf, 80.0], (layout.nodes, 1))
    state_lower[-1, 0] = max(params.asset_floor, terminal_asset_floor)

    control_lower = np.tile([consumption_floor, 0.0, 0.0], (layout.nodes, 1))
    control_upper = np.tile([np.inf, 1.0 - leisure_floor, 1.0], (layout.nodes, 1))
    lower = layout.pack(state_lower, control_lower)
    upper = layout.pack(state_upper, control_upper)
    return lower, upper


def interpolate_decision(
    decision: np.ndarray, old_intervals: int, new_intervals: int, horizon: float
) -> np.ndarray:
    """Interpolate a converged time path to warm-start a finer mesh."""

    old_layout = DecisionLayout(old_intervals)
    new_layout = DecisionLayout(new_intervals)
    old_states, old_controls = old_layout.unpack(jnp.asarray(decision))
    old_states_np = np.asarray(old_states)
    old_controls_np = np.asarray(old_controls)
    old_times = np.linspace(0.0, horizon, old_layout.nodes)
    new_times = np.linspace(0.0, horizon, new_layout.nodes)
    new_states = np.column_stack(
        [np.interp(new_times, old_times, old_states_np[:, column]) for column in range(2)]
    )
    new_controls = np.column_stack(
        [np.interp(new_times, old_times, old_controls_np[:, column]) for column in range(3)]
    )
    return new_layout.pack(new_states, new_controls)
