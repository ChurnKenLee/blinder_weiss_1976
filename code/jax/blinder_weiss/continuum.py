"""Deterministic approximation of an explicitly assumed initial probability law.

The default law is synthetic: a bounded continuous joint law in assets and
log human capital, a face component at the numerical asset floor, and optional
point atoms. It is not an empirical wealth/skill distribution. Preference and
technology heterogeneity require separate policy solutions and explicit type
weights; the nodes here only vary initial states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import lru_cache
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .bellman import BellmanConfig, BellmanSolution, _exprel, endpoint_consumption_capacity
from .boundary import _FLOOR_CONTACT_ULPS, endpoint_floor_contact
from .population import CohortMoments, cohort_moments, simulate_cohort


@dataclass(frozen=True)
class InitialAtom:
    """An initial point mass; capital is K, not log K."""

    assets: float
    human_capital: float
    mass: float


@dataclass(frozen=True)
class SyntheticInitialDistribution:
    r"""A bounded mixture with uniform continuous marginals in (A, log K).

    Within the continuous component, U,V in [0,1] have joint density
    ``1 + 3*correlation*(2*U-1)*(2*V-1)``. Thus Corr(A,log K) is exactly
    ``correlation`` in [-1/3,1/3]; this is not Corr(A,K), and adding face or
    point mass generally changes the overall correlation. Affine transforms
    map U,V to the specified intervals. The face component has uniform log K
    on the same interval. ``asset_floor_mass`` is placed at the supplied
    numerical floor, which only approximates the economic borrowing limit.
    All masses are probabilities, with the residual assigned to the interior.
    """

    asset_lower: float = 2.0
    asset_upper: float = 8.0
    log_human_capital_lower: float = -0.25
    log_human_capital_upper: float = 0.25
    correlation: float = 0.25
    asset_floor_mass: float = 0.05
    atoms: tuple[InitialAtom, ...] = ()


@dataclass(frozen=True)
class PopulationNodes:
    """Initial probability masses, including explicit face/point components."""

    assets: np.ndarray
    human_capital: np.ndarray
    weights: np.ndarray
    component: np.ndarray
    description: str
    initial_law: SyntheticInitialDistribution | None = None
    quadrature_order: int | None = None


@dataclass(frozen=True)
class PopulationStateMoments:
    """State means and numerical-floor mass at *all* age boundaries."""

    time: np.ndarray
    assets: np.ndarray
    human_capital: np.ndarray
    log_human_capital: np.ndarray
    asset_floor_mass: np.ndarray
    near_asset_floor_mass: np.ndarray | None = None
    near_asset_floor_width: float | None = None


@dataclass(frozen=True)
class PopulationResult:
    """Common model-moment interface for cohort, quadrature and transport."""

    backend: str
    moments: CohortMoments
    state_moments: PopulationStateMoments
    diagnostics: dict[str, Any]
    simulation: Any


def _validate_initial_law(law: SyntheticInitialDistribution, asset_floor: float) -> float:
    scalars = (
        law.asset_lower,
        law.asset_upper,
        law.log_human_capital_lower,
        law.log_human_capital_upper,
        law.correlation,
        law.asset_floor_mass,
        asset_floor,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("initial distribution parameters must be finite")
    if not asset_floor <= law.asset_lower < law.asset_upper:
        raise ValueError("asset interval must be ordered and above the supplied asset floor")
    if not law.log_human_capital_lower < law.log_human_capital_upper:
        raise ValueError("log human capital interval must be strictly ordered")
    with np.errstate(over="ignore", under="ignore"):
        capital_bounds = np.exp([law.log_human_capital_lower, law.log_human_capital_upper])
    if not np.all(np.isfinite(capital_bounds)) or np.any(capital_bounds <= 0.0):
        raise ValueError("log capital bounds must exponentiate to finite positive human capital")
    if abs(law.correlation) > 1.0 / 3.0:
        raise ValueError("continuous-component correlation must lie in [-1/3, 1/3]")
    if law.asset_floor_mass < 0.0:
        raise ValueError("asset_floor_mass must be nonnegative")
    for atom in law.atoms:
        if not np.all(np.isfinite((atom.assets, atom.human_capital, atom.mass))):
            raise ValueError("atom values must be finite")
        if atom.assets < asset_floor or atom.human_capital <= 0.0 or atom.mass < 0.0:
            raise ValueError("atoms must have feasible states and nonnegative mass")
    remainder = 1.0 - law.asset_floor_mass - sum(atom.mass for atom in law.atoms)
    if remainder < -1e-14:
        raise ValueError("initial component probabilities must sum to at most one")
    return max(0.0, remainder)


def synthetic_initial_scenarios() -> dict[str, SyntheticInitialDistribution]:
    """Explicit assumed-law sensitivity cases, including a separate point-atom stress.

    The baseline has continuous initial heterogeneity and a 5% floor face; it
    has no interior point atom. These probabilities/correlations are numerical
    scenarios, not estimates. Each case changes one initial-law feature.
    """
    baseline = SyntheticInitialDistribution()
    return {
        "baseline": baseline,
        "no_floor_atom": replace(baseline, asset_floor_mass=0.0),
        "double_floor_share": replace(baseline, asset_floor_mass=0.10),
        "independent_initial_states": replace(baseline, correlation=0.0),
        "negative_initial_correlation": replace(baseline, correlation=-0.25),
        "point_atom_stress": replace(baseline, atoms=(InitialAtom(5.0, 1.0, 0.05),)),
    }


def initial_quadrature(
    law: SyntheticInitialDistribution | None = None,
    *,
    nodes_per_dimension: int = 16,
    asset_floor: float,
) -> PopulationNodes:
    """Return n² interior nodes, n face nodes and the explicitly listed atoms.

    Gauss-Legendre integration is deterministic and has nonnegative normalized
    weights. Zero-probability components are kept to preserve array shapes when
    mixture probabilities change during calibration. There is no random seed
    and no repeated normalization during the subsequent lifecycle rollout.
    """

    law = law or SyntheticInitialDistribution()
    interior_mass = _validate_initial_law(law, asset_floor)
    if (
        isinstance(nodes_per_dimension, bool)
        or not isinstance(nodes_per_dimension, (int, np.integer))
        or nodes_per_dimension < 2
    ):
        raise ValueError("nodes_per_dimension must be an integer at least two")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(nodes_per_dimension)
    unit_nodes = (raw_nodes + 1.0) / 2.0
    unit_weights = raw_weights / 2.0
    u, v = np.meshgrid(unit_nodes, unit_nodes, indexing="ij")
    density = 1.0 + 3.0 * law.correlation * (2.0 * u - 1.0) * (2.0 * v - 1.0)
    interior_weights = interior_mass * np.outer(unit_weights, unit_weights) * density
    log_capital_span = law.log_human_capital_upper - law.log_human_capital_lower
    assets = np.concatenate(
        (
            (law.asset_lower + (law.asset_upper - law.asset_lower) * u).ravel(),
            np.full(nodes_per_dimension, asset_floor),
            np.asarray([atom.assets for atom in law.atoms]),
        )
    )
    capital = np.concatenate(
        (
            np.exp(law.log_human_capital_lower + log_capital_span * v).ravel(),
            np.exp(law.log_human_capital_lower + log_capital_span * unit_nodes),
            np.asarray([atom.human_capital for atom in law.atoms]),
        )
    )
    weights = np.concatenate(
        (
            interior_weights.ravel(),
            law.asset_floor_mass * unit_weights,
            np.asarray([atom.mass for atom in law.atoms]),
        )
    )
    # A single initial roundoff correction; never used to hide transport drift.
    weights /= weights.sum()
    component = np.concatenate(
        (
            np.full(nodes_per_dimension**2, "interior"),
            np.full(nodes_per_dimension, "asset_floor"),
            np.full(len(law.atoms), "point_atom"),
        )
    )
    return PopulationNodes(
        assets,
        capital,
        weights,
        component,
        "synthetic bounded correlated initial law; Gauss-Legendre probability quadrature",
        law,
        nodes_per_dimension,
    )


def sample_initial_population(
    law: SyntheticInitialDistribution | None = None,
    *,
    people: int = 256,
    seed: int = 125,
    asset_floor: float,
) -> PopulationNodes:
    """IID comparison cohort from exactly the same mixture as the quadrature.

    Conditional inversion samples the continuous copula. All people have equal
    weights, so sampling error includes both mixture shares and initial states.
    """

    law = law or SyntheticInitialDistribution()
    interior_mass = _validate_initial_law(law, asset_floor)
    if isinstance(people, bool) or not isinstance(people, (int, np.integer)) or people < 1:
        raise ValueError("people must be a positive integer")
    rng = np.random.default_rng(seed)
    selector, u, uniform_v = rng.uniform(size=(3, people))
    k = 3.0 * law.correlation * (2.0 * u - 1.0)
    # F(V | U=u) = (1-k)*V + k*V²; rationalization is stable near k=0.
    v = 2.0 * uniform_v / (1.0 - k + np.sqrt((1.0 - k) ** 2 + 4.0 * k * uniform_v))
    assets = law.asset_lower + (law.asset_upper - law.asset_lower) * u
    capital = np.exp(
        law.log_human_capital_lower
        + (law.log_human_capital_upper - law.log_human_capital_lower) * v
    )
    component = np.full(people, "interior", dtype="U16")
    face = (selector >= interior_mass) & (selector < interior_mass + law.asset_floor_mass)
    assets[face] = asset_floor
    capital[face] = np.exp(
        law.log_human_capital_lower
        + (law.log_human_capital_upper - law.log_human_capital_lower) * uniform_v[face]
    )
    component[face] = "asset_floor"
    lower = interior_mass + law.asset_floor_mass
    for atom in law.atoms:
        selected = (selector >= lower) & (selector < lower + atom.mass)
        assets[selected], capital[selected], component[selected] = (
            atom.assets,
            atom.human_capital,
            "point_atom",
        )
        lower += atom.mass
    return PopulationNodes(
        assets,
        capital,
        np.full(people, 1.0 / people),
        component,
        f"IID cohort from synthetic bounded correlated initial law; seed={seed}",
        law,
        None,
    )


@lru_cache(maxsize=16)
def _cached_floor_contacts(config: BellmanConfig) -> Any:
    """Identify exact initial floor mass and subsequent binding endpoint controls."""

    @jax.jit
    def contacts(params, states, controls):
        endpoint_capacity = endpoint_consumption_capacity(
            states[:-1, :, 0],
            states[:-1, :, 1],
            controls[..., 1],
            controls[..., 2],
            params,
            params.horizon / config.periods,
            config.asset_minimum,
        )
        step = params.horizon / config.periods
        consumption_factor = step * _exprel(params.interest_rate * step)
        contact = endpoint_floor_contact(
            states[1:, :, 0],
            config.asset_minimum,
            controls[..., 0],
            endpoint_capacity,
            consumption_factor,
        )
        return jnp.concatenate(((states[0, :, 0] == config.asset_minimum)[None, :], contact))

    return contacts


def simulate_population(
    solution: BellmanSolution,
    initial_nodes: PopulationNodes,
    *,
    backend: Literal["quadrature", "cohort", "transport"] = "quadrature",
    participation_hours_threshold: float,
    distribution_grid: Any = None,
    store_snapshots: bool = False,
    near_asset_floor_width: float | None = None,
) -> PopulationResult:
    """Select a population approximation while keeping moment definitions fixed.

    Quadrature is the reference default. ``cohort`` runs the same dynamics,
    normally on IID nodes. Transport remains an explicit opt-in pending joint
    distribution-grid/domain/policy/time convergence acceptance. A supplied
    ``near_asset_floor_width`` reports the inclusive band from the numerical
    floor to floor + width, in model asset units, at every age boundary. This
    band includes exact floor mass but does not change contact classification.
    Keep its width fixed when comparing numerical resolutions.
    """

    if near_asset_floor_width is not None and (
        not np.isfinite(near_asset_floor_width) or near_asset_floor_width <= 0.0
    ):
        raise ValueError("near_asset_floor_width must be finite and positive when supplied")
    if backend not in {"quadrature", "cohort", "transport"}:
        raise ValueError("backend must be quadrature, cohort or transport")
    if backend == "transport":
        from .distribution import build_transport, initialize_distribution, simulate_distribution

        if distribution_grid is None:
            raise ValueError("transport requires an explicit distribution_grid")
        initial_mass = initialize_distribution(
            distribution_grid,
            initial_nodes.assets,
            initial_nodes.human_capital,
            weights=initial_nodes.weights,
        )
        simulation = simulate_distribution(
            solution,
            distribution_grid,
            initial_mass,
            participation_hours_threshold=participation_hours_threshold,
            store_snapshots=store_snapshots,
            near_asset_floor_width=near_asset_floor_width,
        )
        native_state = simulation.state_moments
        state_moments = PopulationStateMoments(
            time=solution.time.copy(),
            assets=native_state.assets,
            human_capital=native_state.human_capital,
            log_human_capital=native_state.log_human_capital,
            asset_floor_mass=native_state.asset_floor_mass,
            near_asset_floor_mass=native_state.near_asset_floor_mass,
            near_asset_floor_width=native_state.near_asset_floor_width,
        )
        initial_stencil = build_transport(
            distribution_grid,
            np.column_stack((initial_nodes.assets, np.log(initial_nodes.human_capital))),
            on_asset_floor=initial_nodes.assets == distribution_grid.asset_floor,
        )
        diagnostics = (
            asdict(simulation.diagnostics)
            if is_dataclass(simulation.diagnostics)
            else dict(simulation.diagnostics)
        )
        diagnostics["initial_law"] = (
            asdict(initial_nodes.initial_law) if initial_nodes.initial_law is not None else None
        )
        diagnostics["quadrature_order"] = initial_nodes.quadrature_order
        initial_weights = initial_nodes.weights
        diagnostics["initial_lower_interior_remap_mass"] = float(
            initial_weights @ initial_stencil.lower_interior_remap
        )
        diagnostics["initial_asset_first_moment_remap_error"] = float(
            initial_weights @ initial_stencil.asset_remap_error
        )
        return PopulationResult(backend, simulation.moments, state_moments, diagnostics, simulation)
    cohort = simulate_cohort(
        solution, initial_nodes.assets, initial_nodes.human_capital, weights=initial_nodes.weights
    )
    moments = cohort_moments(cohort, participation_hours_threshold=participation_hours_threshold)
    floor = solution.config.asset_minimum
    floor_contacts = np.asarray(
        _cached_floor_contacts(solution.config)(solution.params, cohort.states, cohort.controls)
    )
    state_moments = PopulationStateMoments(
        time=solution.time.copy(),
        assets=cohort.assets @ cohort.weights,
        human_capital=cohort.human_capital @ cohort.weights,
        log_human_capital=cohort.log_human_capital @ cohort.weights,
        asset_floor_mass=floor_contacts @ cohort.weights,
        near_asset_floor_mass=(
            (
                floor_contacts
                | ((cohort.assets > floor) & (cohort.assets <= floor + near_asset_floor_width))
            )
            @ cohort.weights
            if near_asset_floor_width is not None
            else None
        ),
        near_asset_floor_width=near_asset_floor_width,
    )
    diagnostics = {
        "total_mass": np.full(solution.config.periods + 1, cohort.weights.sum()),
        "maximum_mass_drift": float(abs(cohort.weights.sum() - 1.0)),
        "minimum_mass": float(cohort.weights.min()),
        "minimum_consumption_capacity_slack": cohort.minimum_consumption_capacity_slack,
        "minimum_assets_during_period": cohort.minimum_assets_during_period,
        "maximum_full_period_floor_violation": max(
            0.0, floor - cohort.minimum_assets_during_period
        ),
        "initial_law": asdict(initial_nodes.initial_law)
        if initial_nodes.initial_law is not None
        else None,
        "quadrature_order": initial_nodes.quadrature_order,
        "numerical_asset_floor": floor,
        "economic_asset_floor": float(solution.params.asset_floor),
        "asset_floor_contact": "exact initial floor; shared endpoint roundoff classifier",
        "asset_floor_contact_roundoff_multiplier": _FLOOR_CONTACT_ULPS,
        "nodes": cohort.weights.size,
        "near_asset_floor_width": near_asset_floor_width,
        "near_asset_floor_definition": "[numerical floor, floor + width], including exact floor",
    }
    return PopulationResult(backend, moments, state_moments, diagnostics, cohort)
