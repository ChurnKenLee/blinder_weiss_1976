"""Euler-based policy smoothness checks, without altering recovered controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import ModelParams


@dataclass(frozen=True)
class RetirementEulerDiagnostics:
    """Adjacent retired-period growth errors and the economically eligible mask.

    Arrays retain any trailing population dimensions. Summaries are per path,
    and are NaN when no eligible adjacent periods exist. Residual units are
    consumption log growth per model year. This diagnoses consumption only;
    smoothness at working or retirement regime switches is a separate check.
    """

    residuals: np.ndarray
    eligible: np.ndarray
    count: np.ndarray
    root_mean_square: np.ndarray
    maximum_absolute: np.ndarray


def retirement_euler_diagnostics(
    time: ArrayLike,
    consumption: ArrayLike,
    hours: ArrayLike,
    assets: ArrayLike,
    params: ModelParams,
    *,
    asset_minimum: float,
    retirement_hours_tolerance: float = 1e-6,
    asset_floor_tolerance: float = 1e-6,
) -> RetirementEulerDiagnostics:
    """Check the exact equal-step, constant-control retirement Euler condition.

    The interior condition is ``dlog(c)/dt = (r-rho)/(1-consumption_power)``.
    Keep pairs only when both periods are retired and all three bounding
    asset nodes are strictly above the supplied numerical floor plus tolerance.
    This excludes borrowing-constraint multipliers without confusing small
    consumption changes with a smooth policy. For retired constant controls,
    assets are monotone within each period, so checking endpoints suffices.
    Time must be uniform; population dimensions follow the leading time axis.
    """

    time = np.asarray(time, dtype=np.float64)
    consumption = np.asarray(consumption, dtype=np.float64)
    hours = np.asarray(hours, dtype=np.float64)
    assets = np.asarray(assets, dtype=np.float64)
    if time.ndim != 1 or time.size < 3 or not np.all(np.isfinite(time)):
        raise ValueError("time must contain at least three finite state boundaries")
    steps = np.diff(time)
    if np.any(steps <= 0.0) or not np.allclose(steps, steps[0], rtol=1e-12, atol=1e-14):
        raise ValueError("retirement Euler diagnostics require equally spaced increasing time")
    if (
        consumption.ndim < 1
        or consumption.shape != hours.shape
        or consumption.shape[0] != time.size - 1
        or assets.shape != (time.size, *consumption.shape[1:])
    ):
        raise ValueError("controls and assets require matching time and population dimensions")
    if (
        not all(np.all(np.isfinite(array)) for array in (consumption, hours, assets))
        or np.any(consumption <= 0.0)
        or np.any(hours < 0.0)
        or np.any(hours > 1.0)
    ):
        raise ValueError("controls and assets must be finite, with c > 0 and 0 <= h <= 1")
    settings = (asset_minimum, retirement_hours_tolerance, asset_floor_tolerance)
    if not all(np.isfinite(value) for value in settings) or min(settings[1:]) < 0.0:
        raise ValueError("asset minimum and nonnegative tolerances must be finite")
    if params.consumption_power >= 1.0 or not np.isfinite(params.consumption_power):
        raise ValueError("consumption_power must be finite and below one")
    if not np.isfinite(params.interest_rate) or not np.isfinite(params.rho):
        raise ValueError("interest and discount rates must be finite")
    growth = (params.interest_rate - params.rho) / (1.0 - params.consumption_power)
    residuals = np.diff(np.log(consumption), axis=0) / steps[0] - growth
    interior = assets > asset_minimum + asset_floor_tolerance
    retired = hours < retirement_hours_tolerance
    eligible = retired[:-1] & retired[1:] & interior[:-2] & interior[1:-1] & interior[2:]
    count = np.sum(eligible, axis=0)
    square_sum = np.sum(np.where(eligible, residuals**2, 0.0), axis=0)
    root_mean_square = np.sqrt(
        np.divide(square_sum, count, out=np.full_like(square_sum, np.nan), where=count > 0)
    )
    maximum_absolute = np.where(
        count > 0, np.max(np.where(eligible, np.abs(residuals), 0.0), axis=0), np.nan
    )
    return RetirementEulerDiagnostics(
        residuals=residuals,
        eligible=eligible,
        count=count,
        root_mean_square=root_mean_square,
        maximum_absolute=maximum_absolute,
    )
