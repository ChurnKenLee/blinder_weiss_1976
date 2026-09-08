# Continuum population and calibration

The population is a probability measure over assets and log human capital,
aged forward under a common Bellman policy solution. Two deterministic
approximations are available: quadrature paths from the initial measure, and
conservative transport of probability masses on an independent state grid.
An IID cohort from the same initial law remains a sampling-error comparison.

The initial law and calibration pilot are **synthetic numerical exercises**.
They do not estimate the 1976 paper's parameters or the downloaded ACS/ATUS
microdata. The measurement choices still needed for empirical targets are in
[data/ipums/README.md](../../data/ipums/README.md).

## Initial measure

`SyntheticInitialDistribution` specifies bounded uniform marginals for assets
and log human capital, with continuous-component joint density proportional to
`1 + 3*correlation*(2*u-1)*(2*v-1)` for unit coordinates `u,v`. The supported
correlation is `Corr(A, log K)` in `[-1/3,1/3]`. Adding a floor-face component or
point atoms changes the mixture's overall correlation. The default law places
90% in assets `[2,8]` and log human capital `[-0.25,0.25]`, 5% on the numerical
asset floor with uniform log human capital, and 5% at `(A,K)=(5,1)`.

`initial_quadrature` uses a tensor Gauss–Legendre rule for the continuous
component and one-dimensional quadrature for the floor face. Point atoms are
retained explicitly. Its nodes carry nonnegative probability weights. Initial
state heterogeneity shares a policy solution; preference or technology types
require separate solutions and explicit type weights.

## Forward mass transport

Transport follows the same exact constant-control transition as the cohort
simulator. It uses a local nonnegative remapping operator, separate from cubic
value interpolation. The forward action is `p_next = P.T @ p`; its backward
expectation action is `P @ phi`. Discount factors do not enter transport.
All periods, including retirement and the terminal boundary, preserve the
cohort's mass; no mortality or cross-sectional age pooling is assumed.

The economic borrowing limit is `ModelParams.asset_floor` (zero by default).
The Bellman solver requires a strictly higher numerical floor (default
`1e-4`) because the benchmark bequest utility is singular at zero. Reported
floor mass belongs to this numerical approximation. Lowering that epsilon is
part of convergence, separate from refinement of the population grid.

The forward grid distinguishes the floor face from strictly interior asset
nodes. Interior destinations never receive floor mass merely because they lie
between the floor and the first interior node. Such destinations must instead
be represented on the interior grid, introducing a measured remapping bias
that decreases as that grid is refined. Floor mass can enter, move along the
face, and leave it when the policy points inward. Human-capital limits and the
upper asset limit are artificial cutoffs, not absorbing economic boundaries.
Material active-mass exits raise an error with diagnostics.

Constraint contact is classified identically for quadrature and transport.
Exact equality identifies initial floor atoms. Subsequent contacts require both
the endpoint asset residual and consumption-capacity slack to fit a float64
roundoff budget scaled by the asset transition's cancellation terms. A bitwise
comparison of separately compiled consumption limits was found to miss true
contacts and spuriously alternate face mass with the first interior row. The
shared classifier fixes that failure; a 140-period regression checks repeated
contact while a separate test preserves genuinely slack interior states.

This follows the deterministic distribution approach described by
[Young](https://www.wouterdenhaan.com/suite/finalversion-young.pdf).
The need to retain constraint mass separately from an interior density is also
explained in [Achdou et al.](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf).
The implementation propagates probability masses, not density values; plotting
a density requires cell volumes and a Jacobian when changing from log K to K.

## Calibration interface

`simulate_population` exposes `quadrature`, `cohort`, and `transport` through
common control and state moment profiles. Quadrature is the reference default;
transport remains opt-in until numerical tolerances have been established.
Control means and earnings refer to decision-period starts. States include
the terminal age. Participation is explicitly `h > threshold`; training is
unconditional `q`, not conditional `q/h`. Nonparticipants remain in all
population denominators.

`CalibrationTargets` supplies an age origin, source, participation convention,
and `AgeMomentTarget` profiles. Each target carries units, positive residual
scales, and nonnegative observation weights. The objective is

```text
sum(weight * ((model - target) / scale)**2) / sum(weight).
```

Age interpolation is explicit, extrapolation is rejected, and terminal-age
control targets are invalid. Observation weights are not an implicit age
distribution. `evaluate_calibration` runs the complete solve, population
simulation, and loss. `fit_scalar_calibration` provides bounded derivative-free
fitting with recorded evaluations. Discrete policy and participation switches
can make the loss irregular; autodiff gradients are not certified.

## Acceptance and remaining work

Conservation alone does not establish distribution accuracy. Compare quadrature
orders, forward grids, Bellman grids, time steps, numerical floors, and artificial
domains separately. Compare CDFs and integrated moments where atoms make density
comparisons misleading. Include perturbed structural parameters and report first
calls separately from repeated solve-and-loss timings.

The present forward method can spread concentrated interior mass. It also
remaps small positive asset destinations upward to preserve their distinction
from floor atoms. These errors must be judged against empirical uncertainty and
the loss changes the calibration optimizer needs to resolve. The full design
and acceptance criteria remain in [CONTINUUM_FORWARD_MEMO.md](../../CONTINUUM_FORWARD_MEMO.md).
