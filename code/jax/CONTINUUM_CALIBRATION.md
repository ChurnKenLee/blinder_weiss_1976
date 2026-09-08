# Continuum population and calibration

The population is a probability measure over assets and log human capital,
aged forward under a common Bellman policy solution. Two deterministic
approximations are available: quadrature paths from the initial measure, and
conservative transport of probability masses on an independent state grid.
An IID cohort from the same initial law remains a sampling-error comparison.

The initial law and calibration pilot are **synthetic numerical exercises**.
They do not estimate the 1976 paper's parameters or the downloaded ACS/ATUS
microdata. Reproducible descriptive ACS/ATUS age profiles are now saved in
[empirical_age_profiles_2024.json](../../output/calibration/empirical_age_profiles_2024.json).
They retain survey units and do not silently become model calibration targets.
Time-endowment normalization, training coverage, money units, age aggregation,
and survey variance remain unresolved measurement inputs; neither source
identifies initial assets or latent human capital on its own. See
[data/ipums/README.md](../../data/ipums/README.md).

## Initial measure

`SyntheticInitialDistribution` specifies bounded uniform marginals for assets
and log human capital, with continuous-component joint density proportional to
`1 + 3*correlation*(2*u-1)*(2*v-1)` for unit coordinates `u,v`. The supported
correlation is `Corr(A, log K)` in `[-1/3,1/3]`. Adding a floor-face component or
point atoms changes the mixture's overall correlation. The default law places
95% in assets `[2,8]` and log human capital `[-0.25,0.25]`, and 5% on the
numerical asset floor with uniform log human capital. There is no default
interior point atom. `synthetic_initial_scenarios()["point_atom_stress"]`
explicitly moves 5% from the continuous component to `(A,K)=(5,1)`; this is a
stress scenario, not an empirically estimated concentration. Other named
scenarios vary the floor share or initial correlation. Nodes retain the exact
initial-law specification and quadrature order for comparison provenance.

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
common control and state moment profiles. An optional positive
`near_asset_floor_width` measures the probability of assets in the inclusive
band `[numerical_floor, numerical_floor + width]`, at every age boundary. This
includes the exact floor mass and strictly interior near-floor mass. Keep the
width fixed in model asset units across numerical grids and time steps;
`CalibrationTargets.near_asset_floor_width` must match it exactly. Exact
`asset_floor_mass` remains a separately reported endpoint-contact statistic.
An interior contact during a period need not leave mass on the floor at its
endpoint. Quadrature is the reference default;
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
fitting with recorded evaluations. The supplied initial parameter is evaluated
first when it lies within the bounds, and the returned `x`/`fun` retain the best
actually evaluated candidate. `selection_source` records whether this was the
initial parameter or the bounded search; `raw_optimizer` preserves the search
candidate and its termination. The evaluation cap includes the initial
candidate. Successful termination does not establish global optimality.
Discrete policy and participation switches can make the loss irregular;
autodiff gradients are not certified.

`evaluate_initial_law` and `fit_initial_law_calibration` reuse a fixed structural
policy solution to vary initial heterogeneity. Supported scalar inputs include
asset/log-capital support endpoints, continuous-component correlation,
`asset_floor_mass`, and `atom_mass` at an explicitly supplied atom index/location.
Bounds must preserve a valid probability law and remain inside the Bellman
domain. Point atoms are never inserted implicitly by a fit. These inputs can be
estimated only with identifying measurements; structural parameters and the
initial law may otherwise offset each other. The synthetic sensitivity output
is a numerical exercise, not a claim of joint empirical identification.

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

## Calibration stability across resolutions

`tools/benchmark_calibration_stability.py` constructs one synthetic target from
the finest requested time solution and quadrature order. All fits use that same
target at common ages; targets are not regenerated to fit each discretization.
The target includes a fixed-width near-floor band, while exact endpoint floor
mass is saved for diagnosis. The tool also reports initial-law stress scenarios
and optionally fits the floor share while reusing the structural policy.

`compare_calibration_resolutions` requires the same structural parameters and
explicit initial law, genuinely different numerical settings, successful
optimizer termination, full-period asset-floor feasibility, and agreement of
moments, loss, and fitted parameters within explicit tolerances. Device,
platform, batch size, and inactive continuous-mode checkpoint settings do not
count as distinct resolutions. Moment differences are divided by target
residual scales, and only positive-weight target observations enter the test.
Default illustrative thresholds in the tool are 0.1 residual-scale units,
0.001 loss units, 0.002 parameter units, and 1e-10 model assets for path
feasibility. These are numerical choices, not survey standard errors.

```bash
PYTHONPATH=code/jax JAX_PLATFORMS=cpu python tools/benchmark_calibration_stability.py \
  --periods 4 8 --orders 4 8 16 --fit-initial-law \
  --output output/solver_benchmarks/calibration_stability_cpu
```

The [small CPU pilot](../../output/solver_benchmarks/calibration_stability_cpu/report.json)
fails the comparisons across its 4/8-period and 4/8/16-order resolutions.
At 4 periods the fitted leisure weights range from 0.9074 to 0.9373; at 8 periods
they range from 0.9361 to the retained initial value 1.0. Thus refinement changes
the fitted parameter substantially even though every bounded search terminates.
 This is expected evidence that a coarse model
cannot be accepted merely because a same-resolution synthetic fit looks good.
The pilot also exposed an optimizer limitation: the bounded search could return
a positive loss despite the already supplied parameter having zero loss. The
fit now preserves that incumbent and reports the raw search separately; keeping
a known parameter is not evidence of recovery. Initial floor-share sensitivity
uses the same fixed policy and recovers its known synthetic share, which only
checks that numerical path. Further time, state-grid, quadrature, floor, and
domain refinement remain necessary before empirical calibration.

For production comparisons, `--config-report` imports a numerical configuration
and `--periods` varies its time grid. `--resolution-config-reports PATH PATH`
instead accepts a separate configuration for each distinct period count, so
joint time/state-grid refinement can be measured. Each effective configuration
is recorded, and the initial floor mixture requires a common numerical floor.
Both modes re-solve the current source; saved policy tables are not assumed to
match that source. Use at least two quadrature orders at each resolution.

The joint-refinement production command is:

```bash
PYTHONPATH=code/jax python tools/benchmark_calibration_stability.py \
  --platform gpu \
  --resolution-config-reports \
    output/solver_benchmarks/boundary_time_refinement/continuous_140/report.json \
    output/solver_benchmarks/boundary_state_refinement/continuous_280_asset241/report.json \
  --orders 32 64 --fit-initial-law --fit-evaluations 30 \
  --output output/solver_benchmarks/calibration_stability_gpu
```

The two configurations use 140 periods/121 asset nodes and 280 periods/241 asset
nodes, with 79 log-capital nodes and continuous asset feasibility. Per-candidate
progress checkpoints retain parameter values, losses and complete evaluation
timings; the final report separately retains raw bounded-search termination
and the selected incumbent/search candidate. Outputs under `resolutions` have
an explicit `running` or `completed` status.

## Historical checkpoint GPU pilot (2026-09-08)

The following saved results use the historical checkpoint feasibility rule and
an initial law with 90% continuous mass, 5% floor mass, and the explicit 5%
interior point atom. They do not describe the current continuous-feasibility,
95%/5% default. Retain the original law and `asset_feasibility="checkpoints"`
when reproducing them. The current continuous rule is described in
[CONTINUOUS_ASSET_FEASIBILITY.md](CONTINUOUS_ASSET_FEASIBILITY.md).

The [completed comparison](../../output/solver_benchmarks/continuum_gpu/report.json)
uses the refined 121×79 Bellman grid with 140 decision periods. All three
transport resolutions remain nonnegative, with mass drift at most 4.45e-16 and
row-sum error at most 2.23e-16. The following differences use quadrature order
64 as the reference; that reference is not a continuum convergence certificate.

| Population approximation | Warm population seconds | Synthetic loss | Max participation difference | Max floor-mass difference |
|---|---:|---:|---:|---:|
| IID cohort, 256 people | 3.38 | 0.09862 | 0.03921 | 0.02228 |
| Quadrature 8×8 | 3.17 | 0.34695 | 0.04713 | 0.02614 |
| Quadrature 16×16 | 3.29 | 0.01667 | 0.02586 | 0.00953 |
| Quadrature 32×32 | 3.66 | 0.000886 | 0.005326 | 0.005537 |
| Quadrature 64×64 | 5.06 | reference | — | — |
| Transport 31×31 | 3.85 | 7.75765 | 0.25182 | 0.06916 |
| Transport 61×61 | 4.92 | 2.75991 | 0.17371 | 0.29147 |
| Transport 121×121 | 10.50 | 0.46411 | 0.11005 | 0.15934 |

Participation and floor-mass differences are fractions: 0.11 means 11 percentage
points. Quadrature 32→64 changes maximum hours by 0.001093, consumption by
0.000901, earnings by 0.003281, and mean assets by 0.02211. Transport refinement
reduces the aggregate loss but does not uniformly improve floor mass. Neither
conservation nor small row-sum error is sufficient for calibration accuracy.

First Bellman solve wall time is 20.20 s, with a separately observed 7.56 s
backend-compile event; repeated solves take 9.38 s. Complete warm solve-and-loss
calls take 14.47–14.48 s for quadrature64, versus 19.88 s for transport121 and
12.76–12.79 s for the 256-person cohort. Parameter perturbations of ±2% in
leisure weight reuse compilation at fixed shape. The historical one-parameter
quadrature fit terminated successfully after 12 evaluations at leisure weight
**1.000113** (known value 1.0, absolute error 0.000113, standardized loss 4.08e-6).
That recorded result predates the incumbent safeguard: the earlier bounded
search did not first evaluate the supplied parameter. It is a historical
synthetic fitting result, not an empirical parameter estimate.

The current fitter evaluates the supplied parameter first and retains the best
actually evaluated candidate. With targets generated from the same policy
configuration, initial law and reference quadrature, the supplied value 1.0
already matches the target at zero loss up to roundoff. A current replay
therefore retains that value rather than reproducing the historical 1.000113
candidate. Retaining a supplied known value does not demonstrate parameter
recovery; the separate resolution-stability exercise tests numerical changes
against one fixed target.

## Boundary accuracy in the lifecycle benchmark

The rounding fix protects a separate reproducible arithmetic failure. It did
not materially change this lifecycle benchmark's floor-mass profiles (maximum
change below 4.2e-17); attributing the benchmark's large discrepancy to rounding
was incorrect. The measured mechanism is the first-cell remapping below.
See the [policy probe](../../output/solver_benchmarks/continuum_gpu/boundary_probe.md).

At the initial
state `(A,log K)=(0.0001,0)`, the recovered consumption binds the first
within-period checkpoint. Later earnings growth leaves endpoint assets about
0.0015003 above the floor. This is a real positive destination, roughly twelve
orders of magnitude larger than the contact error budget. The 121-node
transport grid's first interior gap is about 0.0024305; projecting the true
destination upward changes the following policy, which then reaches the floor.
The 61-node grid has a related cycle. The quadrature paths retain the actual
positive destination.

This is a measured interaction between the period's constant controls and the
first-interior-cell remapping. Increasing the boundary tolerance would create
incorrect floor mass. Further distribution-grid design and separate time-step
refinement are needed; both the interior remapping bias and outer-cell exposure
remain visible in diagnostics. Floor mass is measured at age boundaries, and
an interior checkpoint contact does not by itself create an endpoint atom.

A [supplemental curvature-3 grid](../../output/solver_benchmarks/continuum_gpu_curvature3/report.json)
keeps 121×121 nodes but lowers the first interior gap to 2.03e-5. It reduces
maximum floor-mass error from 0.15934 to 0.08581, yet worsens the aggregate
synthetic loss from 0.46411 to 0.52200; warm transport remains 10.50 s. It is
not promoted. This is evidence that resolving the first cell alone does not
settle distribution accuracy. The CLI's independent
`--distribution-asset-curvature` option permits this comparison without changing
the Bellman grid; the recorded main benchmark uses curvature 2.

## Reproduce the comparison

The benchmark re-solves the supplied numerical configuration and uses the same
initial law for the IID cohort, every quadrature order, and every transport
grid. It records JAX compilation events separately from synchronized first and
repeated calls; event durations can overlap and are not subtracted from wall
time. Complete parameter evaluations include the policy solve, policy recovery,
mass propagation, moment reduction, and loss.

```bash
PYTHONPATH=code/jax python tools/benchmark_continuum.py \
  --platform gpu \
  --config-report output/solver_benchmarks/bicubic_asset121_fine/report.json \
  --asset-feasibility checkpoints \
  --initial-law-report output/solver_benchmarks/continuum_gpu/report.json \
  --quadrature 8 16 32 64 --distribution 31 61 121 \
  --fit-evaluations 30 --output output/solver_benchmarks/continuum_gpu_replay
```

This replays the historical numerical configuration and initial law using the
current source. The separate output directory preserves the recorded historical
artifacts. The current incumbent-preserving fitting trace and evaluation count
will differ from the historical result above; this command does not reconstruct
the old optimizer implementation.

A small CPU smoke run uses `--platform cpu` and omits `--config-report`. Saved
`report.json` contains settings, source checksums, diagnostic profiles, errors,
timings, structural perturbations, and the scalar fitting trace. `profiles.npz`
contains moments, terminal CDFs, and selected distribution snapshots. Outputs
are written by atomic rename and covered by project GitHub autosave.

The synthetic loss matches consumption, hours, training, earnings and
participation at seven ages. Its scales (0.02 for time fractions/participation,
0.1 for consumption/earnings) are chosen numerical tolerances, not estimated
survey standard errors. The fit varies leisure weight on `[0.9,1.1]`, with known
synthetic value 1.0 and parameter tolerance 0.001. This exercise tests the fitting
pipeline within the same numerical model; its retained initial value is not a
recovery test. Identification, empirical mapping and numerical bias under joint
refinement remain separate questions.

For custom targets, the minimal sequence is:

```python
from blinder_weiss import (
    initial_quadrature, simulate_population, weighted_age_moment_loss,
    evaluate_calibration, fit_scalar_calibration,
)

nodes = initial_quadrature(nodes_per_dimension=64,
                           asset_floor=solution.config.asset_minimum)
population = simulate_population(solution, nodes, backend="quadrature",
                                 participation_hours_threshold=0.02)
# targets is a CalibrationTargets object with explicit units/scales/ages/source.
loss = weighted_age_moment_loss(population, targets)
evaluation = evaluate_calibration(solution.params, solution.config, nodes, targets)
fit = fit_scalar_calibration(
    "leisure_weight", (0.9, 1.1), params=solution.params,
    config=solution.config, initial_nodes=nodes, targets=targets,
)
```
