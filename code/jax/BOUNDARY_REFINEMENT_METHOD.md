# Boundary feasibility and time-policy refinement benchmark

`tools/validate_boundary_refinement.py` separates two numerical changes:

1. A saved checkpoint-feasible policy at the original number of periods versus
   a continuous-feasible policy on exactly the same state, control, time, and
   domain settings.
2. The continuous-feasible policy versus a finer time grid, keeping the state
   grid, control search, domain, and initial population quadrature fixed.

The baseline configuration is explicitly assigned `asset_feasibility="checkpoints"`.
A missing field in an old report means checkpoint feasibility; it never inherits
an updated default silently. Candidate reports must explicitly match the expected
continuous configuration. Baseline value/policy arrays are reused, and their
policies are recovered using the current code's explicit legacy method. Source
and artifact checksums identify this replay. The script does not call a replay
a historical timed solve.

The stress population deliberately retains the old synthetic law: 90% correlated
continuous mass on assets `[2,8]` and log capital `[-0.25,0.25]`, 5% on the numerical
asset floor, and 5% at `(A,K)=(5,1)`. These atoms are passed explicitly so a change
to the default initial law cannot change this experiment. Initial quadrature is
identical across all policy runs. This is neither empirical calibration nor a
forward remapping benchmark; the population follows deterministic quadrature paths.

## Independent continuous-path audit

For fixed consumption `c`, active time `h`, and training time `q`, write earnings
at the period's start as `Y=h*g(q/h)*K`, capital growth as `g=a*q-delta`, and the
interest rate as `r`. The exact asset path is

```text
A(s) = exp(r*s)*A0 + Y*exp(r*s)*s*exprel((g-r)*s) - c*s*exprel(r*s).
```

The derivative after dividing out its positive exponential is

```text
D(s) = A'(s)*exp(-r*s)
     = r*A0 + Y - c + Y*g*s*exprel((g-r)*s).
D'(s) = Y*g*exp((g-r)*s).
```

`D` is monotone, so the minimum occurs at an endpoint or at its unique crossing
from negative to positive. The NumPy audit computes that crossing analytically
with `log1p` and its removable zero limit. It independently handles `r=0`, `g=0`,
`r=g`, retirement, and boundary minima. It never calls the production consumption
capacity or minimum-asset helper. Tests compare it with adaptive DOP853 integration
and scalar minimization, and include a path that passes every checkpoint but
violates the asset floor between the initial state and the first checkpoint.

All native population intervals and all saved node policies are audited. Reports
distinguish the economic floor from the positive numerical asset floor, give the
worst physical age and control, and independently reproduce endpoint assets.
The fraction of intervals flagged is not the fraction of time actually spent
below the floor. Checkpoint utility can exceed continuous-feasible utility because
its controls violate the continuously enforced constraint; that is not a welfare
improvement.

## Common physical ages and loss

State comparisons use analytic within-period transitions at common physical-age
boundaries. Control comparisons use the native step controls at common interval
midpoints; earnings use exact human capital at those midpoints. For 140 versus
280 periods, the default common grid has 280 intervals. This avoids comparing
array indices or artificially interpolating a discontinuous control.

Floor mass is compared only at shared native state boundaries. An atom's
probability is never linearly interpolated between ages. Native floor-mass
profiles remain saved in `paths.npz`.

Retirement smoothness uses the exact constant-control equal-step Euler condition,
`dlog(c)/dt=(r-rho)/(1-consumption_power)`. Both adjacent periods must have
`h<1e-6`, and all three enclosing asset nodes must clear the numerical floor by
`1e-6`. Residuals are per model year. A run without eligible retired pairs reports
null, not a fictitious zero error.

The common-age loss uses consumption, hours, training, participation, and earnings
at seven fixed common midpoint ages. Targets come from the finest continuous run.
Residual scales are 0.02 for time fractions/participation and 0.1 for consumption
and earnings; all observations have equal weight. It uses the calibration loss
API on the explicitly sampled common-age profiles. Its zero at the reference is
mechanical, and the loss is a numerical comparison rather than empirical fit.
The target ages are physical ages shared by all runs, including when period
counts differ.

## Commands and artifacts

After the continuous feasibility implementation and tests pass, run:

```bash
PYTHONPATH=code/jax:tools python tools/validate_boundary_refinement.py \
  --baseline output/solver_benchmarks/bicubic_asset121_fine \
  --periods 140 280 --quadrature 64 --platform gpu --warm-solves 1 \
  --node-diagnostics --output output/solver_benchmarks/boundary_time_refinement
```

To reuse already solved continuous candidates, add:

```text
--candidate-folders PATH_TO_CONTINUOUS_140 PATH_TO_CONTINUOUS_280
```

`--common-periods` overrides the shared physical sampling mesh. The default is the
least common multiple of native period counts. `--feasibility-tolerance` defaults
to `1e-10`; changing it is recorded. `--warm-solves` controls repeated policy solves;
first and repeated timings are separate. Population timings explicitly include
first compilation. Optional `--node-diagnostics` adds the existing complete
Bellman identity and feasibility diagnostics to the independent path audit.

`report.json` records configurations, provenance, node/path audits, utility,
smoothness, loss, and separate method/time comparisons. `paths.npz` contains
initial states/weights/components, native paths and controls, analytic minima and
minimizer times, native floor masses, and common-age profiles. Newly solved
policies are saved under `continuous_PERIODS/`. JSON and NPZ writes use atomic
renames. The benchmark never sets a convergence flag: additional state, control,
quadrature, numerical-floor, and domain refinements remain separate requirements.

## Joint time and asset refinement with verified path reuse

The benchmark also accepts `--asset-nodes`, one count per continuous candidate.
Time and asset counts must be nondecreasing. The first continuous candidate must
match both baseline grids, so it isolates the feasibility change. Later candidates
can separately increase time, assets, or both; the report names the comparison
accordingly and lists every changed configuration field.

A `-` in `--candidate-folders` requests a new solve at that position. For example,
after the time-only comparison above:

```bash
PYTHONPATH=code/jax:tools python tools/validate_boundary_refinement.py \
  --baseline output/solver_benchmarks/bicubic_asset121_fine \
  --periods 140 280 280 --asset-nodes 121 121 241 \
  --candidate-folders \
    output/solver_benchmarks/boundary_time_refinement/continuous_140 \
    output/solver_benchmarks/boundary_time_refinement/continuous_280 - \
  --reuse-populations output/solver_benchmarks/boundary_time_refinement/report.json \
  --quadrature 32 --platform gpu --warm-solves 1 --node-diagnostics \
  --output output/solver_benchmarks/boundary_state_refinement
```

The new policy is saved as `continuous_280_asset241`. Earlier population paths
are reused only when their policy-file SHA256, full configuration, model
parameters, initial law, and quadrature weights match. Stored times and initial
states are checked as well. A mismatch triggers a new population evaluation.
The output identifies each reused source explicitly; its new population timing
is null instead of being presented as a fresh timed rollout. Available node
identity diagnostics are reused with the matching policy artifact. Independent
NumPy minima and common-age summaries are still recomputed from the saved arrays.
Source report and path-file hashes are recorded.

The final candidate supplies the synthetic loss targets, so loss values from
separate benchmark reports must not be compared unless the targets coincide.
Reusing a population does not change its initial atom or floor-face weights.
