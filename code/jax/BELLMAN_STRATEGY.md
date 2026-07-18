# Feedback policies from semi-Lagrangian Bellman recursion

## Purpose

The direct-collocation solver answers the question:

> Starting from one specified pair of initial assets and human capital, what
> state and control path maximizes lifetime utility?

The Bellman solver answers the larger question:

> At a given age, what should a person do for every economically relevant pair
> of current assets and human capital?

Its outputs are an approximation to the value function

\[
V(t,A,K)
\]

and the feedback policies

\[
\pi(t,A,K)
=\left(c^*(t,A,K),h^*(t,A,K),q^*(t,A,K)\right),
\qquad q=hx.
\]

The implementation is in
[`blinder_weiss/bellman.py`](blinder_weiss/bellman.py), and the interactive
comparison is in
[`notebooks/03_bellman_policy_functions.py`](notebooks/03_bellman_policy_functions.py).

This is an alternative numerical solution to the same deterministic economic
problem. It is also the natural foundation for adding idiosyncratic shocks.

## 1. Why a second solution method is needed

One open-loop path contains controls only at states visited along that path.
It does not say what the person should do after arriving at a different state.
That distinction is immaterial in a deterministic model when the initial state
is fixed and the planned path is followed exactly. It becomes important when:

- the calibration includes a distribution of initial assets or human capital;
- simulated people experience wage, employment, health, or return shocks;
- a counterfactual changes a state partway through life; or
- the researcher needs policy functions for forward simulation.

Solving the direct-collocation problem separately from every possible state
would implicitly define a policy, but it would repeat a large remaining-life
optimization each time. Bellman's principle of optimality avoids that
duplication by storing the value of continuation problems.

JAX automatic differentiation does not by itself create a policy function.
The policy must still be represented over a state domain. JAX makes the many
local optimization problems fast, differentiable, and suitable for GPU
batching.

## 2. Bellman equation

Write human capital as

\[
y=\log K
\]

and training time as

\[
q=hx, \qquad 0\leq q\leq h\leq1.
\]

In these variables, the continuous-time dynamics are

\[
\dot A=rA+K\,\phi(h,q)-c,
\]

\[
\dot y=aq-\delta,
\]

where

\[
\phi(h,q)=h g(q/h)
\]

is evaluated with the continuous perspective formula already used by the
direct-collocation solver.

The current-value Hamilton–Jacobi–Bellman equation is

\[
0=V_t-\rho V+
\max_{c,h,q}
\left\{
u(c,h)
+V_A\left[rA+e^y\phi(h,q)-c\right]
+V_y(aq-\delta)
\right\}.
\]

The terminal condition is

\[
V(T,A,y)=B(A).
\]

A finite-difference HJB method estimates (V_A), (V_y), and often (V_t)
from neighboring grid values. That is not what the new code does.

## 3. Semi-Lagrangian discretization

Divide the horizon into (N) periods of length

\[
\Delta t=T/N.
\]

The implemented recursion is

\[
V_n(A,y)=
\max_{c,h,q}
\left\{
D_u(\Delta t)u(c,h)
+\beta_{\Delta}V_{n+1}(A',y')
\right\},
\]

where

\[
\beta_{\Delta}=e^{-\rho\Delta t}
\]

and the exact within-period discount integral is

\[
D_u(\Delta t)
=\int_0^{\Delta t}e^{-\rho s}\,ds
=\frac{1-e^{-\rho\Delta t}}{\rho}.
\]

Controls are held constant within a period. The next state generally lies
between state nodes, so (V_{n+1}(A',y')) is evaluated by bilinear
interpolation. This is called a semi-Lagrangian method because the algorithm
follows the state transition implied by a candidate control and evaluates the
continuation value where that transition lands.

The method has several useful properties:

1. It does not numerically differentiate the value function.
2. The terminal condition is imposed directly and recursion proceeds backward.
3. The borrowing constraint enters the feasible control set rather than an
   invented derivative condition at the asset boundary.
4. Off-grid state transitions are allowed.
5. A discrete or quadrature approximation to shocks can later be inserted
   inside the continuation expectation.

It does not eliminate approximation. Time is discrete, controls are
piecewise constant, the state domain is bounded, and continuation values are
interpolated. Those approximations require convergence tests.

## 4. Exact state transition within a Bellman step

With constant (h) and (q), log human capital has constant growth

\[
\lambda=aq-\delta,
\]

so

\[
K(s)=K_0e^{\lambda s}.
\]

Assets solve

\[
\dot A(s)=rA(s)+\phi(h,q)K_0e^{\lambda s}-c.
\]

The exact endpoint is

\[
A(\Delta t)
=e^{r\Delta t}A_0
+\phi(h,q)K_0 I_K(\lambda,r,\Delta t)
-c I_c(r,\Delta t),
\]

where

\[
I_K
=e^{r\Delta t}
\frac{e^{(\lambda-r)\Delta t}-1}{\lambda-r}
\]

and

\[
I_c=\frac{e^{r\Delta t}-1}{r}.
\]

The implementation evaluates both expressions with a stable `exprel`
function when either denominator is close to zero. Therefore the state update
does not add Euler-integration error. Remaining time error comes from treating
the controls as constant within each period.

## 5. Borrowing constraint as a feasible-consumption bound

Let (\underline A_{mathrm{num}}) be the smallest asset value represented by
the Bellman state domain. For fixed ((A,y,h,q)), assets at any checkpoint
(s_j\in(0,\Delta t]) are affine in consumption:

\[
A(s_j)=A^{c=0}(s_j)-cI_c(r,s_j).
\]

Consequently, the largest consumption consistent with the asset floor at that
checkpoint is

\[
\bar c_j
=\frac{A^{c=0}(s_j)-\underline A_{mathrm{num}}}
{I_c(r,s_j)}.
\]

The code uses

\[
\bar c(A,y,h,q)=\min_j\bar c_j
\]

over configurable within-period checkpoints and parameterizes consumption as
a fraction of that capacity. Thus node controls satisfy the borrowing limit
by construction at every checkpoint. This is much more stable than guessing a
finite-difference condition for (V_A) at (A=\underline A).

The benchmark bequest utility has a negative power and is singular at zero.
For that reason, the Bellman domain starts at a small positive numerical value,
`asset_minimum`, rather than exactly zero. The default is (10^{-4}), and the
asset grid is curved to put relatively many nodes near this boundary. A
convergence exercise should lower this value and add nearby nodes.

## 6. State and control approximation

### State nodes

The stored state is ((A,y)), not ((A,K)). The default state domain is:

- 31 curved asset nodes between (10^{-4}) and 35;
- 25 uniform log-human-capital nodes between (-2) and (2.25); and
- 71 time nodes for 70 annual decision periods.

This is 55,025 stored value nodes, so memory is modest. Policy arrays are
stored for consumption, active time, and training time.

The state domain is an economic assumption as well as a numerical choice.
Candidate controls whose next state leaves the domain are rejected. The
simulator and diagnostics report whether the benchmark path remains inside
the domain, but researchers must also check the domains reached by all
calibration types and shocks.

### Global control search

At each state node, the solver first evaluates a tensor product of:

- active-time nodes (h\in[0,1));
- investment-share nodes (x\in[0,1]), with (q=hx); and
- consumption-capacity fractions.

Cosine-spaced nodes provide extra resolution near control boundaries. The
global discrete comparison is important because schooling, work, and
retirement can produce distinct local optima.

### Autodiff refinement

The best discrete candidate initializes a projected Adam refinement. JAX
differentiates each state's Bellman objective with respect to (h), (x), and
the consumption fraction. A proposed refinement is retained only when it is
feasible and raises the node value, so refinement cannot make the discrete
solution worse.

All state-node objectives are independent conditional on the next-period value
array. JAX batches them into one compiled operation. On a CUDA installation,
this is the part that runs naturally on the GPU.

### Neighboring-policy safeguard

A finite global control grid plus a short local refinement can occasionally
miss a better control at an isolated state. Such a miss previously produced a
lower value at the second asset node than at the borrowing-boundary node near
the end of life. More assets cannot reduce the feasible set in the underlying
model, so that pattern was numerical rather than economic.

After independent optimization, the solver therefore takes configurable
`neighbor_policy_sweeps`. At each richer adjacent state it evaluates the
absolute consumption, hours, and training chosen by the poorer neighbor. The
same absolute consumption is translated into the current state's feasible-
capacity parameterization. The inherited policy is accepted only if it is
representable, satisfies all state and domain constraints, and raises the
Bellman objective. Repeated asset and human-capital sweeps propagate useful
candidates without replacing values by an artificial monotone envelope.

This safeguard cannot repair an inadequate upper state domain. For example,
at `asset_maximum`, higher human capital can make otherwise desirable controls
leave the represented asset domain. A remaining monotonicity violation located
at that upper boundary is evidence for expanding the domain, not for forcing
the value array to be monotone after the fact.

### CUDA execution and the backward time loop

Installing a CUDA-enabled JAX build does not require separate economic or
numerical code. The same JAX program is lowered by XLA for either CPU or an
NVIDIA GPU. The implementation makes device use explicit in four places:

1. `BellmanConfig.compute_platform` is `"auto"`, `"gpu"`, or `"cpu"`.
   Automatic selection uses the best backend visible to JAX. Requiring
   `"gpu"` raises an immediate error when CUDA is unavailable, which prevents
   a large calibration run from silently falling back to CPU.
2. `jax.device_put` places the state grids and terminal value on the selected
   device before compilation. `device_index` chooses a particular GPU when
   several are present.
3. `jax.lax.scan` expresses all backward age steps as one compiled loop. The
   continuation value produced at age (n+1) remains on the device and becomes
   the input at age (n), without a Python dispatch or host copy between ages.
4. `jax.device_get` transfers the finished value and policy histories to NumPy
   once, after the recursion is complete. The returned arrays therefore remain
   convenient for plots, diagnostics, and microdata simulation.

The scan is sequential in age because each value function depends on the next
one. The useful GPU parallelism is across state nodes and candidate controls
within an age. CUDA therefore becomes increasingly valuable as the state
space, shock nodes, household types, or control grids grow. The solver uses
64-bit arithmetic for feasibility and value comparisons, so timings should be
measured on the target GPU rather than inferred from consumer-GPU float32
specifications.

## 7. Reading the returned objects

`solve_bellman()` returns a `BellmanSolution` containing:

- `values[n, i, j]`: (V_n(A_i,y_j));
- `consumption_policy[n, i, j]`;
- `hours_policy[n, i, j]`;
- `training_time_policy[n, i, j]`;
- state and time grids; and
- solve time, backend, and selected-device metadata.

The investment share is recovered safely as (x=q/h), with (x=0) when
(h=0).

`policy_at()` bilinearly interpolates the three stored node policies at
arbitrary in-domain states. This is inexpensive, but interpolation and
maximization do not commute: interpolating two argmax controls need not produce
the argmax at the interpolated state.

`greedy_policy_at()` instead batches arbitrary states and repeats the global
control search and autodiff refinement against the interpolated next-period
value function. It also evaluates the interpolated node policy as an incumbent,
so fresh recovery cannot lower the one-period approximate Bellman objective.
The compiled recovery kernel is cached by model configuration and runs on the
same JAX device as the solution.

`simulate_policy()` defaults to `policy_method="greedy"`. Its forward ages are
compiled into a second `jax.lax.scan`, keeping states, continuation values, and
controls on the device. `policy_method="interpolate"` retains the original
control-interpolation rollout. Both should be reported during convergence work:
a smaller Bellman consistency gap by itself does not prove that realized
lifetime utility is closer to the continuous-time optimum.

The interpolated initial value and the utility produced by interpolated
policies need not agree exactly. Linear interpolation of a concave value
function tends to lie below the value function, while interpolating controls
and then simulating them is a different approximation. The signed diagnostic

\[
V_0^{\mathrm{interp}}-J(\pi)
\]

should move toward zero for both recovery methods as state resolution
increases, but it need not be positive on a coarse grid.

## 8. Diagnostics and independent verification

`diagnose_bellman()` checks:

- finiteness of all stored values;
- the Bellman identity at stored nodes under the stored policies;
- consumption-capacity slack;
- (0\leq q\leq h<1);
- containment of every node's next state in the state domain;
- monotonicity violations in assets and human capital; and
- the domain and asset-floor behavior of the interpolated benchmark rollout.

A tiny node Bellman residual is necessary but not sufficient. The stored value
was constructed from the stored policy, so internal consistency can be high
even on a coarse grid. Approximation quality must be judged from:

1. state-domain expansion;
2. asset and human-capital node refinement;
3. control-grid refinement;
4. shorter Bellman periods;
5. stabilization of values and policies at held-out states; and
6. agreement with direct collocation along deterministic paths.

In the fine annual reference run that motivated this change, neighboring-policy
sweeps removed the lower-asset monotonicity failure. The remaining violation
was smaller and localized exactly at `asset_maximum`, identifying upper-domain
truncation. Greedy recovery modestly improved realized utility relative to
plain control interpolation, but both remained below the independently solved
direct-collocation path. Systematic joint time, state, control, and domain
convergence therefore remains necessary before using the value function for
empirical welfare calculations.

## 9. Python API

```python
from blinder_weiss import (
    BellmanConfig,
    diagnose_bellman,
    greedy_policy_at,
    policy_at,
    simulate_policy,
    solve_bellman,
)

config = BellmanConfig(
    periods=70,
    asset_nodes=31,
    human_capital_nodes=25,
    compute_platform="gpu",
)
solution = solve_bellman(config=config)
greedy_simulation = simulate_policy(solution, policy_method="greedy")
interpolated_simulation = simulate_policy(
    solution, policy_method="interpolate"
)
diagnostics = diagnose_bellman(solution, greedy_simulation)

consumption, hours, training_time = policy_at(
    solution,
    period=20,
    assets=5.0,
    human_capital=1.5,
)

greedy_consumption, greedy_hours, greedy_training = greedy_policy_at(
    solution,
    period=20,
    assets=[2.0, 5.0, 10.0],
    human_capital=[1.0, 1.5, 2.0],
)

print(solution.initial_value)
print(greedy_simulation.lifetime_utility)
print(interpolated_simulation.lifetime_utility)
print(solution.backend, solution.device)
print(diagnostics.as_dict())
```

Open the Marimo comparison notebook with:

```bash
gpu-run uv run marimo edit code/jax/notebooks/03_bellman_policy_functions.py
```

## 10. Extension to uncertainty and calibration

Suppose the model includes a discrete productivity or employment state (z)
with transition matrix (P(z'\mid z)). The recursion becomes

\[
V_n(A,y,z)=
\max_{c,h,q}
\left\{
D_u(\Delta t)u(c,h)
+\beta_{\Delta}
\sum_{z'}P(z'\mid z)V_{n+1}(A',y',z')
\right\}.
\]

For continuous shocks, the sum is replaced by quadrature or Monte Carlo
integration. The local control problem and asset-floor construction remain the
same. The state and policy arrays acquire an additional shock dimension.

This stochastic extension is what makes public panel microdata especially
valuable: PSID, SIPP, NLSY, CPS, and HRS can discipline transition processes
and state-dependent responses. The deterministic Bellman implementation should
be treated as a verified base case before adding those dimensions.

For calibration, raw microdata should not enter the Bellman recursion. Survey
code should first construct weighted empirical moments. A simulated population
then evaluates the policy functions, produces comparable model moments, and
passes their differences to an outer SMM or indirect-inference objective.

## 11. Important limitations

The present solver does not yet provide:

- stochastic earnings, employment, health, or returns;
- survival risk or an age-dependent death probability;
- taxes, transfers, pensions, or Social Security;
- household decisions or home production;
- an adaptive state domain;
- formal error bounds for bilinear interpolation; or
- differentiation of the complete backward solve with respect to structural
  parameters.

It does provide the full deterministic feedback object needed to test and
validate those extensions, without returning to finite-difference derivatives
or fragile HJB boundary formulas.
