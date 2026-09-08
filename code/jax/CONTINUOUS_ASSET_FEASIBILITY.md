# Continuous feasibility of the constant-control asset path

The default `BellmanConfig.asset_feasibility="continuous"` constrains assets
throughout each Bellman interval. The old `"checkpoints"` mode remains available
for reproducing historical policies and numerical comparisons. `path_checkpoints`
has no effect on the continuous bound. Saved configurations from before this
change must explicitly load as `"checkpoints"`; silently assigning them the new
feasible-control set would mislabel historical results.

This is exact treatment of the interval's **constant-control approximation**,
up to floating-point arithmetic and the bounded scalar solve described below.
It does not eliminate error from holding controls constant or prove convergence
of values, policies, distributions or calibration. The positive numerical
asset floor still approximates the model's economic borrowing constraint.

## Why finite checkpoints were insufficient

The audited baseline policy at A=0.0001 and log K=0 used consumption
0.3166230692032 for a half-year interval, with hours 0.7072046322267 and
training time 0.4536915522068. It respected all four configured checkpoints
but reached minimum assets about 0.00006919638 between them. This was below
the numerical floor by about 0.00003080362, although still above economic zero.
The new bound lowers consumption to 0.3156394017199 for those fixed hours and
training, and its full-period minimum is the initial numerical floor.

## Exact path minimum for fixed controls

Let w be initial earnings, g the human-capital growth rate, and r the interest
rate. With fixed consumption c, hours h and training q,

\[
w=h\,G(q/h)\,K_0,\quad g=a q-\delta,\qquad
\dot A(t)=r A(t)+w e^{gt}-c.
\]

The model computes w with its division-safe `effective_earnings_share` function.
Define D0=r A0+w-c and d=g-r. Then

\[
\dot A(t)=e^{rt}\left[D_0+w g\,t\,\operatorname{exprel}(d t)\right],
\qquad \operatorname{exprel}(z)=\frac{e^z-1}{z}.
\]

The derivative of the bracket is w g exp(d t), so there is at most one
interior stationary point. With zero earnings or nonpositive growth, endpoints
suffice for the minimum. With w g>0, an interior minimum exists only if the
bracket is negative initially and positive at the endpoint. Its time is

\[
t_* = \frac{\log1p[-d D_0/(w g)]}{d},
\]

with limit -D0/(w g) when d approaches zero. The implementation evaluates a
stable `log1p(z)/z` expression and checks A(0), A(dt), and A(t*) when applicable.
`minimum_assets_during_step(state, control, params, step)` returns this minimum
for every leading batch index. This covers zero interest and g near r without
numerically differentiating the value function or sampling an arbitrary mesh.

## Full-period consumption capacity

`endpoint_consumption_capacity(...)` always denotes the endpoint-only bound.
It uses the stable discounted-budget expression

\[
C(t)=r\underline A+
\frac{A_0-\underline A}{t\operatorname{exprel}(-rt)}+
w\frac{\operatorname{exprel}((g-r)t)}{\operatorname{exprel}(-rt)}.
\]

The full-period bound is infimum C(t) over 0<t<=dt. For nonpositive earnings
growth or zero earnings, the endpoint is sufficient. Starting exactly at the
floor with positive earnings growth, the limiting constraint at t=0 gives
Cmax=r floor+w. This initial condition was missed by positive-time checkpoints.

For positive initial wealth above the floor and growing earnings, an interior
binding time can occur. Write B=A0-floor and

\[
J(t)=w t\left[e^{gt}\operatorname{exprel}(-rt)
-\operatorname{exprel}((g-r)t)\right].
\]

Its derivative is w g exp(g t) t exprel(-r t)>0. When the endpoint cap would
leave assets increasing at a binding endpoint, the unique root J(t*)=B lies
inside the interval. A fixed 32-step JAX bisection locates this time, after
which the implementation evaluates C(t*). Otherwise it evaluates C(dt).
Batched inputs and static shapes preserve compilation reuse. The loop is
skipped when the entire batch needs only endpoint/initial constraints.

## Derivatives used by the optimizer

Differentiating the branch decisions of bisection does not give a reliable
root derivative. Instead, the implementation uses the envelope theorem:
C'(t*)=0 at an interior minimizing time, so differentiating C(t*) while
stopping the derivative of t* gives the capacity derivative. At an endpoint,
the duration remains differentiable; stopping it would lose the time-step
derivative. At the initial floor constraint the analytic limiting capacity is
used. Genuine switches between active constraints can still be nonsmooth.

Focused tests compare directional derivatives over assets, log K, hours,
training, interest, productivity, depreciation, duration and the numerical
floor against several finite-difference steps. Tests avoid crossing a genuine
constraint switch when checking a single smooth derivative.

## Diagnostics and compatibility

All Bellman optimizer, neighbor, conditional-consumption, greedy rollout and
convergence checks use the configured feasibility method. Population backends
use the same method. Exact endpoint atom classification must call
`endpoint_consumption_capacity`, not the full-period bound with one checkpoint.

`BellmanSimulation.minimum_assets` now audits the whole interval, including
legacy simulations. `BellmanDiagnostics.minimum_node_path_assets` audits all
stored node policies. Continuous-mode acceptance checks the full-period
minimum; legacy-mode replay can expose between-checkpoint dips without
silently changing historical controls.

Validation includes the earlier missed dip, independent positive-integrand
quadrature plus a bracketed root for consumption capacity, independent scalar
minimization of the asset path, zero earnings/interest, negative interest,
zero growth and growth near interest, scalar/batched shapes, feasible and
slightly excessive consumption, derivative checks, and a complete small
Bellman solution. These establish interval feasibility, not lifecycle mesh or
calibration convergence.
