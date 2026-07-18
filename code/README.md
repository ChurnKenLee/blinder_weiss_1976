# Code map

`legacy/` contains the original project code and is retained for historical
reference.  It includes experimental finite-difference HJB notebooks that are
not intended for further development.

`reference/` contains MATLAB code used as source material.  It is not an
active dependency of this project.

`jax/blinder_weiss/` contains the reusable JAX model, direct-collocation path
solver, and semi-Lagrangian Bellman policy solver. `jax/notebooks/` contains
the Marimo front ends. Future R analysis belongs in `r/`.

Keep new work out of `legacy/` so it stays clear which implementation is
current.
