# Code map

`legacy/` contains the original project code and is retained for historical
reference.  It includes experimental finite-difference HJB notebooks that are
not intended for further development.

`reference/` contains MATLAB code used as source material.  It is not an
active dependency of this project.

Put the replacement JAX solver in `jax/` and future R analysis in `r/`.  Keep
new work out of `legacy/` so it stays clear which implementation is current.
