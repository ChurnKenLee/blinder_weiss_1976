"""Keep unit tests independent of an interactive process occupying the GPU."""

from __future__ import annotations

import os

# GPU execution is exercised explicitly from the Bellman notebook. The unit
# suite is small and should remain deterministic while that notebook owns VRAM.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
