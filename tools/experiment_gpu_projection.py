"""Prototype joint bicubic derivative projection on a GPU, then certify it.

Uses Chambolle--Pock primal-dual updates with diagonal row/column scaling:
https://optimization-online.org/wp-content/uploads/2010/06/2646.pdf
The CPU QP/certificate helper is independently implemented in the experiments
folder. This tool does not modify the production interpolation algorithm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_reference_helper(directory: Path):
    spec = importlib.util.spec_from_file_location(
        "joint_derivative_reference", directory / "joint_derivative_projection.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def padded_rows(matrix):
    matrix = matrix.tocsr()
    counts = np.diff(matrix.indptr)
    columns = np.zeros((matrix.shape[0], int(counts.max())), dtype=np.int32)
    coefficients = np.zeros_like(columns, dtype=np.float64)
    for row, count in enumerate(counts):
        start = matrix.indptr[row]
        columns[row, :count] = matrix.indices[start : start + count]
        coefficients[row, :count] = matrix.data[start : start + count]
    return jnp.asarray(columns), jnp.asarray(coefficients)


def flat_components(values):
    """Equal orthogonal derivatives are required along classified-flat edges."""
    na, ny = values.shape
    node_ids = np.arange(na * ny).reshape(na, ny)
    edges = []
    for axis in (0, 1):
        left = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
        right = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
        tolerance = 32 * np.finfo(values.dtype).eps * np.maximum(abs(left), abs(right))
        flat = abs(right - left) <= tolerance
        first = np.take(node_ids, np.arange(values.shape[axis] - 1), axis=axis)[flat]
        second = np.take(node_ids, np.arange(1, values.shape[axis]), axis=axis)[flat]
        offset = (1 - axis) * na * ny
        edges.extend(zip(first + offset, second + offset, strict=True))
    indices = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    graph = sparse.coo_matrix(
        (np.ones(len(indices)), (indices[:, 0], indices[:, 1])),
        shape=(2 * na * ny, 2 * na * ny),
    )
    return connected_components(graph, directed=False)[1].astype(np.int32)


def make_projector(problem, *, consensus):
    full_matrix = problem["A"].tocsr()
    n = full_matrix.shape[1]
    full_counts = np.diff(full_matrix.indptr)
    boxes = full_counts == 1
    box_rows = np.flatnonzero(boxes)
    entries = full_matrix.indptr[box_rows]
    indices = full_matrix.indices[entries]
    coefficients = full_matrix.data[entries]
    first = problem["l"][box_rows] / coefficients
    second = problem["u"][box_rows] / coefficients
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    np.maximum.at(lower, indices, np.minimum(first, second))
    np.minimum.at(upper, indices, np.maximum(first, second))
    matrix = full_matrix[~boxes]
    row_ids, row_coefficients = padded_rows(matrix)
    col_ids, col_coefficients = padded_rows(matrix.T)
    row_norm = np.asarray(abs(matrix).sum(axis=1)).ravel()
    col_norm = np.asarray(abs(matrix).sum(axis=0)).ravel()
    # Cauchy--Schwarz gives ||sqrt(Sigma) A sqrt(Tau)|| <= .99.
    sigma = jnp.asarray(0.99 / np.maximum(row_norm, 1e-300))
    tau = jnp.asarray(0.99 / np.maximum(col_norm, 1e-300))
    lower_j, upper_j = map(jnp.asarray, (lower, upper))
    bound_l, bound_u = map(jnp.asarray, (problem["l"][~boxes], problem["u"][~boxes]))
    reference = jnp.asarray(problem["reference"])
    weight = jnp.asarray(problem["P_diagonal"])
    initial = jnp.asarray(problem["x0"])
    if consensus:
        assert np.all(problem["active"])
        groups = jnp.asarray(flat_components(problem["values"]))
        group_lower = jax.ops.segment_max(lower_j, groups, num_segments=n)
        group_upper = jax.ops.segment_min(upper_j, groups, num_segments=n)
        denominator = jax.ops.segment_sum(1 / tau + weight, groups, num_segments=n)
        safe_denominator = jnp.where(denominator > 0, denominator, 1.0)

    def multiply(vector):
        return jnp.sum(row_coefficients * vector[row_ids], axis=1)

    def transpose_multiply(vector):
        return jnp.sum(col_coefficients * vector[col_ids], axis=1)

    def prox(vector):
        numerator = vector / tau + weight * reference
        if consensus:
            means = jax.ops.segment_sum(numerator, groups, num_segments=n) / safe_denominator
            return jnp.clip(means, group_lower, group_upper)[groups]
        return jnp.clip(numerator / (1 / tau + weight), lower_j, upper_j)

    @jax.jit
    def run(iterations):
        def step(_, carry):
            x, y, extrapolated = carry
            trial_y = y + sigma * multiply(extrapolated)
            y = trial_y - sigma * jnp.clip(trial_y / sigma, bound_l, bound_u)
            updated = prox(x - tau * transpose_multiply(y))
            return updated, y, 2 * updated - x

        start = initial
        x, _, _ = jax.lax.fori_loop(
            0, iterations, step, (start, jnp.zeros(matrix.shape[0]), start)
        )
        return x

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("output/solver_benchmarks/experiments"))
    parser.add_argument("--prefix", default="joint_fullgrid_134")
    parser.add_argument("--iterations", type=int, nargs="+", default=[25, 100, 400, 1600, 6400])
    args = parser.parse_args()
    helper = load_reference_helper(args.directory)
    with np.load(args.directory / f"{args.prefix}_vectors.npz") as data:
        problem = {key: data[key] for key in data.files}
    problem["A"] = sparse.load_npz(args.directory / f"{args.prefix}_matrix.npz")
    report = {"problem": args.prefix, "device": str(jax.devices()[0]), "variants": []}
    for consensus in (False, True):
        run = make_projector(problem, consensus=consensus)
        run(jnp.asarray(1)).block_until_ready()
        for iterations in args.iterations:
            started = perf_counter()
            vector = run(jnp.asarray(iterations))
            vector.block_until_ready()
            elapsed = perf_counter() - started
            vector = np.asarray(vector)
            table, info = helper.finish_joint_projection(problem, vector)
            applied = problem["A"] @ vector
            violation = float(np.maximum(problem["l"] - applied, applied - problem["u"]).max())
            row = dict(consensus=consensus, iterations=iterations, seconds=elapsed,
                       qp_violation=max(0.0, violation), **info)
            report["variants"].append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(args.directory / f"gpu_projection_{int(consensus)}_{iterations}.npz",
                                table=table, vector=vector)
    temporary = args.directory / "gpu_projection_audit.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.directory / "gpu_projection_audit.json")


if __name__ == "__main__":
    main()
