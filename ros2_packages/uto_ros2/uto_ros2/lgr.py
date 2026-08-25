"""Legendre-Gauss-Radau operators matching the MATLAB reference implementation."""

import numpy as np


def barycentric_weights(nodes):
    nodes = np.asarray(nodes, dtype=float)
    weights = np.ones(len(nodes))
    for j in range(len(nodes)):
        weights[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    return weights / np.max(np.abs(weights))


def derivative_at_node(nodes, weights, index):
    column = np.zeros(len(nodes))
    for j in range(len(nodes)):
        if j != index:
            column[j] = weights[j] / (weights[index] * (nodes[index] - nodes[j]))
    column[index] = -column.sum()
    return column


def interpolation_matrix(nodes, queries):
    nodes = np.asarray(nodes, float)
    queries = np.atleast_1d(queries).astype(float)
    weights = barycentric_weights(nodes)
    result = np.empty((len(nodes), len(queries)))
    for k, query in enumerate(queries):
        nearest = int(np.argmin(np.abs(query - nodes)))
        if abs(query - nodes[nearest]) <= 32 * np.finfo(float).eps * max(1, abs(query)):
            result[:, k] = 0
            result[nearest, k] = 1
        else:
            inverse = weights / (query - nodes)
            result[:, k] = inverse / inverse.sum()
    return result


def quadrature_weights(nodes):
    nodes = np.asarray(nodes, float)
    k = len(nodes)
    vandermonde = np.vstack([nodes**degree for degree in range(k)])
    moments = np.array([2 / (degree + 1) if degree % 2 == 0 else 0 for degree in range(k)])
    return np.linalg.solve(vandermonde, moments)


def lgr_operators(nodes_per_region):
    """Return K LGR nodes and Kx(K+1) differentiation matrix."""
    k = int(nodes_per_region)
    if k < 1:
        raise ValueError("nodes_per_region must be positive")
    if k == 1:
        tau = np.array([-1.0])
    else:
        m = k - 1
        diagonal = np.zeros(m)
        off = np.zeros(m - 1)
        for n in range(m):
            # Jacobi(alpha=0,beta=1), exactly as lgrOperators() in MATLAB.
            diagonal[n] = 1 / ((2 * n + 1) * (2 * n + 3))
        for n in range(1, m):
            off[n - 1] = (
                2 / (2 * n + 1) * np.sqrt(n * n * (n + 1) * (n + 1) / ((2 * n) * (2 * n + 2)))
            )
        tau = np.r_[
            -1.0,
            np.sort(np.linalg.eigvalsh(np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1))),
        ]
    support = np.r_[tau, 1.0]
    weights = barycentric_weights(support)
    full = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        full[i] = derivative_at_node(support, weights, i)
    return tau, full[:k]


def control_check_grid(nodes, dense_points=31):
    """MATLAB-compatible union of nodes, endpoint, midpoints and dense checks."""
    nodes = np.asarray(nodes, dtype=float)
    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    return np.unique(np.r_[nodes, 1.0, midpoints, np.linspace(-1.0, 1.0, dense_points)])


def interpolate_control(nodes, node_values, queries):
    """Evaluate a control polynomial; node_values has shape [control,K]."""
    return np.asarray(node_values, dtype=float) @ interpolation_matrix(nodes, queries)
