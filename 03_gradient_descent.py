"""Gradient Descent Implementation"""

import numpy as np


def gradient_descent(grad_fn, x0, learning_rate=0.1, n_iterations=1000, tol=1e-6):
    """Generic gradient descent for a differentiable scalar-valued function.

    Args:
        grad_fn: callable that takes x (array) and returns the gradient (same shape).
        x0: starting point (array-like).
        learning_rate: step size for each update.
        n_iterations: maximum number of iterations.
        tol: stop early when the gradient norm drops below this threshold.

    Returns:
        x: the final point found by the optimizer.
        history: list of (x, grad_norm) tuples recorded at each step.
    """
    x = np.array(x0, dtype=float)
    history = []

    for _ in range(n_iterations):
        grad = np.asarray(grad_fn(x), dtype=float)
        grad_norm = np.linalg.norm(grad)
        history.append((x.copy(), grad_norm))

        if grad_norm < tol:
            break

        x = x - learning_rate * grad

    return x, history


if __name__ == "__main__":
    # Demo: minimize f(x, y) = (x - 3)^2 + 2 * (y + 1)^2
    # True minimum at (3, -1) with f = 0

    def f(x):
        return (x[0] - 3) ** 2 + 2 * (x[1] + 1) ** 2

    def grad(x):
        return np.array([2 * (x[0] - 3), 4 * (x[1] + 1)])

    x0 = np.array([0.0, 0.0])
    x_final, history = gradient_descent(grad, x0, learning_rate=0.1, n_iterations=500)

    print(f"Starting point: {x0}")
    print(f"Final point:    {np.round(x_final, 4)}")
    print(f"True minimum:   [3. -1.]")
    print(f"Iterations:     {len(history)}")
    print(f"Final grad norm: {history[-1][1]:.2e}")
