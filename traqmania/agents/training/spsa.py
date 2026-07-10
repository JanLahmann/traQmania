"""Simultaneous Perturbation Stochastic Approximation (SPSA), pure numpy.

SPSA estimates the gradient of a scalar loss ``f(x)`` from exactly TWO
evaluations per iteration, independent of ``dim(x)``: both evaluations
perturb ALL coordinates at once along a random Rademacher direction
``delta`` (each entry independently +1 or -1),

    g_hat = (f(x + c_k * delta) - f(x - c_k * delta)) / (2 * c_k) * delta,

which is an unbiased-to-first-order estimate of the true gradient.  That
two-evaluations property is exactly what makes it the standard optimizer for
quantum hardware, where every loss evaluation is a batch of real circuit
executions and parameter-shift gradients would cost 2 * P evaluations.

Gain sequences follow Spall's standard recommendation:

    a_k = a / (k + 1 + A) ** 0.602        (step size, decaying)
    c_k = c / (k + 1) ** 0.101            (perturbation size, decaying)

with the stability offset ``A`` defaulting to ``iterations / 10``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    iterations: int,
    a: float = 0.15,
    c: float = 0.1,
    A: float | None = None,
    seed: int | None = 0,
    callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Minimize ``f`` with SPSA; exactly ``2 * iterations`` evaluations of ``f``.

    Returns ``{"x": final params, "loss_history": [...]}`` where
    ``loss_history[k]`` is the mean of the two evaluations at iteration ``k``
    (an unbiased smoothed estimate of ``f`` near the current iterate).
    ``callback(k, info)`` receives ``x``, ``loss``, ``f_plus``, ``f_minus``,
    ``a_k`` and ``c_k`` after each iteration.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    if A is None:
        A = iterations / 10.0
    rng = np.random.default_rng(seed)

    loss_history: list[float] = []
    for k in range(int(iterations)):
        a_k = a / (k + 1 + A) ** 0.602
        c_k = c / (k + 1) ** 0.101
        delta = rng.choice(np.array([-1.0, 1.0]), size=x.shape)

        f_plus = float(f(x + c_k * delta))
        f_minus = float(f(x - c_k * delta))
        # 1/delta == delta for Rademacher entries, so this IS the SPSA estimate.
        g_hat = (f_plus - f_minus) / (2.0 * c_k) * delta
        x = x - a_k * g_hat

        loss = 0.5 * (f_plus + f_minus)
        loss_history.append(loss)
        if callback is not None:
            callback(
                k,
                {
                    "x": x.copy(),
                    "loss": loss,
                    "f_plus": f_plus,
                    "f_minus": f_minus,
                    "a_k": a_k,
                    "c_k": c_k,
                },
            )

    return {"x": x, "loss_history": loss_history}
