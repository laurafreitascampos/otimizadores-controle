"""Metricas de convergencia e de resposta transitoria."""

import numpy as np


def epochs_to_target(curve, target):
    """Primeira epoca (1-indexada) em que curve >= target, ou None."""
    for i, value in enumerate(curve):
        if value >= target:
            return i + 1
    return None


def overshoot(trajectory, theta_star, theta0=None):
    """Sobressinal de uma trajetoria escalar, NORMALIZADO pela distancia inicial:

        OS = max(0, max_t s*(theta_t - theta*)) / |theta0 - theta*|,
        s = sign(theta* - theta0)

    A Eq. (8) de An et al. divide por theta*, o que fica indefinido no toy de
    De Jong (theta* = 0). Esta e a definicao usual de controle (resposta ao
    degrau de theta0 para theta*), equivalente a Eq. (8) quando theta0 = 0.
    """
    traj = np.asarray(trajectory, dtype=float)
    theta0 = traj[0] if theta0 is None else theta0
    dist = theta_star - theta0
    if dist == 0:
        return 0.0
    s = np.sign(dist)
    beyond = s * (traj - theta_star)            # > 0 quando passa do alvo
    return float(max(0.0, beyond.max()) / abs(dist))


def settling_step(trajectory, theta_star, tol=0.02, theta0=None):
    """Primeiro passo apos o qual |theta_t - theta*| <= tol*|theta0 - theta*| para sempre."""
    traj = np.asarray(trajectory, dtype=float)
    theta0 = traj[0] if theta0 is None else theta0
    band = tol * abs(theta0 - theta_star)
    outside = np.nonzero(np.abs(traj - theta_star) > band)[0]
    if len(outside) == 0:
        return 0
    last = outside[-1] + 1
    return int(last) if last < len(traj) else None
