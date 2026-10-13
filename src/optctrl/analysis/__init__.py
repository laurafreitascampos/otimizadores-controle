"""Analise das leis de aprendizado como sistemas dinamicos em tempo discreto."""

from .linear import (
    closed_loop_matrix,
    gd_matrix,
    initial_state,
    momentum_boundaries,
    momentum_matrix,
    pid_paper_matrix,
    pid_separated_matrix,
    poles,
    predicted_rho,
    simulate,
    spectral_radius,
)

__all__ = ['closed_loop_matrix', 'gd_matrix', 'initial_state', 'momentum_boundaries',
           'momentum_matrix', 'pid_paper_matrix', 'pid_separated_matrix', 'poles',
           'predicted_rho', 'simulate', 'spectral_radius']
