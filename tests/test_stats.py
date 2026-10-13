"""Confere a distribuicao t e os testes contra valores tabelados conhecidos."""

import math

import pytest

from optctrl.stats import betainc_reg, paired, t_pvalue_two_sided, t_quantile, welch


@pytest.mark.parametrize('t,df,p', [
    (2.262157, 9, 0.05),        # t_{0.975, 9}
    (3.249836, 9, 0.01),        # t_{0.995, 9}
    (2.093024, 19, 0.05),
    (1.959964, 1e7, 0.05),      # tende a normal
    (0.0, 5, 1.0),
])
def test_t_pvalue_table(t, df, p):
    assert math.isclose(t_pvalue_two_sided(t, df), p, rel_tol=1e-5, abs_tol=1e-9)


def test_t_quantile_inverts_pvalue():
    assert math.isclose(t_quantile(0.975, 9), 2.262157, rel_tol=1e-5)


def test_betainc_symmetry():
    assert math.isclose(betainc_reg(2.5, 1.5, 0.3), 1 - betainc_reg(1.5, 2.5, 0.7), rel_tol=1e-12)


def test_welch_known_example():
    # exemplo calculado a mao: medias 5 e 3, variancias amostrais 2.5 e 2.5, n = 5
    x, y = [3, 4, 5, 6, 7], [1, 2, 3, 4, 5]
    r = welch(x, y)
    assert math.isclose(r['diff'], 2.0)
    assert math.isclose(r['t'], 2.0)
    assert math.isclose(r['df'], 8.0)
    assert math.isclose(r['p'], 0.0805, abs_tol=1e-3)


def test_paired_detects_consistent_small_shift():
    """Diferenca pequena mas igual em toda semente: Welch nao ve, pareado ve."""
    base = [0.970, 0.980, 0.975, 0.985, 0.972, 0.978, 0.981, 0.969, 0.977, 0.983]
    x = [b + 0.002 for b in base]
    assert welch(x, base)['p'] > 0.3
    assert paired(x, base)['p'] < 1e-6
