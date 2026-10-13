"""Medicao opcional de energia/CO2 com CodeCarbon (pip install -e ".[energy]").

Migrado de tune_mnist.py. Se o CodeCarbon nao estiver instalado, o rastreador
vira um no-op e os campos ficam None.
"""

import contextlib

# Fator de emissao da rede eletrica brasileira (matriz com muita hidro).
BRASIL_KG_CO2_POR_KWH = 0.12

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except Exception:  # noqa: BLE001
    HAS_CODECARBON = False


@contextlib.contextmanager
def track_energy(enabled=True, kg_co2_per_kwh=BRASIL_KG_CO2_POR_KWH):
    """Uso:
        with track_energy() as e:
            treinar()
        e['energy_kwh'], e['co2_kg']
    """
    out = {'energy_kwh': None, 'co2_kg': None}
    if not (enabled and HAS_CODECARBON):
        yield out
        return
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                               log_level='error')
    tracker.start()
    try:
        yield out
    finally:
        tracker.stop()
        kwh = float(tracker.final_emissions_data.energy_consumed)
        out['energy_kwh'] = kwh
        out['co2_kg'] = kwh * kg_co2_per_kwh
