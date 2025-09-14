# cagliari_ev_scenario.py
# Backwards-compatibility shim: expose the "real" Cagliari instance.

from cagliari_real_scenario import (
    get_cagliari_instance,
    TransportationInstance,
    SUPPLY_NAMES, SUPPLIES, DEMAND_NAMES, DEMANDS, COSTS_EUR, CAPACITIES
)

__all__ = [
    "get_cagliari_instance", "TransportationInstance",
    "SUPPLY_NAMES", "SUPPLIES", "DEMAND_NAMES", "DEMANDS",
    "COSTS_EUR", "CAPACITIES"
]