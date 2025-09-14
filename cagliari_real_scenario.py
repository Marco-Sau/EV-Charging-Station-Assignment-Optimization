# cagliari_real_scenario.py
"""
Cagliari real transportation instance (supplies/demands, capacities, costs in euros)
"""

from __future__ import annotations
from typing import Dict, List

# Supply (origins) – names and amounts (sum = 350)
SUPPLY_NAMES: List[str] = [
    "Quartu Sant'Elena","Quartucciu","Monserrato","Poetto","Su Planu",
    "La Palma","Bonaria","Pirri","Is Mirrionis","San Michele","Centro Città"
]
SUPPLIES: List[int] = [50, 30, 40, 60, 10, 15, 25, 30, 10, 35, 45]

# Demand (destinations) – names and amounts (sum = 350)
DEMAND_NAMES: List[str] = [
    "Cittadella Universitaria","Viale Buon Cammino","Piazza Repubblica",
    "Località Marina","Piazza L'Unione Sarda"
]
# Convention: negative values for demand nodes (network supplies), as in textbook transportation models
DEMANDS: List[int] = [-80, -120, -60, -40, -50]

# Arc capacities (per (supply i, demand j))
CAPACITIES: List[List[int]] = [
    [14,18,14,5,19],
    [14,15,21,5,17],
    [17,18,13,5,19],
    [18,14,13,8,20],
    [7,15,19,7,21],
    [16,19,13,8,20],
    [23,19,12,8,19],
    [5,20,13,5,19],
    [6,7,17,5,15],
    [12,5,15,7,19],
    [11,5,10,5,16],
]

# Costs originally provided in cents → converted here to euros (float)
_COSTS_CENTS: List[List[int]] = [
    [8950,10300,8600,9700,15330],
    [7600,9100,7430,8800,4380],
    [3570,8870,6500,8200,12230],
    [15450,11100,8000,8900,12000],
    [7100,5630,7130,6800,6830],
    [11700,6700,3600,4600,7530],
    [13270,3000,1600,2100,4500],
    [3900,5400,6650,7800,7570],
    [6800,2100,4470,4900,3170],
    [7200,1600,3570,1400,4200],
    [7150,1700,1450,2800,4430],
]
COSTS_EUR: List[List[float]] = [[c / 100.0 for c in row] for row in _COSTS_CENTS]


def get_cagliari_instance() -> Dict:
    """Return the real Cagliari instance, with costs in euros."""
    return {
        "supply_names": SUPPLY_NAMES,
        "supplies": SUPPLIES,
        "demand_names": DEMAND_NAMES,
        "demands": DEMANDS,          # negative for demand nodes
        "capacities": CAPACITIES,
        "costs_eur": COSTS_EUR,
    }