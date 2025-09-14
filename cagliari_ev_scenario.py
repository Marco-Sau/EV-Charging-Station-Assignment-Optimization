# cagliari_ev_scenario.py
# Cagliari real transportation instance (supplies, demands, costs in EUR, capacities)
from __future__ import annotations
from typing import List, Dict, Any

# Names
SUPPLY_NAMES: List[str] = [
    "Quartu Sant'Elena","Quartucciu","Monserrato","Poetto","Su Planu",
    "La Palma","Bonaria","Pirri","Is Mirrionis","San Michele","Centro Città"
]
DEMAND_NAMES: List[str] = [
    "Cittadella Universitaria","Viale Buon Cammino","Piazza Repubblica",
    "Località Marina","Piazza L'Unione Sarda"
]

# Supply/Demand quantities (units) — totals must balance (350)
SUPPLIES: List[int] = [50,30,40,60,10,15,25,30,10,35,45]
DEMANDS: List[int] = [-80,-120,-60,-40,-50]

# Unit costs in EUR (converted from the original cent values /100)
COSTS_EUR: List[List[float]] = [
    [89.50,103.00, 86.00, 97.00,153.30],
    [76.00, 91.00, 74.30, 88.00, 43.80],
    [35.70, 88.70, 65.00, 82.00,122.30],
    [154.50,111.00, 80.00, 89.00,120.00],
    [71.00, 56.30, 71.30, 68.00, 68.30],
    [117.00, 67.00, 36.00, 46.00, 75.30],
    [132.70, 30.00, 16.00, 21.00, 45.00],
    [39.00, 54.00, 66.50, 78.00, 75.70],
    [68.00, 21.00, 44.70, 49.00, 31.70],
    [72.00, 16.00, 35.70, 14.00, 42.00],
    [71.50, 17.00, 14.50, 28.00, 44.30],
]

# Maximum arc capacities (units)
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


def scenario_summary() -> Dict[str, Any]:
    return {
        "n_supplies": len(SUPPLIES),
        "n_demands": len(DEMANDS),
        "total_supply": sum(SUPPLIES),
        "total_demand_abs": -sum(DEMANDS),
        "balanced": sum(SUPPLIES) == -sum(DEMANDS),
        "min_unit_cost_eur": min(min(r) for r in COSTS_EUR),
        "max_unit_cost_eur": max(max(r) for r in COSTS_EUR),
    }