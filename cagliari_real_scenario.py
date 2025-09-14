# cagliari_real_scenario.py
# Real Cagliari transportation instance (supplies/demands/costs/capacities)
from __future__ import annotations
from typing import List, Tuple, Dict, Any

from ev_charging_mcf import Network

# ----------------------------- Instance Data -----------------------------

SUPPLY_NAMES: List[str] = [
    "Quartu Sant'Elena","Quartucciu","Monserrato","Poetto","Su Planu",
    "La Palma","Bonaria","Pirri","Is Mirrionis","San Michele","Centro Città"
]
SUPPLIES: List[int] = [50,30,40,60,10,15,25,30,10,35,45]  # total = 350

DEMAND_NAMES: List[str] = [
    "Cittadella universitaria","Viale Buon Cammino","Piazza Repubblica",
    "Località Marina","Piazza L'Unione Sarda"
]
DEMANDS: List[int] = [-80,-120,-60,-40,-50]  # sum = -350

# Costs originally provided as *cents*; we convert to EUROS (float) by /100.
COSTS_CENTS: List[List[int]] = [
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


def costs_eur() -> List[List[float]]:
    return [[c / 100.0 for c in row] for row in COSTS_CENTS]


def build_transportation_network_eur() -> Tuple[Network, Dict[str, Any]]:
    """
    Build the base transportation network (supply -> demand arcs).
    Costs in EUROS (float). Capacities integer.
    Returns (network, metadata).
    """
    net = Network()
    ns, nd = len(SUPPLIES), len(DEMANDS)
    net.add_n_nodes(ns, SUPPLIES)
    net.add_n_nodes(nd, DEMANDS)

    C = costs_eur()
    for i in range(ns):
        for j in range(nd):
            net.add_arc(i, ns + j, cost=C[i][j], capacity=CAPACITIES[i][j], flow=0)

    meta = {
        "ns": ns,
        "nd": nd,
        "supply_names": SUPPLY_NAMES,
        "demand_names": DEMAND_NAMES,
        "costs_eur": C,
    }
    return net, meta