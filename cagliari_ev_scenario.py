# cagliari_ev_scenario.py
# Realistic scenario adapter for Cagliari.
# This module provides the same orchestration API but uses simple synthetic
# generators unless you plug in real data loaders (TODO sections indicated).

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from ev_charging_optimizer import (
    ElectricVehicle, ChargingStation,
    optimize_ev_charging_assignments_phase1,
)
from ev_charging_mcf import (
    build_ev_charging_network,
    ev_charging_ssp_potentials,
    ev_charging_min_mean_cycle_canceling,
    ev_charging_cycle_canceling,
    analyze_ev_charging_solution,
)

# -----------------------------
# TODO: Replace with real data
# -----------------------------

def load_cagliari_road_network() -> Dict[int, List[Tuple[int, float, float]]]:
    """
    TODO: Load your real road graph here (node -> [(nbr, distance_km, time_min), ...]).
    For now, we return a tiny synthetic graph to keep the pipeline runnable.
    """
    road = {
        0: [(1, 1.0, 3.0), (2, 1.2, 4.0)],
        1: [(0, 1.0, 3.0), (3, 1.0, 3.0)],
        2: [(0, 1.2, 4.0), (3, 1.1, 3.5)],
        3: [(1, 1.0, 3.0), (2, 1.1, 3.5)]
    }
    return road

def load_cagliari_evs() -> List[ElectricVehicle]:
    """
    TODO: Build from your datasets. Here we provide a small synthetic fleet.
    """
    return [
        ElectricVehicle("EV-A", origin_node=0, battery_kwh=50.0, consumption_kwh_per_km=0.16, value_of_time_eur_per_h=0.0),
        ElectricVehicle("EV-B", origin_node=1, battery_kwh=45.0, consumption_kwh_per_km=0.15, value_of_time_eur_per_h=0.0),
        ElectricVehicle("EV-C", origin_node=2, battery_kwh=40.0, consumption_kwh_per_km=0.14, value_of_time_eur_per_h=0.0),
    ]

def load_cagliari_stations() -> List[ChargingStation]:
    """
    TODO: Build from your station inventory. Synthetic placeholders here.
    """
    return [
        ChargingStation("CA-S1", node=3, capacity=2, price_eur_per_kwh=0.55, power_kw=100.0),
        ChargingStation("CA-S2", node=1, capacity=1, price_eur_per_kwh=0.50, power_kw=50.0),
    ]


# -----------------------------
# End-to-end runner
# -----------------------------

def run_cagliari_ev_optimization(solver: str = "ssp",
                                 travel_cents_per_km: int = 10,
                                 bigM_cents: int = 50_000) -> Dict[str, Any]:
    road = load_cagliari_road_network()
    evs = load_cagliari_evs()
    stations = load_cagliari_stations()

    candidates = optimize_ev_charging_assignments_phase1(evs, stations, road,
                                                         travel_cents_per_km=travel_cents_per_km,
                                                         weight="distance")
    net, ev_nodes, sink_nodes, ev_id_to_node, st_id_to_node, evst_to_arc = build_ev_charging_network(
        [e.__dict__ for e in evs], [s.__dict__ for s in stations], candidates, bigM_cents=bigM_cents
    )

    if solver == "ssp":
        summary = ev_charging_ssp_potentials(net, ev_nodes, sink_nodes)
    elif solver == "mmcc":
        summary = ev_charging_min_mean_cycle_canceling(net)
    elif solver == "cycle":
        summary = ev_charging_cycle_canceling(net)
    else:
        raise ValueError("solver must be 'ssp', 'mmcc', or 'cycle'")

    report = analyze_ev_charging_solution(net)
    report["mcf_summary"] = summary
    return report


if __name__ == "__main__":
    import json
    out = run_cagliari_ev_optimization(solver="ssp")
    print(json.dumps(out, indent=2))