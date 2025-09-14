from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from ev_charging_optimizer import (
    ElectricVehicle, ChargingStation,
    optimize_ev_charging_assignments_phase1,
)
from ev_charging_mcf import (
    build_ev_charging_network,
    ev_charging_ssp_potentials,
    ev_charging_cycle_canceling,
    analyze_ev_charging_solution,
)
from results_io import save_report

def load_cagliari_road_network() -> Dict[int, List[Tuple[int, float, float]]]:
    return {
        0: [(1, 1.0, 3.0), (2, 1.2, 4.0)],
        1: [(0, 1.0, 3.0), (3, 1.0, 3.0)],
        2: [(0, 1.2, 4.0), (3, 1.1, 3.5)],
        3: [(1, 1.0, 3.0), (2, 1.1, 3.5)]
    }

def load_cagliari_evs() -> List[ElectricVehicle]:
    return [
        ElectricVehicle("EV-A", origin_node=0, battery_kwh=50.0, consumption_kwh_per_km=0.16),
        ElectricVehicle("EV-B", origin_node=1, battery_kwh=45.0, consumption_kwh_per_km=0.15),
        ElectricVehicle("EV-C", origin_node=2, battery_kwh=40.0, consumption_kwh_per_km=0.14),
    ]

def load_cagliari_stations() -> List[ChargingStation]:
    return [
        ChargingStation("CA-S1", node=3, capacity=2, price_eur_per_kwh=0.55, power_kw=100.0),
        ChargingStation("CA-S2", node=1, capacity=1, price_eur_per_kwh=0.50, power_kw=50.0),
    ]

def run_cagliari_ev_optimization(solver: str = "ssp",
                                 travel_cents_per_km: int = 10,
                                 bigM_cents: int = 50_000,
                                 save_dir: Optional[str] = None,
                                 scenario_name: str = "cagliari",
                                 meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the Cagliari scenario with the chosen solver.

    If save_dir is provided, results are persisted to:
      <save_dir>/<scenario_name>/<timestamp>_<solver>/
    """
    road = load_cagliari_road_network()
    evs = load_cagliari_evs()
    stations = load_cagliari_stations()

    candidates = optimize_ev_charging_assignments_phase1(
        evs, stations, road, travel_cents_per_km=travel_cents_per_km, weight="distance"
    )

    net, ev_nodes, sink_nodes, ev_id_to_node, st_id_to_node, evst_to_arc = build_ev_charging_network(
        [e.__dict__ for e in evs], [s.__dict__ for s in stations], candidates, bigM_cents=bigM_cents
    )

    if solver == "ssp":
        summary = ev_charging_ssp_potentials(net, ev_nodes, sink_nodes)
    elif solver == "cycle":
        summary = ev_charging_cycle_canceling(net)
    else:
        raise ValueError("solver must be 'ssp' or 'cycle'")

    report = analyze_ev_charging_solution(net)
    report["mcf_summary"] = summary

    if save_dir is not None:
        meta2 = {
            "travel_cents_per_km": travel_cents_per_km,
            "bigM_cents": bigM_cents,
            "n_evs": len(evs),
            "n_stations": len(stations),
        }
        if meta:
            meta2.update(meta)
        saved_path = save_report(report, scenario=scenario_name, solver=solver, base_dir=save_dir, meta=meta2)
        report["saved_to"] = saved_path

    return report