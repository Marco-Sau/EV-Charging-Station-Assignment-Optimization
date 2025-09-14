from __future__ import annotations
from typing import List, Dict, Any, Optional
from ev_charging_optimizer import (
    ElectricVehicle, ChargingStation,
    create_demo_road_network, create_demo_ev_fleet, create_demo_charging_stations,
    optimize_ev_charging_assignments_phase1,
)
from ev_charging_mcf import (
    build_ev_charging_network,
    ev_charging_ssp_potentials,
    ev_charging_cycle_canceling,
    analyze_ev_charging_solution,
)
from results_io import save_report

def run_comprehensive_optimization(solver: str = "ssp",
                                   travel_cents_per_km: int = 10,
                                   bigM_cents: int = 50_000,
                                   save_dir: Optional[str] = None,
                                   scenario_name: str = "demo",
                                   meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the grid 'demo' scenario with the chosen solver ('ssp' or 'cycle').

    If save_dir is provided, results are persisted to:
      <save_dir>/<scenario_name>/<timestamp>_<solver>/
    """
    road = create_demo_road_network(n_rows=4, n_cols=4, km_per_edge=1.0, min_per_edge=2.0)
    evs: List[ElectricVehicle] = create_demo_ev_fleet(road, n_evs=6)
    stations: List[ChargingStation] = create_demo_charging_stations(road, n_stations=4)

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
        raise ValueError("solver must be one of: 'ssp', 'cycle'")

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