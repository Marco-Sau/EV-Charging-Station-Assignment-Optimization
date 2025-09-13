# run_ev_optimization.py
# End-to-end pipeline:
#   Phase 1: ECSP-style feasibility enumeration
#   Phase 2: Build MCF and solve via "ssp" | "mmcc" | "cycle"
#   Reporting utilities and quick CLI

from __future__ import annotations
from typing import List, Dict, Any
from ev_charging_optimizer import (
    ElectricVehicle, ChargingStation,
    create_demo_road_network, create_demo_ev_fleet, create_demo_charging_stations,
    optimize_ev_charging_assignments_phase1,
)
from ev_charging_mcf import (
    build_ev_charging_network,
    ev_charging_ssp_potentials,
    ev_charging_min_mean_cycle_canceling,
    ev_charging_cycle_canceling,
    analyze_ev_charging_solution,
)

def run_comprehensive_optimization(solver: str = "ssp",
                                   travel_cents_per_km: int = 10,
                                   bigM_cents: int = 50_000) -> Dict[str, Any]:
    """
    Build a demo, run ECSP feasibility, construct MCF, solve, and analyze.
    solver in {"ssp", "mmcc", "cycle"}.
    """
    # Demo data
    road = create_demo_road_network(n_rows=4, n_cols=4, km_per_edge=1.0, min_per_edge=2.0)
    evs: List[ElectricVehicle] = create_demo_ev_fleet(road, n_evs=6)
    stations: List[ChargingStation] = create_demo_charging_stations(road, n_stations=4)

    # Phase 1: enumerate feasible EV->Station options
    candidates = optimize_ev_charging_assignments_phase1(evs, stations, road,
                                                         travel_cents_per_km=travel_cents_per_km,
                                                         weight="distance")

    # Phase 2: build network and solve
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
        raise ValueError("solver must be one of: 'ssp', 'mmcc', 'cycle'")

    report = analyze_ev_charging_solution(net)
    report["mcf_summary"] = summary
    return report


# -------------------------
# Lightweight benchmark
# -------------------------

def benchmark_all(repeats: int = 3) -> List[Dict[str, Any]]:
    import time
    results: List[Dict[str, Any]] = []
    for method in ("cycle", "mmcc", "ssp"):
        times = []
        totals = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            rep = run_comprehensive_optimization(solver=method)
            dt = time.perf_counter() - t0
            times.append(dt)
            totals.append(rep["total_cost_cents"])
        results.append({"solver": method, "min_s": min(times), "avg_s": sum(times)/len(times),
                        "max_s": max(times), "total_cost_cents": totals[-1]})
    return results


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["ssp", "mmcc", "cycle"], default="ssp")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    if args.bench:
        print(json.dumps(benchmark_all(repeats=3), indent=2))
    else:
        out = run_comprehensive_optimization(solver=args.solver)
        print(json.dumps(out, indent=2))