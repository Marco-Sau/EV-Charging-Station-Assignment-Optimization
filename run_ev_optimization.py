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


def save_results(results: Dict[str, Any], solver: str = "ssp", scenario: str = "demo"):
    """Save results in multiple formats to the results directory."""
    import json
    import csv
    import os
    from datetime import datetime
    
    # Create results directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{scenario}_{solver}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON (complete data)
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save assignments as CSV
    if "assignments" in results and results["assignments"]:
        with open(f"{results_dir}/assignments.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results["assignments"][0].keys())
            writer.writeheader()
            writer.writerows(results["assignments"])
    
    # Save summary as text
    with open(f"{results_dir}/summary.txt", 'w') as f:
        f.write(f"EV Charging Optimization Results\n")
        f.write(f"================================\n\n")
        f.write(f"Scenario: {scenario}\n")
        f.write(f"Algorithm: {solver.upper()}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"Total Cost: {results.get('total_cost_cents', 'N/A')} cents\n")
        f.write(f"Assigned EVs: {len(results.get('assignments', []))}\n")
        f.write(f"Unassigned EVs: {len(results.get('unassigned_evs', []))}\n\n")
        
        if "station_utilization" in results:
            f.write("Station Utilization:\n")
            for station, count in results["station_utilization"].items():
                f.write(f"  {station}: {count} assignments\n")
        
        if "mcf_summary" in results:
            f.write(f"\nAlgorithm Details:\n")
            for key, value in results["mcf_summary"].items():
                f.write(f"  {key}: {value}\n")
    
    print(f"✅ Results saved to: {results_dir}/")
    return results_dir

def save_benchmark_results(benchmark_results: List[Dict[str, Any]]):
    """Save benchmark results in multiple formats."""
    import json
    import csv
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/benchmark_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON
    with open(f"{results_dir}/benchmark.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Save CSV
    with open(f"{results_dir}/benchmark.csv", 'w', newline='') as f:
        if benchmark_results:
            writer = csv.DictWriter(f, fieldnames=benchmark_results[0].keys())
            writer.writeheader()
            writer.writerows(benchmark_results)
    
    # Save summary
    with open(f"{results_dir}/benchmark_summary.txt", 'w') as f:
        f.write(f"EV Charging Algorithm Benchmark\n")
        f.write(f"===============================\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"{'Algorithm':<10} {'Min Time':<10} {'Avg Time':<10} {'Max Time':<10} {'Cost':<10}\n")
        f.write("-" * 60 + "\n")
        
        for result in benchmark_results:
            f.write(f"{result['solver'].upper():<10} "
                   f"{result['min_s']:<10.6f} "
                   f"{result['avg_s']:<10.6f} "
                   f"{result['max_s']:<10.6f} "
                   f"{result['total_cost_cents']:<10}\n")
    
    print(f"✅ Benchmark results saved to: {results_dir}/")
    return results_dir


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["ssp", "mmcc", "cycle"], default="ssp")
    parser.add_argument("--bench", action="store_true", help="Run benchmark comparison of all algorithms")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    args = parser.parse_args()

    if args.bench:
        results = benchmark_all(repeats=3)
        print(json.dumps(results, indent=2))
        if args.save:
            save_benchmark_results(results)
    else:
        out = run_comprehensive_optimization(solver=args.solver)
        print(json.dumps(out, indent=2))
        if args.save:
            save_results(out, solver=args.solver, scenario="demo")