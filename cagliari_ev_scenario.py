# cagliari_ev_scenario.py
# Real Cagliari scenario adapter using transportation problem data.
# This module converts the car repositioning transportation problem into an
# EV charging assignment problem, maintaining the same optimization pipeline.

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
# Real Cagliari Transportation Data
# -----------------------------

# Supply locations (EV origins) - 11 supply points
SUPPLY_NAMES = [
    "Quartu Sant'Elena","Quartucciu","Monserrato","Poetto","Su Planu",
    "La Palma","Bonaria","Pirri","Is Mirrionis","San Michele","Centro Città"
]
SUPPLIES = [50,30,40,60,10,15,25,30,10,35,45]  # Fleet sizes at each location

# Demand locations (Charging stations) - 5 demand points
DEMAND_NAMES = [
    "Cittadella universitaria","Viale Buon Cammino","Piazza Repubblica",
    "Località Marina","Piazza L'Unione Sarda"
]
DEMANDS = [80,120,60,40,50]  # Charging capacity (converted from negative demands)

# Travel costs (supply -> demand) in cost units
COSTS = [
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

# Maximum assignments per route (supply -> demand)
CAPACITIES = [
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

def build_cagliari_distance_matrix() -> Dict[Tuple[int, int], float]:
    """
    Convert the transportation cost matrix into distance estimates.
    Assumes cost is proportional to distance for EV travel.
    """
    distance_matrix = {}
    cost_to_km_factor = 0.01  # Assume 1 cost unit ≈ 0.01 km
    
    for supply_idx in range(len(SUPPLY_NAMES)):
        for demand_idx in range(len(DEMAND_NAMES)):
            supply_node = supply_idx
            demand_node = len(SUPPLY_NAMES) + demand_idx
            distance_km = COSTS[supply_idx][demand_idx] * cost_to_km_factor
            distance_matrix[(supply_node, demand_node)] = distance_km
    
    return distance_matrix

def load_cagliari_road_network() -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Build a complete bipartite road network based on the transportation data.
    Each supply location can reach each demand location directly.
    """
    road = {}
    distance_matrix = build_cagliari_distance_matrix()
    
    # Add supply nodes (EV origins)
    for supply_idx in range(len(SUPPLY_NAMES)):
        road[supply_idx] = []
        
    # Add demand nodes (charging stations)
    for demand_idx in range(len(DEMAND_NAMES)):
        demand_node = len(SUPPLY_NAMES) + demand_idx
        road[demand_node] = []
    
    # Connect supply to demand nodes
    for supply_idx in range(len(SUPPLY_NAMES)):
        for demand_idx in range(len(DEMAND_NAMES)):
            demand_node = len(SUPPLY_NAMES) + demand_idx
            distance_km = distance_matrix[(supply_idx, demand_node)]
            time_min = distance_km * 2.0  # Assume 30 km/h average speed
            
            # Bidirectional connections
            road[supply_idx].append((demand_node, distance_km, time_min))
            road[demand_node].append((supply_idx, distance_km, time_min))
    
    return road

def load_cagliari_evs() -> List[ElectricVehicle]:
    """
    Generate EV fleet based on supply locations and quantities.
    Creates multiple EVs per supply location based on the supply quantity.
    """
    evs = []
    ev_counter = 0
    
    for supply_idx, (location_name, supply_qty) in enumerate(zip(SUPPLY_NAMES, SUPPLIES)):
        # Create EVs at this supply location
        for i in range(min(supply_qty, 10)):  # Limit to 10 EVs per location for computational efficiency
            clean_name = location_name.replace(' ', '').replace("'", '')
            ev_id = f"EV-{clean_name}-{i+1}"
            
            # Vary battery capacity and consumption slightly for realism
            base_battery = 50.0
            base_consumption = 0.16
            battery_kwh = base_battery + (i % 3 - 1) * 5.0  # 45, 50, or 55 kWh
            consumption = base_consumption + (i % 3 - 1) * 0.01  # 0.15, 0.16, or 0.17 kWh/km
            
            evs.append(ElectricVehicle(
                ev_id=ev_id,
                origin_node=supply_idx,
                battery_kwh=battery_kwh,
                consumption_kwh_per_km=consumption,
                value_of_time_eur_per_h=0.0
            ))
            ev_counter += 1
    
    return evs

def load_cagliari_stations() -> List[ChargingStation]:
    """
    Generate charging stations based on demand locations and capacities.
    """
    stations = []
    
    for demand_idx, (location_name, demand_qty) in enumerate(zip(DEMAND_NAMES, DEMANDS)):
        clean_name = location_name.replace(' ', '').replace("'", '')
        station_id = f"CS-{clean_name}"
        station_node = len(SUPPLY_NAMES) + demand_idx
        
        # Vary pricing based on location (city center more expensive)
        base_price = 0.50
        if "Centro" in location_name or "Repubblica" in location_name:
            price_eur_per_kwh = base_price + 0.10  # City center premium
        elif "Marina" in location_name or "Poetto" in location_name:
            price_eur_per_kwh = base_price + 0.05  # Tourist area premium
        else:
            price_eur_per_kwh = base_price
        
        # Set capacity based on demand quantity
        capacity = min(demand_qty, 50)  # Limit for computational efficiency
        
        # Vary power rating
        power_kw = 100.0 if capacity > 30 else 50.0
        
        stations.append(ChargingStation(
            station_id=station_id,
            node=station_node,
            capacity=capacity,
            price_eur_per_kwh=price_eur_per_kwh,
            power_kw=power_kw
        ))
    
    return stations


# -----------------------------
# End-to-end runner and comparison
# -----------------------------

def run_cagliari_ev_optimization(solver: str = "ssp",
                                 travel_cents_per_km: int = 100,  # Scale up for transportation costs
                                 bigM_cents: int = 500_000) -> Dict[str, Any]:
    """
    Run EV charging optimization on the real Cagliari scenario.
    Uses scaled costs to match the transportation problem magnitude.
    """
    road = load_cagliari_road_network()
    evs = load_cagliari_evs()
    stations = load_cagliari_stations()

    print(f"Cagliari scenario: {len(evs)} EVs, {len(stations)} stations")
    print(f"EV origins: {[ev.origin_node for ev in evs[:10]]}...")  # Show first 10
    print(f"Station nodes: {[s.node for s in stations]}")

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
    
    # Add Cagliari-specific analysis
    report["cagliari_analysis"] = analyze_cagliari_assignments(report, evs, stations)
    
    return report

def analyze_cagliari_assignments(report: Dict[str, Any], evs: List[ElectricVehicle], stations: List[ChargingStation]) -> Dict[str, Any]:
    """
    Analyze the assignments in terms of the original Cagliari locations.
    """
    supply_assignments = {name: 0 for name in SUPPLY_NAMES}
    demand_assignments = {name: 0 for name in DEMAND_NAMES}
    
    for assignment in report.get("assignments", []):
        ev_id = assignment["ev_id"]
        station_id = assignment["station_id"]
        
        # Find EV origin location
        for ev in evs:
            if ev.ev_id == ev_id:
                supply_location = SUPPLY_NAMES[ev.origin_node]
                supply_assignments[supply_location] += 1
                break
        
        # Find station location
        for station in stations:
            if station.station_id == station_id:
                demand_idx = station.node - len(SUPPLY_NAMES)
                demand_location = DEMAND_NAMES[demand_idx]
                demand_assignments[demand_location] += 1
                break
    
    return {
        "supply_utilization": supply_assignments,
        "demand_utilization": demand_assignments,
        "total_assigned_evs": sum(supply_assignments.values()),
        "total_station_usage": sum(demand_assignments.values())
    }

def compare_with_transportation_problem():
    """
    Compare our EV assignment results with the original transportation problem structure.
    """
    print("\n" + "="*80)
    print("CAGLIARI SCENARIO COMPARISON")
    print("="*80)
    
    print(f"\nOriginal Transportation Problem:")
    print(f"  Supply locations: {len(SUPPLY_NAMES)} ({SUPPLY_NAMES})")
    print(f"  Supply quantities: {SUPPLIES} (total: {sum(SUPPLIES)})")
    print(f"  Demand locations: {len(DEMAND_NAMES)} ({DEMAND_NAMES})")
    print(f"  Demand quantities: {DEMANDS} (total: {sum(DEMANDS)})")
    print(f"  Total supply: {sum(SUPPLIES)}, Total demand: {sum(DEMANDS)}")
    print(f"  Balanced: {'Yes' if sum(SUPPLIES) == sum(DEMANDS) else 'No'}")
    
    # Test all three algorithms
    results = {}
    for solver in ["ssp", "cycle", "mmcc"]:
        print(f"\n{'-'*60}")
        print(f"Testing {solver.upper()} Algorithm")
        print(f"{'-'*60}")
        
        try:
            start_time = __import__('time').time()
            result = run_cagliari_ev_optimization(solver=solver)
            end_time = __import__('time').time()
            
            results[solver] = {
                "total_cost": result["total_cost_cents"],
                "assignments": len(result["assignments"]),
                "unassigned": len(result["unassigned_evs"]),
                "runtime_ms": (end_time - start_time) * 1000,
                "cagliari_analysis": result["cagliari_analysis"]
            }
            
            print(f"  Algorithm: {solver.upper()}")
            print(f"  Total cost: {result['total_cost_cents']:,} cents")
            print(f"  Assigned EVs: {len(result['assignments'])}")
            print(f"  Unassigned EVs: {len(result['unassigned_evs'])}")
            print(f"  Runtime: {results[solver]['runtime_ms']:.2f} ms")
            
            cagliari = result["cagliari_analysis"]
            print(f"  Supply utilization: {cagliari['total_assigned_evs']} EVs")
            print(f"  Station usage: {cagliari['total_station_usage']} assignments")
            
        except Exception as e:
            print(f"  ERROR in {solver}: {e}")
            results[solver] = {"error": str(e)}
    
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    
    for solver, data in results.items():
        if "error" not in data:
            print(f"{solver.upper():>6}: Cost={data['total_cost']:>8,} cents, "
                  f"Assigned={data['assignments']:>3}, Runtime={data['runtime_ms']:>6.2f}ms")
        else:
            print(f"{solver.upper():>6}: ERROR - {data['error']}")


def save_cagliari_results(results: Dict[str, Any], solver: str = "ssp"):
    """Save Cagliari scenario results in multiple formats."""
    import json
    import csv
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cagliari_{solver}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save complete JSON
    with open(f"{results_dir}/cagliari_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save assignments CSV
    if "assignments" in results and results["assignments"]:
        with open(f"{results_dir}/assignments.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results["assignments"][0].keys())
            writer.writeheader()
            writer.writerows(results["assignments"])
    
    # Save transportation comparison
    with open(f"{results_dir}/cagliari_analysis.txt", 'w') as f:
        f.write("CAGLIARI EV CHARGING OPTIMIZATION ANALYSIS\n")
        f.write("==========================================\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Algorithm: {solver.upper()}\n\n")
        
        f.write("Original Transportation Problem:\n")
        f.write(f"  Supply locations: {len(SUPPLY_NAMES)} ({', '.join(SUPPLY_NAMES)})\n")
        f.write(f"  Supply quantities: {SUPPLIES} (total: {sum(SUPPLIES)})\n")
        f.write(f"  Demand locations: {len(DEMAND_NAMES)} ({', '.join(DEMAND_NAMES)})\n")
        f.write(f"  Demand quantities: {DEMANDS} (total: {sum(DEMANDS)})\n\n")
        
        f.write("EV Optimization Results:\n")
        f.write(f"  Total cost: {results.get('total_cost_cents', 'N/A')} cents\n")
        f.write(f"  Assigned EVs: {len(results.get('assignments', []))}\n")
        f.write(f"  Unassigned EVs: {len(results.get('unassigned_evs', []))}\n\n")
        
        if "cagliari_analysis" in results:
            cagliari = results["cagliari_analysis"]
            f.write("Supply Utilization:\n")
            for location, count in cagliari["supply_utilization"].items():
                f.write(f"  {location}: {count} EVs\n")
            
            f.write("\nDemand Utilization:\n")
            for location, count in cagliari["demand_utilization"].items():
                f.write(f"  {location}: {count} assignments\n")
    
    # Save cost matrix CSV
    with open(f"{results_dir}/cagliari_transportation_data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Supply_Location", "Supply_Quantity"] + [f"Cost_to_{d}" for d in DEMAND_NAMES] + [f"Capacity_to_{d}" for d in DEMAND_NAMES]
        writer.writerow(header)
        
        for i, supply_name in enumerate(SUPPLY_NAMES):
            row = [supply_name, SUPPLIES[i]]
            row.extend(COSTS[i])
            row.extend(CAPACITIES[i])
            writer.writerow(row)
    
    print(f"✅ Cagliari results saved to: {results_dir}/")
    return results_dir

def save_cagliari_comparison():
    """Run all algorithms and save comprehensive comparison."""
    import os
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cagliari_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Test all algorithms and collect results
    algorithms_results = {}
    algorithms = ["ssp", "cycle", "mmcc"]
    
    for solver in algorithms:
        print(f"Running {solver.upper()} algorithm...")
        try:
            result = run_cagliari_ev_optimization(solver=solver)
            algorithms_results[solver] = result
            
            # Save individual result
            save_cagliari_results(result, solver)
            
        except Exception as e:
            print(f"Error with {solver}: {e}")
            algorithms_results[solver] = {"error": str(e)}
    
    # Save comparison summary
    with open(f"{results_dir}/comparison_summary.json", 'w') as f:
        json.dump(algorithms_results, f, indent=2, default=str)
    
    # Save comparison table
    with open(f"{results_dir}/algorithm_comparison.txt", 'w') as f:
        f.write("CAGLIARI ALGORITHM COMPARISON\n")
        f.write("=============================\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"{'Algorithm':<10} {'Cost':<12} {'Assigned':<10} {'Unassigned':<12} {'Status':<10}\n")
        f.write("-" * 65 + "\n")
        
        for solver, result in algorithms_results.items():
            if "error" not in result:
                cost = result.get("total_cost_cents", "N/A")
                assigned = len(result.get("assignments", []))
                unassigned = len(result.get("unassigned_evs", []))
                status = "✅ OK"
            else:
                cost = "ERROR"
                assigned = "N/A"
                unassigned = "N/A"
                status = "❌ FAIL"
            
            f.write(f"{solver.upper():<10} {str(cost):<12} {str(assigned):<10} {str(unassigned):<12} {status:<10}\n")
    
    print(f"✅ Comparison results saved to: {results_dir}/")
    return results_dir, algorithms_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        # Save comprehensive comparison
        save_cagliari_comparison()
    else:
        # Just display comparison
        compare_with_transportation_problem()