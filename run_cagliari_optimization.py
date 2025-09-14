# run_cagliari_optimization.py
# Runner for the Cagliari transportation instance with SSP and Cycle-Canceling
from __future__ import annotations

import argparse
from typing import Optional, Dict, Any
from ev_charging_mcf import (
    successive_shortest_path,
    cycle_canceling,
    analyze_transportation_solution,
    save_solution_json_csv,
    set_global_seed,
)
from cagliari_real_scenario import build_transportation_network_eur


def run(algo: str, seed: Optional[int], output_dir: str) -> Dict[str, Any]:
    set_global_seed(seed)

    net, meta = build_transportation_network_eur()
    ns, nd = meta["ns"], meta["nd"]

    if algo == "cycle":
        print("[RUN] Algorithm: Cycle-Canceling")
        cycle_canceling(net)
    elif algo == "ssp":
        print("[RUN] Algorithm: Successive Shortest Path (with potentials)")
        successive_shortest_path(net)
    else:
        raise ValueError("Unknown algorithm. Use 'cycle' or 'ssp'.")

    result = analyze_transportation_solution(
        net,
        ns,
        nd,
        meta["supply_names"],
        meta["demand_names"],
        meta["costs_eur"],
    )

    shipped = sum(row["flow"] for row in result["flows"])
    print(f"[RESULT] Total shipped units: {shipped}")
    print(f"[RESULT] Objective (EUR): {result['objective_eur']:.2f}")

    # Save
    tag = "cycle" if algo == "cycle" else "ssp"
    save_solution_json_csv(output_dir, tag, seed, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Cagliari EV reassignment via MCF (transportation).")
    parser.add_argument("--algorithm", choices=["cycle", "ssp"], default="cycle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    run(args.algorithm, args.seed, args.output_dir)


if __name__ == "__main__":
    main()