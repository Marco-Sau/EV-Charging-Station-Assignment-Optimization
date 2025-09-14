# run_ev_optimization.py
"""
One-command runner for the Cagliari real scenario.
- Run both algorithms by default, or choose specific algorithm: "ssp", "cycle", or "both"
- Fixed seed for reproducibility
- Saves results under results/
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from datetime import datetime

from cagliari_real_scenario import get_cagliari_instance
from ev_charging_mcf import (
    build_transportation_network,
    cycle_canceling,
    ssp_with_potentials,
    total_cost,
    extract_nonzero_flows,
)

DEFAULT_SEED = 42


def ensure_results_dir(path: str = "results") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_results(payload: dict, algo: str, outdir: str = "results") -> None:
    ensure_results_dir(outdir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"cagliari_{algo}_{stamp}"
    json_path = os.path.join(outdir, base + ".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[✓] Results saved: {json_path}")


def run(algo: str = "ssp", seed: int = DEFAULT_SEED) -> dict:
    random.seed(seed)

    inst = get_cagliari_instance()
    supply_names = inst["supply_names"]
    supplies = inst["supplies"]
    demand_names = inst["demand_names"]
    demands = inst["demands"]           # negative
    caps = inst["capacities"]
    costs = inst["costs_eur"]

    ns, nd = len(supplies), len(demands)
    print(f"Cagliari scenario: {ns} supplies, {nd} demands; total = {sum(supplies)} = {-sum(demands)}")

    net = build_transportation_network(supplies, demands, costs, caps)

    t0 = time.time()
    if algo.lower() == "ssp":
        ssp_with_potentials(net)
    elif algo.lower() == "cycle":
        cycle_canceling(net)
    else:
        raise ValueError("Unknown algorithm. Use 'ssp' or 'cycle'.")
    dt = time.time() - t0

    obj = total_cost(net, ns, nd, costs)
    flows = extract_nonzero_flows(net, ns, nd, supply_names, demand_names, costs)

    result = {
        "algorithm": algo.lower(),
        "seed": seed,
        "objective_eur": obj,
        "num_arcs_with_flow": len(flows),
        "flows": flows,
        "timing_sec": dt,
        "instance": {
            "total_supply": sum(supplies),
            "total_demand": -sum(demands),
            "ns": ns,
            "nd": nd,
        },
    }
    save_results(result, algo.lower())
    print(f"[✓] Done. Algorithm={algo}, Objective={obj:.2f} EUR, Time={dt:.4f}s, Flows={len(flows)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Cagliari MCF optimization")
    parser.add_argument("--algo", type=str, default="both", choices=["ssp", "cycle", "both"], help="MCF algorithm(s) to run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (for reproducibility hooks)")
    args = parser.parse_args()
    
    if args.algo == "both":
        print("Running both SSP and Cycle-Canceling algorithms...")
        print("=" * 50)
        run(algo="ssp", seed=args.seed)
        print("=" * 50)
        run(algo="cycle", seed=args.seed)
        print("=" * 50)
        print("Both algorithms completed!")
    else:
        run(algo=args.algo, seed=args.seed)


if __name__ == "__main__":
    main()