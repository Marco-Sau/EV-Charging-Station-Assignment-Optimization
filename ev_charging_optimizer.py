#!/usr/bin/env python3
"""
Orchestration for the Cagliari scenario:
- seeding for reproducibility
- run Cycle-Canceling and SSP
- save results in deterministic order under results/<timestamp>/ (or custom dir)
"""

import os
import sys
import csv
import json
import time
import random
from typing import Dict, List, Optional

from cagliari_real_scenario import (
    SUPPLY_NAMES, DEMAND_NAMES, SUPPLIES, DEMANDS, COSTS_EUR, CAPACITIES,
    build_transportation_network,
)
from ev_charging_mcf import (
    Network, cycle_canceling, successive_shortest_path,
)


# ----------------------------- Reproducibility ------------------------------

def seed_everything(seed: int) -> None:
    """Best-effort seeding for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np  # optional
        np.random.seed(seed)
    except Exception:
        pass
    phs = os.environ.get("PYTHONHASHSEED", None)
    if phs is None or phs in {"random", ""}:
        print("[WARN] PYTHONHASHSEED is not fixed. For full reproducibility run as:",
              f"PYTHONHASHSEED=0 python run_ev_optimization.py --seed {seed}",
              file=sys.stderr)


# ----------------------------- I/O helpers ---------------------------------

def _stable_sorted_flows(rows: List[Dict]) -> List[Dict]:
    """Deterministic ordering for tabular output."""
    return sorted(rows, key=lambda r: (r["from"], r["to"]))

def _ensure_output_dir(output_root: Optional[str]) -> str:
    root = output_root or "results"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"cagliari_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_solution(name: str, out_dir: str, objective_eur: float, flows: List[Dict], seed: int) -> None:
    flows_sorted = _stable_sorted_flows(flows)
    # CSV
    csv_path = os.path.join(out_dir, f"flows_{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from", "to", "flow", "unit_cost_eur", "arc_cost_eur"])
        w.writeheader()
        for r in flows_sorted:
            w.writerow(r)
    # JSON summary
    summary_path = os.path.join(out_dir, f"summary_{name}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "algorithm": name,
            "objective_eur": objective_eur,
            "nonzero_arcs": len(flows_sorted),
            "seed": seed
        }, f, indent=2)

def save_report(out_dir: str, cycle_obj: float, ssp_obj: float, seed: int) -> None:
    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as f:
        json.dump({
            "scenario": "Cagliari (transportation/MCF in EUR)",
            "seed": seed,
            "algorithms": {
                "cycle_canceling": {"objective_eur": cycle_obj},
                "ssp": {"objective_eur": ssp_obj}
            }
        }, f, indent=2)


# ----------------------------- Solvers wrappers -----------------------------

def _collect_solution_from_net(net: Network) -> Dict:
    """Collect objective and flows for supply->demand arcs (excludes super arcs)."""
    ns, nd = len(SUPPLIES), len(DEMANDS)
    flows = []
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            f = net.get_arc_flow(aid)
            if f > 0:
                unit = COSTS_EUR[i][j]
                flows.append({
                    "from": SUPPLY_NAMES[i],
                    "to": DEMAND_NAMES[j],
                    "flow": int(f),
                    "unit_cost_eur": unit,
                    "arc_cost_eur": unit * int(f)
                })
    obj = sum(r["arc_cost_eur"] for r in flows)
    return {"objective_eur": obj, "flows": flows}

def optimize_with_cycle_canceling() -> Dict:
    net = build_transportation_network()
    cycle_canceling(net)
    return _collect_solution_from_net(net)

def optimize_with_ssp() -> Dict:
    net = build_transportation_network()
    successive_shortest_path(net)
    return _collect_solution_from_net(net)


# ----------------------------- Public pipeline -----------------------------

def main_optimization_pipeline(seed: int = 42, output_dir: Optional[str] = None) -> Dict:
    seed_everything(seed)

    print(">> Cagliari MCF optimization (EUR)")
    print(f"   Supplies: {sum(SUPPLIES)}  |  Demands: {sum(DEMANDS)}")
    print(f"   Matrix: {len(SUPPLIES)} x {len(DEMANDS)}  (capacitated arcs)\n")

    # Cycle-Canceling
    print("   [1/2] Cycle-Canceling ...")
    cc = optimize_with_cycle_canceling()
    print(f"        objective: €{cc['objective_eur']:.2f}, nonzero arcs: {len(cc['flows'])}")

    # SSP
    print("   [2/2] Successive Shortest Path ...")
    ssp = optimize_with_ssp()
    print(f"        objective: €{ssp['objective_eur']:.2f}, nonzero arcs: {len(ssp['flows'])}")

    out_dir = _ensure_output_dir(output_dir)
    save_solution("cycle_canceling", out_dir, cc["objective_eur"], cc["flows"], seed)
    save_solution("ssp", out_dir, ssp["objective_eur"], ssp["flows"], seed)
    save_report(out_dir, cc["objective_eur"], ssp["objective_eur"], seed)

    print(f"\n[OK] Results written to: {out_dir}")
    return {
        "output_dir": out_dir,
        "seed": seed,
        "cycle_canceling": cc,
        "ssp": ssp
    }