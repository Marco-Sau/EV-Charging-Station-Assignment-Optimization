from __future__ import annotations
import os, csv, json, argparse, sys
from datetime import datetime
from typing import Dict, Any, List

from run_ev_optimization import run_comprehensive_optimization
from cagliari_ev_scenario import run_cagliari_ev_optimization

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def run_all(
    save_dir: str = "results",
    scenarios: List[str] = ("demo", "cagliari"),
    solvers: List[str] = ("ssp", "cycle"),
    travel_cents_per_km: int = 10,
    bigM_cents: int = 50_000,
) -> List[Dict[str, Any]]:
    _ensure_dir(save_dir)
    runs: List[Dict[str, Any]] = []
    for scen in scenarios:
        for solver in solvers:
            meta = {
                "travel_cents_per_km": travel_cents_per_km,
                "bigM_cents": bigM_cents,
            }
            if scen == "demo":
                rep = run_comprehensive_optimization(
                    solver=solver,
                    travel_cents_per_km=travel_cents_per_km,
                    bigM_cents=bigM_cents,
                    save_dir=save_dir,
                    scenario_name="demo",
                    meta=meta,
                )
            elif scen == "cagliari":
                rep = run_cagliari_ev_optimization(
                    solver=solver,
                    travel_cents_per_km=travel_cents_per_km,
                    bigM_cents=bigM_cents,
                    save_dir=save_dir,
                    scenario_name="cagliari",
                    meta=meta,
                )
            else:
                raise ValueError(f"Unknown scenario: {scen}")

            runs.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario": scen,
                "solver": solver,
                "assigned": len(rep.get("assignments", [])),
                "unassigned": len(rep.get("unassigned_evs", [])),
                "total_cost_cents": int(rep.get("total_cost_cents", 0)),
                "saved_to": rep.get("saved_to", ""),
                "station_utilization": rep.get("station_utilization", {}),
            })

    # Consolidated index
    idx_dir = _ensure_dir(os.path.join(save_dir))
    idx_csv = os.path.join(idx_dir, "_index.csv")
    idx_json = os.path.join(idx_dir, "_index.json")

    fieldnames = ["timestamp", "scenario", "solver", "assigned", "unassigned", "total_cost_cents", "saved_to"]
    need_header = not os.path.exists(idx_csv) or os.path.getsize(idx_csv) == 0
    with open(idx_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with open(idx_json, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    return runs

def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description="Run all EV charging optimization scenarios with both solvers and persist results.")
    p.add_argument("--save-dir", default="results", help="Base directory to save outputs.")
    p.add_argument("--scenarios", nargs="+", default=["demo", "cagliari"], choices=["demo", "cagliari"],
                   help="Which scenarios to run.")
    p.add_argument("--solvers", nargs="+", default=["ssp", "cycle"], choices=["ssp", "cycle"],
                   help="Which MCF solvers to run.")
    p.add_argument("--travel-cents-per-km", type=int, default=10, help="Travel cost in cents per km.")
    p.add_argument("--bigM-cents", type=int, default=50_000, help="Penalty (cents) for EV→T unassigned arcs.")
    args = p.parse_args(argv)

    runs = run_all(
        save_dir=args.save_dir,
        scenarios=args.scenarios,
        solvers=args.solvers,
        travel_cents_per_km=args.travel_cents_per_km,
        bigM_cents=args.bigM_cents,
    )

    print("\n== EV Charging Optimization: consolidated run summary ==")
    for r in runs:
        print(f"[{r['scenario']:^8}] solver={r['solver']:<6}  assigned={r['assigned']:<3}  "
              f"unassigned={r['unassigned']:<3}  total_cost={r['total_cost_cents']:<8}  saved_to={r['saved_to']}")
    print("\nIndex written to:")
    print(f"  {os.path.join(args.save_dir, '_index.csv')}")
    print(f"  {os.path.join(args.save_dir, '_index.json')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())