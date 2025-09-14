from __future__ import annotations
import os, json, csv
from datetime import datetime
from typing import Dict, Any, List, Optional

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_report(report: Dict[str, Any],
                scenario: str,
                solver: str,
                base_dir: str = "results",
                meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Persist an MCF optimization report to disk.

    Creates:
      <base_dir>/<scenario>/<YYYYMMDD_HHMMSS>_<solver>/
        - summary.json
        - assignments.csv
        - station_utilization.csv
        - unassigned.csv
        - meta.json

    Returns the output directory path.
    """
    ts = _timestamp()
    out_dir = os.path.join(base_dir, scenario, f"{ts}_{solver}")
    _ensure_dir(out_dir)

    # 1) summary.json
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 2) assignments.csv
    assignments: List[Dict[str, Any]] = report.get("assignments", [])
    if assignments:
        headers = sorted({k for row in assignments for k in row.keys()})
        with open(os.path.join(out_dir, "assignments.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(assignments)
    else:
        with open(os.path.join(out_dir, "assignments.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ev_id","station_id","distance_km","time_min","energy_kwh","travel_cost_cents","charging_cost_cents","total_cost_cents","path_nodes"])

    # 3) station_utilization.csv
    util = report.get("station_utilization", {})
    with open(os.path.join(out_dir, "station_utilization.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "utilization"])
        for sid, u in util.items():
            w.writerow([sid, u])

    # 4) unassigned.csv
    unassigned = report.get("unassigned_evs", [])
    with open(os.path.join(out_dir, "unassigned.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ev_id"])
        for ev in unassigned:
            w.writerow([ev])

    # 5) meta.json
    meta_all = {"scenario": scenario, "solver": solver}
    if meta:
        meta_all.update(meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)

    return out_dir