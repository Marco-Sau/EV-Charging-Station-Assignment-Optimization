# ev_charging_optimizer.py
# EV-feasibility (Phase 1) and data models + demo generators.
# Implements a simple, robust feasibility search: for each EV, run Dijkstra
# on the road graph (nonnegative costs), then check energy feasibility and compute
# travel/charging costs per reachable station. For larger resource models, you can
# replace dijkstra_shortest_paths() with a multi-criteria label-correcting variant.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import heapq
import math

# ---------------------
# Data models
# ---------------------

@dataclass
class ElectricVehicle:
    ev_id: str
    origin_node: int
    battery_kwh: float
    consumption_kwh_per_km: float  # energy usage per km
    value_of_time_eur_per_h: float = 0.0  # optional monetization of time

@dataclass
class ChargingStation:
    station_id: str
    node: int
    capacity: int
    price_eur_per_kwh: float
    power_kw: float = 50.0  # used only for charging-time estimation

@dataclass
class EVLabel:
    node: int
    dist_km: float
    time_min: float
    energy_kwh: float
    prev: Optional[int]

class ChargingCostCalculator:
    """
    Encapsulates travel and charging cost models.
    - Travel cost: cents per km + (optional) monetized time
    - Charging cost: energy_to_recover_kwh * station_price (in cents)
    """
    def __init__(self,
                 travel_cents_per_km: int = 10,
                 include_time_value: bool = True):
        self.travel_cents_per_km = travel_cents_per_km
        self.include_time_value = include_time_value

    def travel_cost_cents(self, distance_km: float, time_min: float, ev: ElectricVehicle) -> int:
        cost = self.travel_cents_per_km * distance_km
        if self.include_time_value and ev.value_of_time_eur_per_h > 0:
            cost += 100 * ev.value_of_time_eur_per_h * (time_min / 60.0)
        return int(round(cost))

    def charging_cost_cents(self, energy_kwh: float, station: ChargingStation) -> int:
        return int(round(100.0 * energy_kwh * station.price_eur_per_kwh))

    def charging_time_min(self, energy_kwh: float, station: ChargingStation) -> float:
        # Simplified constant-power model
        if station.power_kw <= 0:
            return math.inf
        return 60.0 * energy_kwh / station.power_kw


# ---------------------
# Road network
# ---------------------

# RoadGraph: node -> List[(neighbor, distance_km, time_min)]
RoadGraph = Dict[int, List[Tuple[int, float, float]]]

def dijkstra_shortest_paths(road: RoadGraph, source: int, weight: str = "distance") -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Optional[int]]]:
    """
    Single-source shortest path with nonnegative weights.
    weight: "distance" or "time" (controls primary optimization).
    Returns (dist_km, time_min, parent).
    """
    dist_km = {u: math.inf for u in road}
    time_min = {u: math.inf for u in road}
    parent: Dict[int, Optional[int]] = {u: None for u in road}
    dist_km[source] = 0.0
    time_min[source] = 0.0

    def key_val(d, t):
        return d if weight == "distance" else t

    pq: List[Tuple[float, int]] = [(0.0, source)]
    visited = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in visited: 
            continue
        visited.add(u)
        for v, dk, tm in road.get(u, []):
            nd = dist_km[u] + dk
            nt = time_min[u] + tm
            if key_val(nd, nt) < key_val(dist_km[v], time_min[v]):
                dist_km[v] = nd
                time_min[v] = nt
                parent[v] = u
                heapq.heappush(pq, (key_val(nd, nt), v))
    return dist_km, time_min, parent

def reconstruct_path(parent: Dict[int, Optional[int]], target: int) -> List[int]:
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = parent[v]
    return list(reversed(path))


# ---------------------
# ECSP-style feasibility
# ---------------------

def is_compatible(ev: ElectricVehicle, station: ChargingStation) -> bool:
    # Placeholder compatibility logic; extend as needed (connector types, etc.)
    return True

def ev_feasible_stations(ev: ElectricVehicle,
                         stations: List[ChargingStation],
                         road: RoadGraph,
                         cost_calc: ChargingCostCalculator,
                         weight: str = "distance") -> List[Dict]:
    """
    For a single EV, compute feasible station options:
      1) run Dijkstra from EV origin,
      2) for each station, check energy feasibility (distance * consumption <= battery),
      3) compute travel + charging costs and times.
    Returns a list of assignment candidates with metadata.
    """
    dist_km, time_min, parent = dijkstra_shortest_paths(road, ev.origin_node, weight=weight)

    candidates: List[Dict] = []
    for st in stations:
        if not is_compatible(ev, st):
            continue
        d = dist_km.get(st.node, math.inf)
        t = time_min.get(st.node, math.inf)
        if not math.isfinite(d):
            continue  # unreachable
        energy_used = d * ev.consumption_kwh_per_km
        if energy_used > ev.battery_kwh:
            continue  # infeasible w.r.t. battery (no intermediate charging modeled)
        travel_cents = cost_calc.travel_cost_cents(d, t, ev)
        charge_cents = cost_calc.charging_cost_cents(energy_used, st)  # simple "replenish energy used" model
        charge_time = cost_calc.charging_time_min(energy_used, st)
        total_cents = travel_cents + charge_cents
        path_nodes = reconstruct_path(parent, st.node)
        candidates.append({
            "ev_id": ev.ev_id,
            "station_id": st.station_id,
            "distance_km": d,
            "time_min": t,
            "energy_kwh": energy_used,
            "travel_cost_cents": travel_cents,
            "charging_cost_cents": charge_cents,
            "charging_time_min": charge_time,
            "total_cost_cents": total_cents,
            "path_nodes": path_nodes,
        })
    return candidates


# ---------------------
# Demo data generators
# ---------------------

def create_demo_road_network(n_rows: int = 4, n_cols: int = 4,
                             km_per_edge: float = 1.0,
                             min_per_edge: float = 2.0) -> RoadGraph:
    """
    Build a small grid graph for demos: nodes are 0..(R*C-1), 4-neighborhood.
    Distances/time are constant per edge.
    """
    def nid(r, c): return r * n_cols + c
    road: RoadGraph = {nid(r, c): [] for r in range(n_rows) for c in range(n_cols)}
    for r in range(n_rows):
        for c in range(n_cols):
            u = nid(r, c)
            if r + 1 < n_rows:
                v = nid(r + 1, c)
                road[u].append((v, km_per_edge, min_per_edge))
                road[v].append((u, km_per_edge, min_per_edge))
            if c + 1 < n_cols:
                v = nid(r, c + 1)
                road[u].append((v, km_per_edge, min_per_edge))
                road[v].append((u, km_per_edge, min_per_edge))
    return road

def create_demo_ev_fleet(road: RoadGraph, n_evs: int = 6) -> List[ElectricVehicle]:
    nodes = list(road.keys())
    evs: List[ElectricVehicle] = []
    for i in range(n_evs):
        evs.append(ElectricVehicle(
            ev_id=f"EV{i+1}",
            origin_node=nodes[i % len(nodes)],
            battery_kwh=40.0,
            consumption_kwh_per_km=0.15,  # ~15 kWh / 100 km
            value_of_time_eur_per_h=0.0
        ))
    return evs

def create_demo_charging_stations(road: RoadGraph, n_stations: int = 4) -> List[ChargingStation]:
    nodes = list(road.keys())
    step = max(1, len(nodes)//n_stations)
    stations: List[ChargingStation] = []
    for i in range(n_stations):
        stations.append(ChargingStation(
            station_id=f"S{i+1}",
            node=nodes[(i*step) % len(nodes)],
            capacity= max(1, (i % 3) + 1),
            price_eur_per_kwh=0.50 + 0.05 * (i % 3),
            power_kw=50.0
        ))
    return stations


# ---------------------
# Orchestration: Phase 1
# ---------------------

def optimize_ev_charging_assignments_phase1(evs: List[ElectricVehicle],
                                            stations: List[ChargingStation],
                                            road: RoadGraph,
                                            travel_cents_per_km: int = 10,
                                            weight: str = "distance") -> List[Dict]:
    """
    Phase 1: for each EV, enumerate feasible station options.
    Returns a flat list of candidate dicts (one per EV-station option).
    """
    cost_calc = ChargingCostCalculator(travel_cents_per_km=travel_cents_per_km)
    all_candidates: List[Dict] = []
    for ev in evs:
        cand = ev_feasible_stations(ev, stations, road, cost_calc, weight=weight)
        all_candidates.extend(cand)
    return all_candidates