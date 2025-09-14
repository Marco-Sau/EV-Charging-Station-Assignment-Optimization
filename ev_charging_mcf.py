from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from heapq import heappush, heappop

INF = 10**18

# =============================
# Residual network primitives
# =============================

class EVChargingNetwork:
    def __init__(self):
        self.adj: List[List[Dict[str, Any]]] = []
        self.kind: List[Optional[str]] = []
        self.node_meta: List[Optional[Dict[str, Any]]] = []
        self._n = 0

    def add_node(self, kind: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> int:
        nid = self._n
        self._n += 1
        self.adj.append([])
        self.kind.append(kind)
        self.node_meta.append(meta)
        return nid

    def add_arc(self, u: int, v: int, cap: int, cost: int, meta: Optional[Dict[str, Any]] = None) -> None:
        fwd = {"to": v, "rev": len(self.adj[v]), "cap": cap, "cost": cost,
               "flow": 0, "is_forward": True, "meta": meta}
        rev = {"to": u, "rev": len(self.adj[u]), "cap": 0, "cost": -cost,
               "flow": 0, "is_forward": False, "meta": None}
        self.adj[u].append(fwd)
        self.adj[v].append(rev)

def residual_push(network: EVChargingNetwork, u: int, ei: int, delta: int) -> None:
    arc = network.adj[u][ei]
    v, rev = arc["to"], arc["rev"]
    arc["cap"] -= delta
    network.adj[v][rev]["cap"] += delta
    if arc["is_forward"]:
        arc["flow"] += delta
    else:
        fwd = network.adj[v][rev]
        fwd["flow"] -= delta

# =========================================
# Network builder for assignment MCF model
# =========================================

def build_ev_charging_network(
    evs, stations, feasible_assignments, bigM_cents: int = 50_000
):
    net = EVChargingNetwork()
    ev_nodes: List[int] = []
    st_nodes: List[int] = []
    ev_id_to_node: Dict[str, int] = {}
    st_id_to_node: Dict[str, int] = {}

    # EV nodes
    for ev in evs:
        ev_id = ev["ev_id"] if isinstance(ev, dict) else ev.ev_id
        nid = net.add_node(kind="ev", meta={"ev_id": ev_id})
        ev_nodes.append(nid)
        ev_id_to_node[ev_id] = nid

    # Station nodes
    for st in stations:
        st_id = st["station_id"] if isinstance(st, dict) else st.station_id
        cap = int(st["capacity"] if isinstance(st, dict) else st.capacity)
        nid = net.add_node(kind="station", meta={"station_id": st_id, "capacity": cap})
        st_nodes.append(nid)
        st_id_to_node[st_id] = nid

    # Super nodes
    S = net.add_node(kind="S")
    T = net.add_node(kind="T")

    # S->EV arcs (cap 1, cost 0)
    for e in ev_nodes:
        net.add_arc(S, e, cap=1, cost=0)

    # Station->T arcs (capacity from data, cost 0)
    for s, st in zip(st_nodes, stations):
        cap = int(st["capacity"] if isinstance(st, dict) else st.capacity)
        net.add_arc(s, T, cap=cap, cost=0)

    # EV->Station assignment arcs
    evst_to_arc: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for opt in feasible_assignments:
        e = ev_id_to_node[opt["ev_id"]]
        s = st_id_to_node[opt["station_id"]]
        c = int(opt["total_cost_cents"])
        meta = {k: opt.get(k) for k in ["ev_id","station_id","distance_km","time_min","energy_kwh","travel_cost_cents","charging_cost_cents","path_nodes"]}
        ei = len(net.adj[e])
        net.add_arc(e, s, cap=1, cost=c, meta=meta)
        evst_to_arc[(opt["ev_id"], opt["station_id"])] = (e, ei)

    # EV->T "unassigned" arcs (cost Big-M)
    for ev_id, e in ev_id_to_node.items():
        net.add_arc(e, T, cap=1, cost=bigM_cents, meta={"ev_id": ev_id, "unassigned": True})

    # Circulation arcs (close cycles + initial feasible flow)
    net.add_arc(T, S, cap=10**9, cost=0)              # close cycles
    total_supply = len(ev_nodes)
    net.add_arc(S, T, cap=total_supply, cost=bigM_cents)  # initial feasible flow when saturated

    return net, ev_nodes, [T], ev_id_to_node, st_id_to_node, evst_to_arc

# ==============================================
# Feasible-circulation bootstrap for canceling
# ==============================================

def _saturate_S_to_T_for_feasible_circulation(network: EVChargingNetwork) -> None:
    S_nodes = [i for i, k in enumerate(network.kind) if k == "S"]
    T_nodes = [i for i, k in enumerate(network.kind) if k == "T"]
    if not S_nodes or not T_nodes:
        return
    S, T = S_nodes[0], T_nodes[0]
    for ei, arc in enumerate(network.adj[S]):
        if arc["is_forward"] and arc["to"] == T:
            cap = arc["cap"]
            if cap > 0:
                residual_push(network, S, ei, cap)
            return

# ===============================
# (a) Generic cycle-canceling
# ===============================

def bellman_ford_negative_cycle(network: EVChargingNetwork):
    n = len(network.adj)
    dist = [0] * n
    pred: List[Optional[Tuple[int, int]]] = [None] * n
    updated = -1
    for _ in range(n):
        updated = -1
        for u in range(n):
            for ei, arc in enumerate(network.adj[u]):
                if arc["cap"] <= 0:
                    continue
                v = arc["to"]
                nd = dist[u] + arc["cost"]
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = (u, ei)
                    updated = v
        if updated == -1:
            return [], 0
    if updated == -1:
        return [], 0
    x = updated
    for _ in range(n):
        if pred[x] is None:
            break
        x = pred[x][0]
    if pred[x] is None:
        return [], 0
    cycle_nodes = [x]
    y = pred[x][0]
    guard = 0
    while y != x and y is not None and guard <= n + 5:
        cycle_nodes.append(y)
        if pred[y] is None:
            return [], 0
        y = pred[y][0]
        guard += 1
    if y != x:
        return [], 0
    edges: List[Tuple[int, int]] = []
    cyc_cost = 0
    for v in cycle_nodes:
        pu = pred[v]
        if pu is None:
            return [], 0
        u, ei = pu
        edges.append((u, ei))
        cyc_cost += network.adj[u][ei]["cost"]
    edges.reverse()
    return edges, cyc_cost

def ev_charging_cycle_canceling(network: EVChargingNetwork, max_iters: int = 10**6):
    _saturate_S_to_T_for_feasible_circulation(network)
    iters = 0
    total_delta = 0
    while iters < max_iters:
        cycle, cyc_cost = bellman_ford_negative_cycle(network)
        if not cycle or cyc_cost >= 0:
            break
        bottleneck = min(network.adj[u][ei]["cap"] for (u, ei) in cycle)
        for (u, ei) in cycle:
            residual_push(network, u, ei, bottleneck)
        total_delta += bottleneck * cyc_cost
        iters += 1
    return {"iterations": iters, "total_cost_change": total_delta, "terminated": (iters < max_iters)}

# =========================================
# (b) Successive Shortest Path (potentials)
# =========================================

class SSPPotentialsSolver:
    def __init__(self):
        pass

    def solve(self, network: EVChargingNetwork, sources: List[int], sinks: List[int]):
        n = len(network.adj)
        assigned = 0
        augments = 0
        total_reduced_cost = 0
        deficit = set(sinks)
        for s in sources:
            pi = [0] * n  # reset per EV (simple but fine for small instances)
            dist, parent = self._dijkstra_reduced(network, s, pi, deficit)
            t = self._best_sink(dist, deficit)
            if t is None or dist[t] >= INF:
                continue
            path = self._reconstruct(parent, s, t)
            if not path:
                continue
            for (u, ei) in path:
                residual_push(network, u, ei, 1)
            assigned += 1
            augments += 1
            total_reduced_cost += dist[t]
        return {"assigned": assigned, "augmentations": augments, "total_cost_reduced": total_reduced_cost, "pi": []}

    def _dijkstra_reduced(self, network: EVChargingNetwork, s: int, pi: List[int], deficit: set):
        n = len(network.adj)
        dist = [INF] * n
        parent: List[Optional[Tuple[int, int]]] = [None] * n
        dist[s] = 0
        seen = [False] * n
        pq: List[Tuple[int, int]] = [(0, s)]
        while pq:
            d, u = heappop(pq)
            if seen[u]:
                continue
            seen[u] = True
            if u in deficit:
                break
            for ei, arc in enumerate(network.adj[u]):
                if arc["cap"] <= 0:
                    continue
                v = arc["to"]
                rc = arc["cost"] + pi[u] - pi[v]
                nd = d + rc
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    heappush(pq, (nd, v))
        return dist, parent

    def _best_sink(self, dist: List[int], deficit: set):
        best, node = INF, None
        for t in deficit:
            if dist[t] < best:
                best, node = dist[t], t
        return node

    def _reconstruct(self, parent, s: int, t: int):
        path: List[Tuple[int, int]] = []
        v = t
        while v != s:
            pu = parent[v]
            if pu is None:
                return []
            u, ei = pu
            path.append((u, ei))
            v = u
        path.reverse()
        return path

def ev_charging_ssp_potentials(network: EVChargingNetwork, ev_source_nodes: List[int], terminal_sink_nodes: List[int]):
    solver = SSPPotentialsSolver()
    return solver.solve(network, ev_source_nodes, terminal_sink_nodes)

# ==============================
# Solution analysis / reporting
# ==============================

def analyze_ev_charging_solution(network: EVChargingNetwork):
    assignments: List[Dict[str, Any]] = []
    station_util: Dict[str, int] = {}
    unassigned: List[str] = []

    node_to_station_id: Dict[int, str] = {i: (network.node_meta[i]["station_id"]) 
                                          for i, k in enumerate(network.kind) if k == "station"}
    node_to_ev_id: Dict[int, str] = {i: (network.node_meta[i]["ev_id"]) 
                                     for i, k in enumerate(network.kind) if k == "ev"}
    T_nodes = [i for i, k in enumerate(network.kind) if k == "T"]

    for u in range(len(network.adj)):
        if network.kind[u] != "ev":
            continue
        ev_id = node_to_ev_id[u]
        chosen_station: Optional[str] = None
        for ei, arc in enumerate(network.adj[u]):
            if arc["is_forward"] and arc["flow"] > 0:
                v = arc["to"]
                if v in node_to_station_id:
                    st_id = node_to_station_id[v]
                    meta = arc.get("meta", {}) or {}
                    assignments.append({
                        "ev_id": ev_id,
                        "station_id": st_id,
                        "distance_km": meta.get("distance_km"),
                        "time_min": meta.get("time_min"),
                        "energy_kwh": meta.get("energy_kwh"),
                        "travel_cost_cents": meta.get("travel_cost_cents"),
                        "charging_cost_cents": meta.get("charging_cost_cents"),
                        "total_cost_cents": (meta.get("travel_cost_cents", 0) + meta.get("charging_cost_cents", 0)),
                        "path_nodes": meta.get("path_nodes", []),
                    })
                    chosen_station = st_id
                    station_util[st_id] = station_util.get(st_id, 0) + 1
                elif v in T_nodes and (arc.get("meta") or {}).get("unassigned"):
                    unassigned.append(ev_id)
        if chosen_station is None and ev_id not in unassigned:
            unassigned.append(ev_id)

    total_cost = sum(int(a.get("total_cost_cents", 0)) for a in assignments)
    return {
        "total_cost_cents": total_cost,
        "assignments": assignments,
        "station_utilization": station_util,
        "unassigned_evs": unassigned,
    }