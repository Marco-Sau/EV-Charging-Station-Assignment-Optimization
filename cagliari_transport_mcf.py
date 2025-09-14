# cagliari_transport_mcf.py
"""
Cagliari Transportation (Minimum-Cost Flow: Cycle-Canceling & SSP)
==================================================================

Implements:
- Network/Residual structures (costs in euros, float-safe)
- Reverse BFS (distance labels)
- Preflow-Push (FIFO) to obtain a feasible flow on an augmented network
- Bellman–Ford negative-cycle detection on residual graph
- Cycle-canceling loop until no negative cycle exists
- Successive Shortest Path (SSP) with potentials (robust, progress-guaranteed)

Instance: Cagliari supplies/demands, costs, capacities (transportation network)
The instance below is the MAIN scenario.

Outputs:
- `solve_and_collect(algorithm=...)` returns objective and nonzero flows
- `save_results(...)` writes CSV + JSON to `results/` with seed/algorithm tags
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from sys import maxsize
from queue import Queue
import os
import json
import csv
import math
import time
import random
import heapq
from dataclasses import dataclass

INF_INT = maxsize
INF = 10**18
NO_ID = -1


# ----------------------------- Data Structures -----------------------------

class Node:
    def __init__(self, id: int, supply: int):
        self.id = id
        self.supply = supply  # positive = supply, negative = demand

    def get_supply(self) -> int:
        return self.supply


class Arc:
    def __init__(self, id: int, tail: int, head: int, cost: float = 0.0, capacity: int = INF_INT):
        self.id = id
        self.tail = tail
        self.head = head
        self.cost = float(cost)            # euros
        self.capacity = int(capacity)

    def get_tail(self) -> int: return self.tail
    def get_head(self) -> int: return self.head
    def get_cost(self) -> float: return self.cost
    def get_capacity(self) -> int: return self.capacity
    def set_capacity(self, c: int): self.capacity = int(c)


class Network:
    def __init__(self):
        self.nodes: List[Node] = []
        self.arcs: List[Arc] = []
        self.out_adj_list: List[List[int]] = []
        self.in_adj_list: List[List[int]] = []
        self.flows: List[int] = []

        # residual representation
        self.res_arcs: List[Arc] = []
        self.res_out_adj_list: List[List[int]] = []
        self.res_in_adj_list: List[List[int]] = []

    # ---- nodes ----
    def add_node(self, supply: int) -> int:
        nid = len(self.nodes)
        self.nodes.append(Node(nid, supply))
        self.out_adj_list.append([])
        self.in_adj_list.append([])
        self.res_out_adj_list.append([])
        self.res_in_adj_list.append([])
        return nid

    def add_n_nodes(self, n: int, supplies=None):
        if supplies is None:
            supplies = [0] * n
        for i in range(n):
            self.add_node(int(supplies[i]))

    def get_nodes_ids(self) -> List[int]:
        return list(range(len(self.nodes)))

    def get_number_of_nodes(self) -> int:
        return len(self.nodes)

    def get_node_supply(self, node_id: int) -> int:
        return self.nodes[node_id].get_supply()

    # ---- arcs ----
    @staticmethod
    def get_res_arc_id(arc_id: int) -> int:
        return 2 * arc_id

    @staticmethod
    def get_res_comp_arc_id(res_arc_id: int) -> int:
        return res_arc_id + 1 if res_arc_id % 2 == 0 else res_arc_id - 1

    @staticmethod
    def get_arc_id(res_arc_id: int) -> int:
        return res_arc_id // 2

    def add_arc(self, tail: int, head: int, cost: float = 0.0, capacity: int = INF_INT, flow: int = 0) -> int:
        aid = len(self.arcs)
        self.arcs.append(Arc(aid, tail, head, cost, capacity))
        self.out_adj_list[tail].append(aid)
        self.in_adj_list[head].append(aid)
        self.flows.append(int(flow))

        # residual pair
        rid = self.get_res_arc_id(aid)
        cid = self.get_res_comp_arc_id(rid)

        # forward residual: cap = capacity - flow, cost = +cost
        self.res_arcs.append(Arc(rid, tail, head, cost, max(0, capacity - flow)))
        # backward residual: cap = flow, cost = -cost
        self.res_arcs.append(Arc(cid, head, tail, -cost, max(0, flow)))

        self.res_out_adj_list[tail].append(rid)
        self.res_out_adj_list[head].append(cid)
        self.res_in_adj_list[head].append(rid)
        self.res_in_adj_list[tail].append(cid)
        return aid

    def get_arcs_ids(self): return list(range(len(self.arcs)))
    def get_out_adj_list(self, node_id: int): return self.out_adj_list[node_id]
    def get_in_adj_list(self, node_id: int): return self.in_adj_list[node_id]
    def get_arc_tail(self, arc_id: int): return self.arcs[arc_id].get_tail()
    def get_arc_head(self, arc_id: int): return self.arcs[arc_id].get_head()
    def get_arc_cost(self, arc_id: int): return self.arcs[arc_id].get_cost()
    def get_arc_capacity(self, arc_id: int): return self.arcs[arc_id].get_capacity()
    def get_arc_flow(self, arc_id: int): return self.flows[arc_id]

    def set_arc_flow(self, arc_id: int, flow: int):
        flow = int(flow)
        if self.arcs[arc_id].get_capacity() < flow:
            raise Exception("Capacity bounds violated.")
        self.flows[arc_id] = flow
        rid = self.get_res_arc_id(arc_id)
        cid = self.get_res_comp_arc_id(rid)
        # update residual capacities
        self.res_arcs[rid].set_capacity(self.arcs[arc_id].get_capacity() - flow)
        self.res_arcs[cid].set_capacity(flow)

    # ---- residual getters/pushers ----
    def get_res_arcs_ids(self): return list(range(len(self.res_arcs)))
    def get_res_out_adj_list(self, node_id: int): return self.res_out_adj_list[node_id]
    def get_res_in_adj_list(self, node_id: int): return self.res_in_adj_list[node_id]
    def get_res_arc_tail(self, res_arc_id: int): return self.res_arcs[res_arc_id].get_tail()
    def get_res_arc_head(self, res_arc_id: int): return self.res_arcs[res_arc_id].get_head()
    def get_res_arc_cost(self, res_arc_id: int): return self.res_arcs[res_arc_id].get_cost()
    def get_res_arc_capacity(self, res_arc_id: int): return self.res_arcs[res_arc_id].get_capacity()

    def push_res_arc_delta_flow(self, res_arc_id: int, delta_flow: int):
        delta_flow = int(delta_flow)
        if delta_flow == 0:
            return
        aid = self.get_arc_id(res_arc_id)
        flow = self.flows[aid]
        if res_arc_id % 2 == 0:  # forward residual
            self.set_arc_flow(aid, flow + delta_flow)
        else:  # backward residual
            self.set_arc_flow(aid, flow - delta_flow)


# ----------------------------- Algorithms ----------------------------------

def backward_breadth_first_search(network: Network, target: int) -> Tuple[List[int], List[int]]:
    n = network.get_number_of_nodes()
    marked = [False] * n
    succ = [NO_ID] * n
    dist = [INF_INT] * n
    q: Queue = Queue()
    marked[target] = True
    dist[target] = 0
    q.put(target)
    while not q.empty():
        v = q.get()
        for in_arc in network.get_in_adj_list(v):
            tail = network.get_arc_tail(in_arc)
            if not marked[tail]:
                marked[tail] = True
                succ[tail] = v
                dist[tail] = dist[v] + 1
                q.put(tail)
    return succ, dist


def preflow_push_fifo(network: Network, source: int, sink: int):
    _, distances = backward_breadth_first_search(network, sink)
    distances[source] = network.get_number_of_nodes()
    # zero flows
    for a in network.get_arcs_ids():
        network.set_arc_flow(a, 0)
    # node excesses
    excess = [0] * network.get_number_of_nodes()
    active: Queue = Queue()
    # saturate source out arcs
    for aid in network.get_out_adj_list(source):
        cap = network.get_arc_capacity(aid)
        if cap > 0:
            head = network.get_arc_head(aid)
            network.set_arc_flow(aid, cap)
            excess[source] -= cap
            excess[head] += cap
            if head != sink:
                active.put(head)
    while not active.empty():
        i = active.get()
        while excess[i] > 0:
            min_d = INF_INT
            pushed = False
            for rid in network.get_res_out_adj_list(i):
                j = network.get_res_arc_head(rid)
                rc = network.get_res_arc_capacity(rid)
                if distances[i] == distances[j] + 1 and rc > 0:
                    delta = min(excess[i], rc)
                    network.push_res_arc_delta_flow(rid, delta)
                    excess[i] -= delta
                    excess[j] += delta
                    if j != sink and excess[j] == delta:
                        active.put(j)
                    pushed = True
                    break
                if rc > 0 and distances[j] < min_d:
                    min_d = distances[j]
            if not pushed:
                distances[i] = min_d + 1
                active.put(i)
                break


def bellman_ford_cycle_detection(network: Network) -> List[int]:
    n = network.get_number_of_nodes()
    dist = [float('inf')] * n
    pred_node = [NO_ID] * n
    pred_arc = [NO_ID] * n
    dist[0] = 0.0
    last_mod = NO_ID
    for _ in range(n):
        last_mod = NO_ID
        for rid in network.get_res_arcs_ids():
            u = network.get_res_arc_tail(rid)
            v = network.get_res_arc_head(rid)
            w = network.get_res_arc_cost(rid)
            cap = network.get_res_arc_capacity(rid)
            if cap > 0 and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred_node[v] = u
                pred_arc[v] = rid
                last_mod = v
        if last_mod == NO_ID:
            return []
    # Enter the cycle
    x = last_mod
    for _ in range(n):
        x = pred_node[x]
    cycle = []
    start = x
    while True:
        if x == start and len(cycle) > 1:
            break
        cycle.append(pred_arc[x])
        x = pred_node[x]
    cycle.reverse()
    return cycle


def cycle_canceling(network: Network):
    """Add super source/sink, get feasible flow via preflow-push, then cancel negative cycles."""
    # classify nodes
    supply_nodes, demand_nodes = [], []
    for nid in network.get_nodes_ids():
        s = network.get_node_supply(nid)
        if s > 0: supply_nodes.append(nid)
        elif s < 0: demand_nodes.append(nid)

    # add super source/sink
    s_id = network.add_node(0)
    t_id = network.add_node(0)
    for nid in supply_nodes:
        network.add_arc(s_id, nid, 0.0, network.get_node_supply(nid), 0)
    for nid in demand_nodes:
        network.add_arc(nid, t_id, 0.0, -network.get_node_supply(nid), 0)

    # feasible flow
    preflow_push_fifo(network, s_id, t_id)

    # Keep s_id/t_id to keep residual references valid.
    cycle = bellman_ford_cycle_detection(network)
    while cycle:
        delta = min(network.get_res_arc_capacity(rid) for rid in cycle)
        for rid in cycle:
            network.push_res_arc_delta_flow(rid, delta)
        cycle = bellman_ford_cycle_detection(network)


def successive_shortest_path(network: Network):
    """
    Robust SSP with potentials on the residual graph.
    - Builds super source/sink arcs (like in cycle_canceling), but sends flow via successive shortest paths.
    - Uses reduced costs c_pi = c + pi[u] - pi[v] with nonnegative property to enable Dijkstra.
    - Guards: no-path (infeasible), zero-delta augmentation (degenerate), iteration cap.
    """
    # classify nodes
    supply_nodes, demand_nodes = [], []
    for nid in network.get_nodes_ids():
        s = network.get_node_supply(nid)
        if s > 0: supply_nodes.append(nid)
        elif s < 0: demand_nodes.append(nid)

    # add super source/sink
    s_id = network.add_node(0)
    t_id = network.add_node(0)
    total_supply = 0
    for nid in supply_nodes:
        cap = network.get_node_supply(nid)
        total_supply += cap
        network.add_arc(s_id, nid, 0.0, cap, 0)
    for nid in demand_nodes:
        network.add_arc(nid, t_id, 0.0, -network.get_node_supply(nid), 0)

    n = network.get_number_of_nodes()
    pi = [0.0] * n  # node potentials
    remaining = total_supply
    max_iter = total_supply + n * n  # safety cap

    it = 0
    while remaining > 0:
        it += 1
        if it > max_iter:
            raise RuntimeError("SSP did not converge within iteration cap (probable degeneracy).")

        # Dijkstra on residual with reduced costs (cap > 0)
        dist = [float('inf')] * n
        pred = [NO_ID] * n
        dist[s_id] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, s_id)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            if u == t_id:
                break
            for rid in network.get_res_out_adj_list(u):
                v = network.get_res_arc_head(rid)
                cap = network.get_res_arc_capacity(rid)
                if cap <= 0:
                    continue
                rc = network.get_res_arc_cost(rid) + pi[u] - pi[v]
                # If numerical issues produce a slightly negative rc, clamp to zero for Dijkstra validity
                if rc < 0 and rc > -1e-12:
                    rc = 0.0
                # If rc is truly negative, Dijkstra is unsafe; however with potentials maintained, this should not occur.
                nd = d_u + rc
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = rid
                    heapq.heappush(pq, (nd, v))

        if not math.isfinite(dist[t_id]):
            raise ValueError(f"Infeasible residual (no s→t path) while remaining={remaining}.")

        # Reconstruct residual path
        path: List[int] = []
        x = t_id
        while x != s_id:
            rid = pred[x]
            if rid == NO_ID:
                raise RuntimeError("Path reconstruction failed in SSP.")
            path.append(rid)
            x = network.get_res_arc_tail(rid)
        path.reverse()

        # Bottleneck
        delta = min(network.get_res_arc_capacity(rid) for rid in path)
        if delta <= 0:
            raise RuntimeError("Degenerate augmentation (delta=0) in SSP.")

        # Augment
        for rid in path:
            network.push_res_arc_delta_flow(rid, delta)

        remaining -= delta

        # Update potentials for visited nodes (classic SSP update)
        for v in range(n):
            if math.isfinite(dist[v]):
                pi[v] += dist[v]


# ----------------------------- Instance + Runner ----------------------------

# Supplies and demands (integers), names, capacities (integers)
SUPPLY_NAMES = [
    "Quartu Sant'Elena","Quartucciu","Monserrato","Poetto","Su Planu",
    "La Palma","Bonaria","Pirri","Is Mirrionis","San Michele","Centro Città"
]
SUPPLIES = [50,30,40,60,10,15,25,30,10,35,45]

DEMAND_NAMES = [
    "Cittadella universitaria","Viale Buon Cammino","Piazza Repubblica",
    "Località Marina","Piazza L'Unione Sarda"
]
DEMANDS = [-80,-120,-60,-40,-50]

# Costs were originally in cents in many datasets; we convert to euros (float) here.
# If your source is already in euros, keep as-is.
COSTS_CENTS = [
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
COSTS_EUR: List[List[float]] = [[c / 100.0 for c in row] for row in COSTS_CENTS]

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


def build_transportation_network() -> Network:
    net = Network()
    net.add_n_nodes(len(SUPPLIES), SUPPLIES)
    net.add_n_nodes(len(DEMANDS), DEMANDS)
    ns, nd = len(SUPPLIES), len(DEMANDS)
    for i in range(ns):
        for j in range(nd):
            net.add_arc(i, ns + j, cost=COSTS_EUR[i][j], capacity=CAPACITIES[i][j], flow=0)
    return net


def total_cost(network: Network) -> float:
    """Compute total cost in euros based on original arc costs and current flows."""
    ns, nd = len(SUPPLIES), len(DEMANDS)
    c = 0.0
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            c += COSTS_EUR[i][j] * network.get_arc_flow(aid)
    return c


def solve_and_collect(algorithm: str = "cycle") -> Dict[str, Any]:
    """
    algorithm: "cycle" (Cycle-Canceling) or "ssp" (Successive Shortest Path)
    Returns: {"objective": float, "flows": List[...], "algorithm": str}
    """
    if algorithm not in {"cycle", "ssp"}:
        raise ValueError("algorithm must be one of {'cycle','ssp'}")

    net = build_transportation_network()
    if algorithm == "cycle":
        cycle_canceling(net)
    else:
        successive_shortest_path(net)

    ns, nd = len(SUPPLIES), len(DEMANDS)
    flows = []
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            f = net.get_arc_flow(aid)
            if f > 0:
                flows.append({
                    "from": SUPPLY_NAMES[i],
                    "to": DEMAND_NAMES[j],
                    "flow": int(f),
                    "unit_cost_eur": COSTS_EUR[i][j],
                    "arc_cost_eur": round(COSTS_EUR[i][j] * f, 2),
                })
    return {"objective_eur": round(total_cost(net), 2), "flows": flows, "algorithm": algorithm}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_results(out: Dict[str, Any], seed: int, output_dir: str = "results") -> str:
    """
    Save results to CSV and JSON. Returns the base path prefix used.
    Files created:
      - {output_dir}/cagliari_{algorithm}_seed{seed}_YYYYmmdd_HHMMSS.csv
      - {output_dir}/cagliari_{algorithm}_seed{seed}_YYYYmmdd_HHMMSS.json
    """
    ensure_dir(output_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    algo = out.get("algorithm", "unknown")
    base = os.path.join(output_dir, f"cagliari_{algo}_seed{seed}_{ts}")

    # CSV (flows)
    csv_path = f"{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["from", "to", "flow", "unit_cost_eur", "arc_cost_eur"])
        writer.writeheader()
        for row in out["flows"]:
            writer.writerow(row)

    # JSON (summary)
    summary = {
        "algorithm": algo,
        "seed": seed,
        "objective_eur": out["objective_eur"],
        "num_flows": len(out["flows"]),
        "timestamp": ts,
    }
    json_path = f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "flows": out["flows"]}, f, indent=2)

    return base