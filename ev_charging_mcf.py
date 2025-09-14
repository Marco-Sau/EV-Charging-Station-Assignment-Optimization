# ev_charging_mcf.py
# Core network/min-cost-flow machinery (residual network, Cycle-Canceling, SSP)
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from queue import Queue
import heapq
import math
import json
import csv
import os
from datetime import datetime, timezone
import random

INF = 10**18
NO_ID = -1


# ----------------------------- Data Structures -----------------------------

@dataclass
class Node:
    id: int
    supply: int  # positive = supply, negative = demand, zero = transshipment


@dataclass
class Arc:
    id: int
    tail: int
    head: int
    cost: float = 0.0   # *** in EUROS ***
    capacity: int = INF


class Network:
    """
    Directed network with parallel residual representation.
    Costs are in EUROS (float). Capacities/flows are integers.
    """
    def __init__(self):
        self.nodes: List[Node] = []
        self.arcs: List[Arc] = []
        self.out_adj_list: List[List[int]] = []
        self.in_adj_list: List[List[int]] = []
        self.flows: List[int] = []

        # residual graph
        self.res_arcs: List[Arc] = []  # forward/backward residual arcs
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

    def add_n_nodes(self, n: int, supplies: Optional[List[int]] = None):
        if supplies is None:
            supplies = [0] * n
        for s in supplies:
            self.add_node(s)

    def get_nodes_ids(self) -> List[int]:
        return list(range(len(self.nodes)))

    def get_number_of_nodes(self) -> int:
        return len(self.nodes)

    def get_node_supply(self, node_id: int) -> int:
        return self.nodes[node_id].supply

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

    def add_arc(
        self,
        tail: int,
        head: int,
        cost: float = 0.0,
        capacity: int = INF,
        flow: int = 0,
    ) -> int:
        aid = len(self.arcs)
        self.arcs.append(Arc(aid, tail, head, cost, capacity))
        self.out_adj_list[tail].append(aid)
        self.in_adj_list[head].append(aid)
        self.flows.append(flow)

        # residual pair
        rid = self.get_res_arc_id(aid)
        cid = self.get_res_comp_arc_id(rid)
        # forward residual: capacity - flow
        self.res_arcs.append(Arc(rid, tail, head, cost, max(0, capacity - flow)))
        # backward residual: -cost, flow units
        self.res_arcs.append(Arc(cid, head, tail, -cost, max(0, flow)))
        self.res_out_adj_list[tail].append(rid)
        self.res_out_adj_list[head].append(cid)
        self.res_in_adj_list[head].append(rid)
        self.res_in_adj_list[tail].append(cid)
        return aid

    # base arc getters
    def get_arcs_ids(self) -> List[int]:
        return list(range(len(self.arcs)))

    def get_out_adj_list(self, node_id: int) -> List[int]:
        return self.out_adj_list[node_id]

    def get_in_adj_list(self, node_id: int) -> List[int]:
        return self.in_adj_list[node_id]

    def get_arc_tail(self, arc_id: int) -> int:
        return self.arcs[arc_id].tail

    def get_arc_head(self, arc_id: int) -> int:
        return self.arcs[arc_id].head

    def get_arc_cost(self, arc_id: int) -> float:
        return self.arcs[arc_id].cost

    def get_arc_capacity(self, arc_id: int) -> int:
        return self.arcs[arc_id].capacity

    def get_arc_flow(self, arc_id: int) -> int:
        return self.flows[arc_id]

    def set_arc_flow(self, arc_id: int, flow: int):
        if flow < 0 or flow > self.arcs[arc_id].capacity:
            raise RuntimeError("Capacity bounds violated when setting flow.")
        self.flows[arc_id] = flow

        rid = self.get_res_arc_id(arc_id)
        cid = self.get_res_comp_arc_id(rid)
        # update residual capacities
        self.res_arcs[rid].capacity = self.arcs[arc_id].capacity - flow
        self.res_arcs[cid].capacity = flow

    # ---- residual getters/pushers ----
    def get_res_arcs_ids(self) -> List[int]:
        return list(range(len(self.res_arcs)))

    def get_res_out_adj_list(self, node_id: int) -> List[int]:
        return self.res_out_adj_list[node_id]

    def get_res_in_adj_list(self, node_id: int) -> List[int]:
        return self.res_in_adj_list[node_id]

    def get_res_arc_tail(self, res_arc_id: int) -> int:
        return self.res_arcs[res_arc_id].tail

    def get_res_arc_head(self, res_arc_id: int) -> int:
        return self.res_arcs[res_arc_id].head

    def get_res_arc_cost(self, res_arc_id: int) -> float:
        return self.res_arcs[res_arc_id].cost

    def get_res_arc_capacity(self, res_arc_id: int) -> int:
        return self.res_arcs[res_arc_id].capacity

    def push_res_arc_delta_flow(self, res_arc_id: int, delta_flow: int):
        """Push delta along residual arc (forward if even id, backward if odd)."""
        if delta_flow <= 0:
            return
        if self.res_arcs[res_arc_id].capacity < delta_flow:
            raise RuntimeError("Residual capacity insufficient for requested push.")
        base_arc_id = self.get_arc_id(res_arc_id)
        current = self.get_arc_flow(base_arc_id)
        if res_arc_id % 2 == 0:  # forward residual
            self.set_arc_flow(base_arc_id, current + delta_flow)
        else:  # backward residual
            self.set_arc_flow(base_arc_id, current - delta_flow)


# ----------------------------- Helpers / I/O -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts_eu() -> str:
    # Europe/Rome formatting as naive timestamp label (no tz needed for filenames)
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def save_solution_json_csv(
    out_dir: str,
    algo_tag: str,
    seed: Optional[int],
    result: Dict[str, Any],
):
    ensure_dir(out_dir)
    ts = now_ts_eu()
    seed_tag = f"seed{seed}" if seed is not None else "seedNA"
    base = os.path.join(out_dir, f"solution_{algo_tag}_{seed_tag}_{ts}")
    json_path = base + ".json"
    csv_path = base + ".csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    flows = result.get("flows", [])
    if flows:
        fieldnames = list(flows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flows)

    print(f"[SAVE] JSON: {json_path}")
    if flows:
        print(f"[SAVE] CSV : {csv_path}")


def set_global_seed(seed: Optional[int]):
    if seed is not None:
        random.seed(seed)


# ----------------------------- Algorithms ----------------------------------

def backward_breadth_first_search(network: Network, target: int) -> Tuple[List[int], List[int]]:
    n = network.get_number_of_nodes()
    marked = [False] * n
    succ = [NO_ID] * n
    dist = [INF] * n
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
    """Maximum flow (feasibility) using FIFO Preflow-Push."""
    _, distances = backward_breadth_first_search(network, sink)
    distances[source] = network.get_number_of_nodes()

    # zero flows
    for a in network.get_arcs_ids():
        network.set_arc_flow(a, 0)

    # node excesses
    excess = [0] * network.get_number_of_nodes()
    active: Queue = Queue()

    # saturate source outgoing arcs
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
            min_d = INF
            pushed = False
            # optional randomized iteration for determinism w.r.t. seed
            out_rids = list(network.get_res_out_adj_list(i))
            # small deterministic shuffle controlled by random; harmless for correctness
            if len(out_rids) > 1:
                out_rids.sort()  # stable order first
            for rid in out_rids:
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
                # relabel
                if min_d >= INF // 2:
                    break
                distances[i] = min_d + 1
                active.put(i)
                break


def bellman_ford_cycle_detection(network: Network) -> List[int]:
    """Detect a negative cycle in the residual graph; return cycle residual arc IDs (may be empty)."""
    n = network.get_number_of_nodes()
    INFLL = 10**18
    dist = [INFLL] * n
    pred_node = [NO_ID] * n
    pred_arc = [NO_ID] * n
    dist[0] = 0
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

    # enter the cycle
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
    """
    Transportation-style Cycle-Canceling:
    1) Add super source/sink linked to supplies/demands to obtain feasible flow by max-flow.
    2) Keep them in the graph (simplifies residual bookkeeping).
    3) Cancel negative cycles until none remain.
    """
    # classify nodes
    supply_nodes, demand_nodes = [], []
    for nid in network.get_nodes_ids():
        s = network.get_node_supply(nid)
        if s > 0:
            supply_nodes.append(nid)
        elif s < 0:
            demand_nodes.append(nid)

    total_supply = sum(network.get_node_supply(n) for n in supply_nodes)
    total_demand = -sum(network.get_node_supply(n) for n in demand_nodes)
    if total_supply != total_demand:
        raise ValueError("Unbalanced instance: total supply must equal total demand.")

    # add super source/sink
    s_id = network.add_node(0)
    t_id = network.add_node(0)
    for nid in supply_nodes:
        network.add_arc(s_id, nid, 0.0, network.get_node_supply(nid), 0)
    for nid in demand_nodes:
        network.add_arc(nid, t_id, 0.0, -network.get_node_supply(nid), 0)

    # initial feasible flow
    preflow_push_fifo(network, s_id, t_id)

    # negative-cycle canceling
    while True:
        cycle = bellman_ford_cycle_detection(network)
        if not cycle:
            break
        delta = min(network.get_res_arc_capacity(rid) for rid in cycle)
        if delta <= 0:
            break
        for rid in cycle:
            network.push_res_arc_delta_flow(rid, delta)


# ------------------ Successive Shortest Path (with potentials) ------------------

def _dijkstra_reduced_costs(
    network: Network,
    source: int,
    sink: int,
    pi: List[float]
) -> Tuple[List[int], List[float]]:
    """Dijkstra on residual graph using reduced costs c_pi(u,v) = c(u,v) + pi[u] - pi[v] (must be >= 0)."""
    n = network.get_number_of_nodes()
    dist = [math.inf] * n
    parent_res_arc = [NO_ID] * n
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == sink:
            break
        for rid in network.get_res_out_adj_list(u):
            if network.get_res_arc_capacity(rid) <= 0:
                continue
            v = network.get_res_arc_head(rid)
            rc = network.get_res_arc_cost(rid)
            red = rc + pi[u] - pi[v]
            if red < 0:
                # Due to floating/previous updates, minimal negative can appear; clamp to 0
                red = max(red, 0.0)
            nd = d + red
            if nd < dist[v] - 1e-15:
                dist[v] = nd
                parent_res_arc[v] = rid
                heapq.heappush(pq, (nd, v))

    return parent_res_arc, dist


def successive_shortest_path(network: Network):
    """
    SSP for balanced transportation:
    - Build super source/sink (same as Cycle-Canceling),
    - Iteratively send one shortest augmenting path worth of flow using reduced costs.
    """
    # classify nodes
    supply_nodes, demand_nodes = [], []
    for nid in network.get_nodes_ids():
        s = network.get_node_supply(nid)
        if s > 0:
            supply_nodes.append(nid)
        elif s < 0:
            demand_nodes.append(nid)

    total_supply = sum(network.get_node_supply(n) for n in supply_nodes)
    total_demand = -sum(network.get_node_supply(n) for n in demand_nodes)
    if total_supply != total_demand:
        raise ValueError("Unbalanced instance: total supply must equal total demand.")

    # add super source/sink
    s_id = network.add_node(0)
    t_id = network.add_node(0)
    for nid in supply_nodes:
        network.add_arc(s_id, nid, 0.0, network.get_node_supply(nid), 0)
    for nid in demand_nodes:
        network.add_arc(nid, t_id, 0.0, -network.get_node_supply(nid), 0)

    # node potentials (start at 0)
    n = network.get_number_of_nodes()
    pi = [0.0] * n

    remaining = total_supply
    # Loop guard to prevent infinite loops in degenerate cases
    max_iter = 10_000

    it = 0
    while remaining > 0 and it < max_iter:
        it += 1
        parent_res_arc, dist = _dijkstra_reduced_costs(network, s_id, t_id, pi)
        if not math.isfinite(dist[t_id]):
            # No more s->t path; if remaining>0 this is infeasible (shouldn't happen in balanced TP)
            break

        # Compute bottleneck on the residual path from t to s
        path_rids: List[int] = []
        v = t_id
        bottleneck = INF
        while v != s_id:
            rid = parent_res_arc[v]
            if rid == NO_ID:
                bottleneck = 0
                break
            path_rids.append(rid)
            cap = network.get_res_arc_capacity(rid)
            if cap < bottleneck:
                bottleneck = cap
            v = network.get_res_arc_tail(rid)

        if bottleneck <= 0 or bottleneck == INF:
            break

        # Augment by delta = min(bottleneck, remaining)
        delta = min(bottleneck, remaining)
        for rid in path_rids:
            network.push_res_arc_delta_flow(rid, delta)

        remaining -= delta

        # Update potentials: pi[v] += dist[v] (classic SSP)
        for v_id in range(n):
            if math.isfinite(dist[v_id]):
                pi[v_id] += dist[v_id]

    if remaining != 0:
        # Not fatal; but indicates disconnection (shouldn't for balanced TP with finite caps)
        print(f"[SSP] Warning: remaining flow {remaining} not shipped.")


# ----------------------------- Analysis ----------------------------------

def analyze_transportation_solution(
    network: Network,
    ns: int,
    nd: int,
    supply_names: List[str],
    demand_names: List[str],
    costs_eur: List[List[float]]
) -> Dict[str, Any]:
    """
    Extract nonzero flows on original supply->demand arcs (first ns*nd arcs).
    Return dict with objective_eur (float) and detailed flows.
    """
    flows = []
    objective = 0.0
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            f = network.get_arc_flow(aid)
            if f > 0:
                unit = costs_eur[i][j]
                arc_cost = unit * f
                objective += arc_cost
                flows.append({
                    "from": supply_names[i],
                    "to": demand_names[j],
                    "flow": int(f),
                    "unit_cost_eur": round(unit, 2),
                    "arc_cost_eur": round(arc_cost, 2),
                })
    result = {
        "objective_eur": round(objective, 2),
        "flows": flows,
    }
    return result