# ev_charging_mcf.py
"""
Network flow primitives + Minimum-Cost Flow solvers (Cycle-Canceling and SSP with potentials).

- Network / residual representation
- Preflow-Push (FIFO) to get feasible flow (for cycle-canceling)
- Bellman–Ford negative-cycle detection on residual graph
- Cycle-canceling loop
- Successive Shortest Path (SSP) with potentials (Dijkstra on reduced costs)
- Builders/utilities for transportation networks
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from sys import maxsize
from queue import Queue
import heapq

INF = maxsize
NO_ID = -1


# ----------------------------- Data Structures -----------------------------

class Node:
    def __init__(self, id: int, supply: int):
        self.id = id
        self.supply = supply

    def get_supply(self) -> int:
        return self.supply


class Arc:
    def __init__(self, id: int, tail: int, head: int, cost: float = 0.0, capacity: int = INF):
        self.id = id
        self.tail = tail
        self.head = head
        self.cost = float(cost)
        self.capacity = capacity

    def get_tail(self) -> int: return self.tail
    def get_head(self) -> int: return self.head
    def get_cost(self) -> float: return self.cost
    def get_capacity(self) -> int: return self.capacity
    def set_capacity(self, c: int): self.capacity = c


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
            self.add_node(supplies[i])

    def get_nodes_ids(self): return list(range(len(self.nodes)))
    def get_number_of_nodes(self): return len(self.nodes)
    def get_node_supply(self, node_id: int): return self.nodes[node_id].get_supply()

    # ---- arcs ----
    @staticmethod
    def get_res_arc_id(arc_id: int) -> int: return 2 * arc_id
    @staticmethod
    def get_res_comp_arc_id(res_arc_id: int) -> int: return res_arc_id + 1 if res_arc_id % 2 == 0 else res_arc_id - 1
    @staticmethod
    def get_arc_id(res_arc_id: int) -> int: return res_arc_id // 2

    def add_arc(self, tail: int, head: int, cost: float = 0.0, capacity: int = INF, flow: int = 0) -> int:
        aid = len(self.arcs)
        self.arcs.append(Arc(aid, tail, head, cost, capacity))
        self.out_adj_list[tail].append(aid)
        self.in_adj_list[head].append(aid)
        self.flows.append(flow)
        # residual pair
        rid = self.get_res_arc_id(aid)
        cid = self.get_res_comp_arc_id(rid)
        self.res_arcs.append(Arc(rid, tail, head, cost, capacity - flow))
        self.res_arcs.append(Arc(cid, head, tail, -cost, flow))
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
        if delta_flow <= 0:
            raise RuntimeError("Attempt to push non-positive flow in residual network.")
        aid = self.get_arc_id(res_arc_id)
        flow = self.flows[aid]
        if res_arc_id % 2 == 0:  # forward residual
            self.set_arc_flow(aid, flow + delta_flow)
        else:  # backward residual
            self.set_arc_flow(aid, flow - delta_flow)


# ----------------------------- Algorithms ----------------------------------

def backward_breadth_first_search(network: Network, target: int):
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
            active.put(head)
    while not active.empty():
        i = active.get()
        while excess[i] > 0:
            min_d = INF
            pushed = False
            for rid in network.get_res_out_adj_list(i):
                j = network.get_res_arc_head(rid)
                rc = network.get_res_arc_capacity(rid)
                if rc <= 0:
                    continue
                if distances[i] == distances[j] + 1:
                    delta = min(excess[i], rc)
                    network.push_res_arc_delta_flow(rid, delta)
                    excess[i] -= delta
                    excess[j] += delta
                    if j != sink and excess[j] == delta:
                        active.put(j)
                    pushed = True
                    break
                if distances[j] < min_d:
                    min_d = distances[j]
            if not pushed:
                distances[i] = min_d + 1
                active.put(i)
                break


def bellman_ford_cycle_detection(network: Network):
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

    # keep s_id/t_id to keep residual references valid.

    # cancel negative cycles
    cycle = bellman_ford_cycle_detection(network)
    while cycle:
        delta = min(network.get_res_arc_capacity(rid) for rid in cycle)
        for rid in cycle:
            network.push_res_arc_delta_flow(rid, delta)
        cycle = bellman_ford_cycle_detection(network)


# -------------------- Successive Shortest Path (with potentials) ------------

def _dijkstra_reduced_costs(net: Network, s: int, potentials: List[float]) -> Tuple[List[float], List[int]]:
    n = net.get_number_of_nodes()
    dist = [float('inf')] * n
    prev_res_arc = [NO_ID] * n
    dist[s] = 0.0
    pq: List[Tuple[float, int, int]] = []  # (distance, node_id, tiebreaker)
    heapq.heappush(pq, (0.0, s, -1))
    while pq:
        d_u, u, _ = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        for rid in net.get_res_out_adj_list(u):
            rc = net.get_res_arc_capacity(rid)
            if rc <= 0:
                continue
            v = net.get_res_arc_head(rid)
            cost = net.get_res_arc_cost(rid)
            red_cost = cost + potentials[u] - potentials[v]
            if red_cost < 0:
                # Should not happen once potentials are maintained; safe-guard
                red_cost = 0.0
            nd = d_u + red_cost
            if nd + 1e-15 < dist[v]:
                dist[v] = nd
                prev_res_arc[v] = rid
                # tie-breaker: residual arc id to avoid thrashing on equal costs
                heapq.heappush(pq, (nd, v, rid))
    return dist, prev_res_arc


def ssp_with_potentials(network: Network) -> None:
    """
    Successive Shortest Path with node potentials (AMO).
    Assumes balanced totals and nonnegative original costs on supply->demand arcs.
    """
    # classify nodes
    supply_nodes, demand_nodes = [], []
    for nid in network.get_nodes_ids():
        s = network.get_node_supply(nid)
        if s > 0: supply_nodes.append(nid)
        elif s < 0: demand_nodes.append(nid)

    total_supply = sum(network.get_node_supply(n) for n in supply_nodes)
    total_demand = -sum(network.get_node_supply(n) for n in demand_nodes)
    if total_supply != total_demand:
        raise RuntimeError(f"Unbalanced instance: supply={total_supply}, demand={total_demand}")

    # add super-source and super-sink arcs
    s_id = network.add_node(0)
    t_id = network.add_node(0)
    for nid in supply_nodes:
        cap = network.get_node_supply(nid)
        network.add_arc(s_id, nid, 0.0, cap, 0)
    for nid in demand_nodes:
        cap = -network.get_node_supply(nid)
        network.add_arc(nid, t_id, 0.0, cap, 0)

    # zero all flows (including newly added)
    for a in network.get_arcs_ids():
        network.set_arc_flow(a, 0)

    n = network.get_number_of_nodes()
    potentials = [0.0] * n
    flow_sent = 0

    while flow_sent < total_demand:
        # shortest path in reduced costs from super source to super sink
        dist, prev_res_arc = _dijkstra_reduced_costs(network, s_id, potentials)
        if dist[t_id] == float('inf'):
            raise RuntimeError("No s->t path found while demand remains; instance infeasible.")

        # compute bottleneck delta on the path
        path_rids: List[int] = []
        v = t_id
        while v != s_id:
            rid = prev_res_arc[v]
            if rid == NO_ID:
                raise RuntimeError("Path reconstruction failed.")
            path_rids.append(rid)
            v = network.get_res_arc_tail(rid)
        path_rids.reverse()

        delta = min(network.get_res_arc_capacity(rid) for rid in path_rids)
        if delta <= 0:
            raise RuntimeError("Degenerate augmentation (delta=0).")
        # push flow
        for rid in path_rids:
            network.push_res_arc_delta_flow(rid, delta)
        flow_sent += delta

        # update potentials: pi'(v) = pi(v) + dist(v) for reachable v
        for v_id in range(n):
            if dist[v_id] < float('inf'):
                potentials[v_id] += dist[v_id]
        # loop continues until all demand is satisfied


# ----------------------------- Builders/Utils -------------------------------

def build_transportation_network(
    supplies: List[int],
    demands: List[int],
    costs: List[List[float]],
    capacities: List[List[int]],
) -> Network:
    """
    Build a transportation network:
    - supply nodes (positive supply)
    - demand nodes (negative supply)
    - arcs supply_i -> demand_j with capacity and cost (euros)
    """
    ns, nd = len(supplies), len(demands)
    if any(d >= 0 for d in demands):
        raise ValueError("Demands must be negative integers (e.g., -80).")

    net = Network()
    net.add_n_nodes(ns, supplies)
    net.add_n_nodes(nd, demands)

    for i in range(ns):
        for j in range(nd):
            # supply node i -> demand node (ns + j)
            net.add_arc(i, ns + j, cost=costs[i][j], capacity=capacities[i][j], flow=0)
    return net


def total_cost(network: Network, ns: int, nd: int, costs: List[List[float]]) -> float:
    c = 0.0
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            c += costs[i][j] * network.get_arc_flow(aid)
    return c


def extract_nonzero_flows(
    network: Network,
    ns: int,
    nd: int,
    supply_names: List[str],
    demand_names: List[str],
    costs: List[List[float]],
) -> List[Dict]:
    flows = []
    for i in range(ns):
        for j in range(nd):
            aid = i * nd + j
            f = network.get_arc_flow(aid)
            if f > 0:
                flows.append({
                    "from": supply_names[i],
                    "to": demand_names[j],
                    "flow": f,
                    "unit_cost_eur": costs[i][j],
                    "arc_cost_eur": costs[i][j] * f
                })
    return flows