"""Turn a pipeline into an execution plan.

The important pass here is *region fusion*: contiguous dataflow nodes are
grouped into one region, which compiles to a single PySpark function and
therefore a single lazy Spark plan. A node's output is written to a table
only where something outside the region needs it (an agent, a router, an
approval, or a region that runs later).

The plan is a graph of *units* - regions and control nodes - which maps
one-to-one onto the generated LangGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Pipeline


@dataclass
class Unit:
    id: str
    kind: str  # "region" | "control"
    nodes: list[str]  # topological order; one entry for control units
    # region: node id -> why its output is written to a table
    materialize: dict[str, str] = field(default_factory=dict)
    # dataset references this unit reads from state (node ids in other units)
    reads: list[str] = field(default_factory=list)
    # how incoming unconditional edges combine: "all" waits for every predecessor
    join: str = "all"

    def to_json(self) -> dict:
        return vars(self)


@dataclass(frozen=True)
class UnitEdge:
    source: str
    target: str
    label: str | None = None
    back: bool = False


@dataclass
class Plan:
    units: list[Unit]
    edges: list[UnitEdge]
    node_unit: dict[str, str]
    problems: list[tuple[str, str, str | None]] = field(default_factory=list)  # (level, message, node)

    def unit(self, unit_id: str) -> Unit:
        return next(u for u in self.units if u.id == unit_id)

    def preds(self, unit_id: str, *, forward_only: bool = True) -> list[UnitEdge]:
        return [e for e in self.edges if e.target == unit_id and not (forward_only and e.back)]

    def succs(self, unit_id: str) -> list[UnitEdge]:
        return [e for e in self.edges if e.source == unit_id]

    def to_json(self) -> dict:
        return {
            "units": [u.to_json() for u in self.units],
            "edges": [vars(e) for e in self.edges],
            "node_unit": self.node_unit,
        }


def find_back_edges(p: Pipeline) -> set[int]:
    """Indices of edges that close a cycle (depth-first, in spec order)."""
    out: dict[str, list[tuple[int, str]]] = {n.id: [] for n in p.nodes}
    for i, e in enumerate(p.edges):
        out[e.source].append((i, e.target))
    state: dict[str, int] = {}  # 1 = on stack, 2 = done
    back: set[int] = set()

    def visit(n: str) -> None:
        state[n] = 1
        for i, t in out[n]:
            if state.get(t) == 1:
                back.add(i)
            elif t not in state:
                visit(t)
        state[n] = 2

    has_incoming = {e.target for e in p.edges}
    for n in sorted(p.nodes, key=lambda n: n.id in has_incoming):  # roots first, stable
        if n.id not in state:
            visit(n.id)
    return back


def topo_order(ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    indeg = {i: 0 for i in ids}
    for _, t in edges:
        indeg[t] += 1
    ready = [i for i in ids if indeg[i] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s, t in edges:
            if s == n:
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
    return order


def build_plan(p: Pipeline) -> Plan:
    ids = [n.id for n in p.nodes]
    kind = {n.id: n.kind for n in p.nodes}
    back = find_back_edges(p)
    fwd = [(e.source, e.target) for i, e in enumerate(p.edges) if i not in back]
    order = topo_order(ids, fwd)
    pos = {n: i for i, n in enumerate(order)}

    # gates[n] = control nodes that must have run before n
    gates: dict[str, frozenset[str]] = {}
    for n in order:
        g: set[str] = set()
        for s, t in fwd:
            if t == n:
                g |= gates[s]
                if kind[s] == "control":
                    g.add(s)
        gates[n] = frozenset(g)

    # ---- region fusion: union dataflow nodes joined by a data edge with identical gates
    parent = {n: n for n in ids}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for s, t in fwd:
        if kind[s] == "dataflow" and kind[t] == "dataflow" and gates[s] == gates[t]:
            parent[find(s)] = find(t)

    groups: dict[str, list[str]] = {}
    for n in order:
        if kind[n] == "dataflow":
            groups.setdefault(find(n), []).append(n)

    units: list[Unit] = []
    node_unit: dict[str, str] = {}
    for members in groups.values():
        u = Unit(id=f"region_{members[-1]}", kind="region", nodes=members)
        units.append(u)
        node_unit.update({m: u.id for m in members})
    for n in order:
        if kind[n] == "control":
            units.append(Unit(id=n, kind="control", nodes=[n]))
            node_unit[n] = n
    units.sort(key=lambda u: pos[u.nodes[0]])
    by_id = {u.id: u for u in units}

    # ---- materialisation points and cross-unit reads
    for e in p.edges:
        if kind[e.source] != "dataflow":
            continue
        su, tu = by_id[node_unit[e.source]], by_id[node_unit[e.target]]
        if su is tu:
            continue
        why = (
            f"read by {p.node(e.target).type} '{e.target}'"
            if kind[e.target] == "control"
            else f"read by '{e.target}', which runs later in {tu.id}"
        )
        su.materialize.setdefault(e.source, why)
        if e.source not in tu.reads:
            tu.reads.append(e.source)
    for n in p.nodes:
        if n.type == "sink":
            by_id[node_unit[n.id]].materialize[n.id] = f"sink to {n.config.get('table', '?')}"

    # ---- unit edges
    seen: set[UnitEdge] = set()
    edges: list[UnitEdge] = []
    for i, e in enumerate(p.edges):
        s, t = node_unit[e.source], node_unit[e.target]
        if s == t:
            continue
        nt = p.node(e.source).node_type
        label = e.label if (nt.free_branches or nt.branches) else None
        ue = UnitEdge(s, t, label, i in back)
        if ue not in seen:
            seen.add(ue)
            edges.append(ue)

    # ---- transitive reduction of unconditional edges (a data dependency that is
    #      already implied by a longer path must not trigger the target early)
    def reachable(src: str, dst: str, skip: UnitEdge) -> bool:
        stack, visited = [src], set()
        while stack:
            n = stack.pop()
            for e in edges:
                if e.source != n or e.back or e == skip or e.target in visited:
                    continue
                if e.target == dst:
                    return True
                visited.add(e.target)
                stack.append(e.target)
        return False

    for e in list(edges):
        if e.label is None and not e.back and reachable(e.source, e.target, e):
            edges.remove(e)

    plan = Plan(units=units, edges=edges, node_unit=node_unit)

    # ---- how fan-in combines: exclusive router branches merge with "any",
    #      independent predecessors with "all"
    ctx: dict[str, dict[str, set[str]]] = {}

    def edge_ctx(e: UnitEdge) -> dict[str, set[str]]:
        c = {r: set(ls) for r, ls in ctx[e.source].items()}
        if e.label is not None:
            c.setdefault(e.source, set()).add(e.label)
        return c

    def exclusive(a: dict, b: dict) -> bool:
        return any(r in b and not (a[r] & b[r]) for r in a)

    for u in units:  # already topological
        pe = plan.preds(u.id)
        cs = [edge_ctx(e) for e in pe]
        merged: dict[str, set[str]] = {}
        for c in cs:
            for r, ls in c.items():
                merged.setdefault(r, set()).update(ls)
        ctx[u.id] = merged
        if len(pe) < 2:
            continue
        pairs = [(i, j) for i in range(len(pe)) for j in range(i + 1, len(pe))]
        excl = [exclusive(cs[i], cs[j]) for i, j in pairs]
        if all(excl):
            u.join = "any"
        elif not any(excl) and all(e.label is None for e in pe):
            u.join = "all"
            if any(c != cs[0] for c in cs):
                plan.problems.append((
                    "warning",
                    f"'{u.id}' waits for {', '.join(e.source for e in pe)}, but some of them sit on a "
                    "router branch. If that branch is not taken this step never runs.",
                    u.nodes[0],
                ))
        else:
            plan.problems.append((
                "error",
                f"'{u.id}' is reached both through a router branch and from an independent step "
                f"({', '.join(e.source for e in pe)}). Connect the independent step before the router "
                "so the order is unambiguous.",
                u.nodes[0],
            ))
    return plan
