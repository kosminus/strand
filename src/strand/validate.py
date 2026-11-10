"""Checks a pipeline before it is compiled. The canvas shows these inline."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .plan import Plan, build_plan, find_back_edges
from .spec import ID_RE, NODE_TYPES, RESERVED_IDS, Pipeline

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TABLE_RE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*){0,2}$")


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str
    node: str | None = None

    def to_json(self) -> dict:
        return vars(self)


def validate(p: Pipeline) -> tuple[list[Issue], Plan | None]:
    issues: list[Issue] = []
    err = lambda msg, node=None: issues.append(Issue("error", msg, node))  # noqa: E731
    warn = lambda msg, node=None: issues.append(Issue("warning", msg, node))  # noqa: E731

    if not NAME_RE.match(p.name or ""):
        err("Pipeline name must be lowercase letters, digits and underscores, starting with a letter.")
    if not p.nodes:
        err("The pipeline has no nodes.")

    # ---- nodes
    seen: set[str] = set()
    for n in p.nodes:
        if n.id in seen:
            err(f"Duplicate node id '{n.id}'.", n.id)
        seen.add(n.id)
        if not ID_RE.match(n.id) or n.id in RESERVED_IDS or n.id.startswith("region_"):
            err(f"'{n.id}' is not a usable node id (lowercase identifier, not a reserved word).", n.id)
        t = NODE_TYPES.get(n.type)
        if t is None:
            err(f"Unknown node type '{n.type}'.", n.id)
            continue
        for f in t.fields:
            if f.required and n.get(f.name) in (None, "", []):
                err(f"{t.label}: '{f.label}' is required.", n.id)
        _check_config(n, err, warn)

    if any(i.level == "error" for i in issues):
        return issues, None

    # ---- edges
    for e in p.edges:
        if e.source not in seen or e.target not in seen:
            err(f"Edge {e.source} -> {e.target} refers to a node that does not exist.")
            continue
        if e.source == e.target:
            err("A node cannot connect to itself.", e.source)
    if any(i.level == "error" for i in issues):
        return issues, None

    for n in p.nodes:
        t = n.node_type
        data_in = p.data_inputs(n.id)
        if t.kind == "dataflow":
            ports = {port.name: port for port in t.inputs}
            used: dict[str, int] = {}
            for e in data_in:
                if p.node(e.source).type == "sink":
                    err(f"A sink has no data output: '{e.source}' cannot feed '{n.id}'.", n.id)
                port = e.port or (t.inputs[0].name if len(t.inputs) == 1 else None)
                if not t.inputs:
                    err(f"{t.label} does not take a data input (from '{e.source}').", n.id)
                elif port not in ports:
                    err(f"{t.label} needs each input on one of: {', '.join(ports)}.", n.id)
                else:
                    used[port] = used.get(port, 0) + 1
            for port in t.inputs:
                c = used.get(port.name, 0)
                if port.required and c == 0:
                    err(f"{t.label} is missing its '{port.name}' input.", n.id)
                if port.max is not None and c > port.max:
                    err(f"{t.label} takes one '{port.name}' input, found {c}.", n.id)
            if n.type != "sink" and not p.outgoing(n.id):
                warn("Output is not used by anything.", n.id)

        out = p.outgoing(n.id)
        if t.free_branches or t.branches:
            for e in out:
                if not e.label:
                    err(f"Every connection leaving {t.label.lower()} '{n.id}' needs a branch label.", n.id)
                elif t.branches and e.label not in t.branches:
                    err(f"Branches of '{n.id}' must be one of: {', '.join(t.branches)}.", n.id)
            if not out:
                warn(f"{t.label} has no outgoing branches.", n.id)
        if n.type == "agent" and not data_in and not p.incoming(n.id):
            warn("Agent has no inputs; it will only see its task text.", n.id)

    # ---- loops must be bounded and closed by a router
    for i in find_back_edges(p):
        e = p.edges[i]
        if p.node(e.source).type != "router":
            err(
                f"The connection {e.source} -> {e.target} closes a loop. Loops must be closed by a "
                "router, which bounds the number of iterations.",
                e.source,
            )
    if any(i.level == "error" for i in issues):
        return issues, None

    plan = build_plan(p)
    for level, msg, node in plan.problems:
        issues.append(Issue(level, msg, node))
    return issues, plan


def _check_config(n, err, warn) -> None:
    c = n.get
    if n.type == "source":
        if not c("table") and not c("path"):
            err("Source needs a table or a path.", n.id)
        if c("table") and not TABLE_RE.match(c("table")):
            err(f"'{c('table')}' is not a valid table name.", n.id)
    elif n.type == "sink":
        if c("table") and not TABLE_RE.match(c("table")):
            err(f"'{c('table')}' is not a valid table name.", n.id)
        if c("mode") == "merge" and not c("merge_keys"):
            err("Merge mode needs merge keys.", n.id)
    elif n.type == "python" and c("code"):
        _check_python(c("code"), "exec", n.id, err)
        if re.search(r"\.rdd\b|sparkContext|\bsc\.", c("code")):
            warn("RDDs and sparkContext are not available on serverless or shared clusters.", n.id)
    elif n.type == "router" and c("expr"):
        _check_python(c("expr"), "eval", n.id, err)
    elif n.type == "llm_map":
        if c("prompt") and not re.search(r"\{\w+\}", c("prompt")):
            warn("The prompt uses no {column}; every row will get the same prompt.", n.id)
    elif n.type == "agent":
        if c("include_sample"):
            warn("Sample rows will be sent to the model and stored in checkpoints and traces.", n.id)


def _check_python(code: str, mode: str, node: str, err) -> None:
    try:
        if mode == "eval":
            ast.parse(code.strip(), mode="eval")
        else:
            ast.parse("def _f(df, spark):\n" + "\n".join("    " + l for l in code.splitlines()))
    except SyntaxError as e:
        err(f"Python syntax error: {e.msg} (line {e.lineno}).", node)
