"""The graph spec: what the canvas edits and the compiler reads.

A pipeline is a list of nodes and edges. Every node type is either

* ``dataflow`` - produces a lazy Spark DataFrame, runs on Spark compute, or
* ``control``  - runs on the driver as a LangGraph node and only ever sees
  dataset *references* (table name, schema, metrics), never rows.

The kind of an edge follows from its source node: an edge leaving a dataflow
node carries data, an edge leaving a control node only orders execution.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

Kind = Literal["dataflow", "control"]

ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
RESERVED_IDS = {"spark", "ctx", "out", "datasets", "outputs", "state", "rt", "transforms", "df"} | set(
    keyword.kwlist
)


# --------------------------------------------------------------------------- node types


@dataclass(frozen=True)
class FieldDef:
    name: str
    label: str
    type: Literal["text", "textarea", "code", "select", "list", "number", "bool"] = "text"
    required: bool = False
    default: Any = None
    options: tuple[str, ...] = ()
    help: str = ""
    language: str = ""  # for "code": sql | python


@dataclass(frozen=True)
class Port:
    name: str
    max: int | None = 1  # None = unlimited
    required: bool = True


@dataclass(frozen=True)
class NodeType:
    type: str
    kind: Kind
    label: str
    description: str
    fields: tuple[FieldDef, ...]
    # Data input ports. Control nodes accept any number of dataset references.
    inputs: tuple[Port, ...] = ()
    # Fixed branch labels for outgoing edges (approval). Routers use free labels.
    branches: tuple[str, ...] = ()
    free_branches: bool = False

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "fields": [vars(f) | {"options": list(f.options)} for f in self.fields],
            "inputs": [vars(p) for p in self.inputs],
            "branches": list(self.branches),
            "free_branches": self.free_branches,
        }


F = FieldDef

NODE_TYPES: dict[str, NodeType] = {
    t.type: t
    for t in [
        NodeType(
            "source", "dataflow", "Source", "Read a Unity Catalog table or a path in a volume.",
            (
                F("table", "Table", help="catalog.schema.table. Leave empty to read from a path."),
                F("path", "Path", help="/Volumes/... used when no table is given."),
                F("format", "Format", "select", default="delta", options=("delta", "parquet", "csv", "json")),
            ),
        ),
        NodeType(
            "filter", "dataflow", "Filter", "Keep rows matching a SQL condition.",
            (F("condition", "Condition", "code", required=True, language="sql"),),
            inputs=(Port("in"),),
        ),
        NodeType(
            "sql", "dataflow", "SQL", "Spark SQL over the inputs. Refer to an input as {node_id}.",
            (F("query", "Query", "code", required=True, language="sql",
               help="Example: SELECT status, count(*) AS n FROM {tickets} GROUP BY status"),),
            inputs=(Port("in", max=None),),
        ),
        NodeType(
            "python", "dataflow", "Python", "PySpark code. Receives `df` and `spark`, must return a DataFrame.",
            (F("code", "Function body", "code", required=True, language="python",
               default="return df", help="DataFrame API only (no RDDs): this must run on Spark Connect."),),
            inputs=(Port("in"),),
        ),
        NodeType(
            "join", "dataflow", "Join", "Join two inputs on shared columns.",
            (
                F("on", "Join columns", "list", required=True, help="Column names present on both sides."),
                F("how", "Join type", "select", default="inner",
                  options=("inner", "left", "right", "full", "left_semi", "left_anti")),
            ),
            inputs=(Port("left"), Port("right")),
        ),
        NodeType(
            "llm_map", "dataflow", "LLM map", "Call a model on every row as a Spark batch (ai_query).",
            (
                F("endpoint", "Serving endpoint", required=True, default="databricks-claude-sonnet-4"),
                F("prompt", "Prompt", "textarea", required=True,
                  help="Use {column} to insert a column value. Example: Classify this ticket: {body}"),
                F("output_column", "Output column", required=True, default="llm_output"),
            ),
            inputs=(Port("in"),),
        ),
        NodeType(
            "sink", "dataflow", "Delta sink", "Write the input to a table.",
            (
                F("table", "Table", required=True, help="catalog.schema.table"),
                F("mode", "Mode", "select", default="overwrite", options=("overwrite", "append", "merge")),
                F("merge_keys", "Merge keys", "list", help="Required for merge mode."),
            ),
            inputs=(Port("in"),),
        ),
        NodeType(
            "agent", "control", "Agent", "An LLM agent that loops over tools. Sees dataset references, not rows.",
            (
                F("endpoint", "Serving endpoint", required=True, default="databricks-claude-sonnet-4"),
                F("system_prompt", "System prompt", "textarea", default="You are a careful data operations agent."),
                F("instructions", "Task", "textarea", required=True,
                  help="What the agent should do. Connected datasets are summarised for it automatically."),
                F("uc_functions", "Unity Catalog functions", "list", help="catalog.schema.function, used as tools."),
                F("vector_indexes", "Vector Search indexes", "list", help="catalog.schema.index, used as retrievers."),
                F("choices", "Decision choices", "list",
                  help="If set, the agent must end with one of these. Route on outputs['<id>']['choice']."),
                F("include_sample", "Show sample rows to the model", "bool", default=False,
                  help="Off by default: samples end up in prompts, checkpoints and traces."),
                F("max_steps", "Max tool steps", "number", default=8),
            ),
        ),
        NodeType(
            "router", "control", "Router", "Pick a branch from a Python expression over `datasets` and `outputs`.",
            (
                F("expr", "Expression", "code", required=True, language="python",
                  help="Must evaluate to a branch label, or a bool (branches 'true' / 'false')."),
                F("max_iterations", "Max loop iterations", "number", default=3,
                  help="If this router closes a loop, it stops after this many passes."),
            ),
            free_branches=True,
        ),
        NodeType(
            "approval", "control", "Human approval", "Pause the run until someone approves or rejects.",
            (F("message", "Message", "textarea", required=True, default="Approve this step?"),),
            branches=("approved", "rejected"),
        ),
    ]
}


# --------------------------------------------------------------------------- models


class UI(BaseModel):
    x: float = 0
    y: float = 0


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    ui: UI | None = None

    @property
    def node_type(self) -> NodeType | None:
        return NODE_TYPES.get(self.type)

    @property
    def kind(self) -> Kind | None:
        t = self.node_type
        return t.kind if t else None

    def get(self, name: str) -> Any:
        """Config value with the type's default applied."""
        value = self.config.get(name)
        if value in (None, "", []):
            t = self.node_type
            for f in t.fields if t else ():
                if f.name == name:
                    return [] if f.type == "list" and f.default is None else f.default
        if isinstance(value, str):
            for f in self.node_type.fields if self.node_type else ():
                if f.name == name and f.type == "list":
                    return [v.strip() for v in re.split(r"[,\n]", value) if v.strip()]
        return value


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    source: str = Field(validation_alias=AliasChoices("source", "from"))
    target: str = Field(validation_alias=AliasChoices("target", "to"))
    port: str | None = None  # target input port (join: left / right)
    label: str | None = None  # branch label when the source is a router or approval


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["postgres", "sqlite", "memory"] = "postgres"
    # postgres (Lakebase): "scope/key" of a secret holding the connection string.
    dsn_secret: str | None = None
    path: str | None = None  # sqlite


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    catalog: str = "main"
    staging_schema: str = Field(default="strand_staging", validation_alias=AliasChoices("staging_schema", "schema"))
    table_format: str = "delta"
    sample_rows: int = 5
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)


class Pipeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    settings: Settings = Field(default_factory=Settings)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def is_data_edge(self, e: Edge) -> bool:
        """True when the edge carries a dataset (its source is a dataflow node)."""
        return self.node(e.source).kind == "dataflow"

    def data_inputs(self, node_id: str) -> list[Edge]:
        return [e for e in self.incoming(node_id) if self.is_data_edge(e)]


def load(text: str) -> Pipeline:
    return Pipeline.model_validate(yaml.safe_load(text))


def dump(p: Pipeline) -> str:
    data = p.model_dump(exclude_none=True, exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
