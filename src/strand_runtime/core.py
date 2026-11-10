"""Run-time support for generated strand pipelines.

Everything here is deliberately small: the generated code is the product,
this package only holds what would otherwise be repeated in every pipeline.

The one rule it enforces: LangGraph state holds *references* to datasets
(table name, columns, row count, version). DataFrames never leave a region.
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, Any, Callable, TypedDict


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    datasets: Annotated[dict[str, dict], _merge]  # node id -> dataset reference
    outputs: Annotated[dict[str, Any], _merge]  # node id -> agent / router / approval result
    visits: Annotated[dict[str, int], _merge]  # unit id -> times executed (loops)


def _ident(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))


class RunContext:
    def __init__(
        self,
        spark,
        *,
        pipeline: str,
        run_id: str,
        settings: dict,
        chat_model: Callable[[str], Any] | None = None,
        tools: Callable[[list[str], list[str], list[str]], list] | None = None,
    ):
        self.spark = spark
        self.pipeline = pipeline
        self.run_id = _ident(run_id)
        self.settings = settings
        self.iteration = 1  # set by run_region; makes appends idempotent per loop pass
        self._chat_model = chat_model
        self._tools = tools
        self._events: list[tuple] = []
        self._schema_ready = False

    # ------------------------------------------------------------------ naming
    @property
    def staging(self) -> str:
        cat, sch = self.settings.get("catalog"), self.settings.get("staging_schema", "strand_staging")
        return f"{cat}.{sch}" if cat else sch

    @property
    def fmt(self) -> str:
        return self.settings.get("table_format", "delta")

    def staging_table(self, node_id: str) -> str:
        # Run-scoped: a retry or a resume overwrites its own snapshot and nothing else,
        # and a paused run does not depend on Delta time travel surviving VACUUM.
        return f"{self.staging}.{self.pipeline}__{node_id}__{self.run_id}"

    def _ensure_schema(self) -> None:
        if not self._schema_ready:
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.staging}")
            self._schema_ready = True

    # ------------------------------------------------------------------ datasets
    def describe(self, node_id: str, table: str) -> dict:
        df = self.spark.read.table(table)
        version = None
        if self.fmt == "delta":
            try:
                version = self.spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").collect()[0]["version"]
            except Exception:
                pass
        return {
            "node": node_id,
            "table": table,
            "columns": [{"name": f.name, "type": f.dataType.simpleString()} for f in df.schema.fields],
            "row_count": df.count(),
            "version": version,
        }

    def materialize(self, node_id: str, df) -> dict:
        """Write a region output to its run-scoped staging table and return a reference."""
        self._ensure_schema()
        table = self.staging_table(node_id)
        df.write.format(self.fmt).mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        ref = self.describe(node_id, table)
        self.log_event("materialize", node_id, ref)
        return ref

    def read(self, ref: dict):
        return self.spark.read.table(ref["table"])

    def sample(self, ref: dict, n: int | None = None) -> list[dict]:
        n = n if n is not None else int(self.settings.get("sample_rows", 5))
        rows = self.read(ref).limit(n).collect()
        return json.loads(json.dumps([r.asDict(recursive=True) for r in rows], default=str))

    def write_sink(self, node_id: str, df, *, table: str, mode: str = "overwrite", merge_keys: list[str] | None = None) -> dict:
        """Write to a user table so that re-running the same step does not duplicate rows."""
        exists = self.spark.catalog.tableExists(table)
        if mode == "merge" and exists:
            cond = " AND ".join(f"t.`{k}` = s.`{k}`" for k in merge_keys or [])
            from delta.tables import DeltaTable

            (DeltaTable.forName(self.spark, table).alias("t").merge(df.alias("s"), cond)
             .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
        else:
            writer = df.write.format(self.fmt).mode("append" if mode == "append" else "overwrite")
            if mode == "append" and self.fmt == "delta":
                # Delta skips a write whose (txnAppId, txnVersion) it has already committed,
                # so a step that is re-run after a crash or a resume appends once.
                writer = writer.option("txnAppId", f"strand:{self.pipeline}:{self.run_id}:{node_id}")
                writer = writer.option("txnVersion", str(self.iteration))
            writer.saveAsTable(table)
        ref = self.describe(node_id, table)
        self.log_event("sink", node_id, {**ref, "mode": mode})
        return ref

    # ------------------------------------------------------------------ models and tools
    def chat_model(self, endpoint: str):
        if self._chat_model:
            return self._chat_model(endpoint)
        from databricks_langchain import ChatDatabricks

        return ChatDatabricks(endpoint=endpoint)

    def tools(self, uc_functions: list[str], vector_indexes: list[str], mcp_servers: list[str] = ()) -> list:
        if self._tools:
            return self._tools(uc_functions, vector_indexes, list(mcp_servers))
        tools: list = self._mcp_tools(list(mcp_servers)) if mcp_servers else []
        if uc_functions:
            from databricks_langchain import UCFunctionToolkit

            tools += UCFunctionToolkit(function_names=list(uc_functions)).tools
        for index in vector_indexes:
            from databricks_langchain import VectorSearchRetrieverTool

            tools.append(VectorSearchRetrieverTool(
                index_name=index,
                tool_name="search_" + _ident(index.split(".")[-1]),
                tool_description=f"Search the vector index {index}.",
            ))
        return tools

    def _mcp_tools(self, urls: list[str]) -> list:
        """Tools exposed by MCP servers. Servers in this workspace get the job's own credentials;
        any other server is called without authentication."""
        import asyncio
        from urllib.parse import urlparse

        from langchain_mcp_adapters.client import MultiServerMCPClient

        workspace, auth = None, {}
        try:
            from databricks.sdk import WorkspaceClient

            cfg = WorkspaceClient().config
            workspace, auth = urlparse(cfg.host).netloc, cfg.authenticate()
        except Exception:
            pass
        servers = {}
        for i, url in enumerate(urls):
            server = {"url": url, "transport": "streamable_http"}
            if workspace and urlparse(url).netloc == workspace:
                server["headers"] = dict(auth)
            servers[f"mcp_{i}"] = server
        return asyncio.run(MultiServerMCPClient(servers).get_tools())

    # ------------------------------------------------------------------ audit log
    def log_event(self, kind: str, node_id: str, payload: Any) -> None:
        self._events.append((self.pipeline, self.run_id, time.time(), node_id, kind, json.dumps(payload, default=str)))

    def flush_events(self) -> None:
        """Append buffered events (agent inputs/outputs, writes, decisions) to a table."""
        if not self._events:
            return
        self._ensure_schema()
        df = self.spark.createDataFrame(
            self._events, "pipeline string, run_id string, ts double, node string, kind string, payload string"
        )
        df.write.format(self.fmt).mode("append").saveAsTable(f"{self.staging}.strand_events")
        self._events = []

    def record_approval(self, node_id: str, status: str, payload: Any) -> None:
        self._ensure_schema()
        df = self.spark.createDataFrame(
            [(self.pipeline, self.run_id, node_id, status, time.time(), json.dumps(payload, default=str))],
            "pipeline string, thread_id string, node string, status string, ts double, payload string",
        )
        df.write.format(self.fmt).mode("append").saveAsTable(f"{self.staging}.strand_approvals")


_CTX: RunContext | None = None


def set_context(ctx: RunContext | None) -> None:
    global _CTX
    _CTX = ctx


def current() -> RunContext:
    if _CTX is None:
        raise RuntimeError("No strand run is active. Start the pipeline through strand_runtime.run().")
    return _CTX


# ---------------------------------------------------------------------- node helpers


def run_region(state: State, unit_id: str, fn: Callable[[RunContext, dict], dict]) -> dict:
    """Execute one fused dataflow region and store the references it produced."""
    ctx = current()
    n = (state.get("visits") or {}).get(unit_id, 0) + 1
    ctx.iteration = n
    refs = fn(ctx, state.get("datasets") or {})
    return {"datasets": refs, "visits": {unit_id: n}}


def router_result(state: State, node_id: str, value: Any, *, max_iterations: int) -> dict:
    n = (state.get("visits") or {}).get(node_id, 0) + 1
    branch = str(value).lower() if isinstance(value, bool) else ("" if value is None else str(value))
    if n > max_iterations:
        branch = "exhausted"
    current().log_event("route", node_id, {"value": value, "branch": branch, "visit": n})
    return {"outputs": {node_id: {"branch": branch, "value": value}}, "visits": {node_id: n}}


def approval_result(node_id: str, decision: Any) -> dict:
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision)}
    approved = bool(decision.get("approved"))
    result = {**decision, "approved": approved, "branch": "approved" if approved else "rejected"}
    current().log_event("approval", node_id, result)
    return {"outputs": {node_id: result}}


def branch_of(state: State, node_id: str) -> str:
    return ((state.get("outputs") or {}).get(node_id) or {}).get("branch", "")
