"""Job entry point: start a run, or resume one that stopped for approval.

A run that reaches a human-approval node does not wait. It records the
pending approval, the job ends, and the cluster is released. Approving
starts a new job run with the same thread id, and LangGraph continues from
the checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import uuid
from typing import Any, Callable

from .core import RunContext, set_context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="", help="Job run id; used as the thread id of a new run.")
    ap.add_argument("--thread-id", default="", help="Thread to resume.")
    ap.add_argument("--resume", default="", help='JSON decision for a pending approval, e.g. {"approved": true}')
    return ap.parse_args(argv)


def get_spark():
    from pyspark.sql import SparkSession

    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


@contextlib.contextmanager
def open_checkpointer(settings: dict, spark=None):
    cp = settings.get("checkpoint") or {}
    kind = cp.get("kind", "postgres")
    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
    elif kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(cp.get("path") or "strand_checkpoints.sqlite") as saver:
            yield saver
    else:
        # Lakebase speaks the Postgres protocol. Checkpoints are many small writes,
        # which is what Postgres is good at and Delta is not.
        dsn = os.environ.get("STRAND_CHECKPOINT_DSN")
        if not dsn and cp.get("dsn_secret"):
            from pyspark.dbutils import DBUtils

            scope, key = cp["dsn_secret"].split("/", 1)
            dsn = DBUtils(spark).secrets.get(scope, key)
        if not dsn:
            raise RuntimeError(
                "No checkpoint store configured. Set settings.checkpoint.dsn_secret to 'scope/key' of a "
                "secret holding a Postgres (Lakebase) connection string, or STRAND_CHECKPOINT_DSN."
            )
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(dsn) as saver:
            saver.setup()
            yield saver


def run(
    build_graph: Callable[[Any], Any],
    *,
    pipeline: str,
    settings: dict,
    argv: list[str] | None = None,
    thread_id: str | None = None,
    resume: Any = None,
    spark=None,
    checkpointer=None,
    chat_model=None,
    tools=None,
) -> dict:
    """Run or resume a pipeline. Returns a status dict and prints it as JSON."""
    if thread_id is None and resume is None:
        args = parse_args(argv)
        thread_id = args.thread_id or args.run_id or uuid.uuid4().hex[:12]
        resume = json.loads(args.resume) if args.resume else None
    thread_id = thread_id or uuid.uuid4().hex[:12]
    spark = spark or get_spark()

    with contextlib.suppress(Exception):  # tracing is optional
        import mlflow

        mlflow.langchain.autolog()

    ctx = RunContext(spark, pipeline=pipeline, run_id=thread_id, settings=settings, chat_model=chat_model, tools=tools)
    set_context(ctx)
    cm = contextlib.nullcontext(checkpointer) if checkpointer is not None else open_checkpointer(settings, spark)
    try:
        with cm as saver:
            graph = build_graph(saver)
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
            if resume is not None:
                from langgraph.types import Command

                pending = _interrupts(graph.get_state(config))
                if not pending:
                    raise RuntimeError(f"Run '{thread_id}' is not waiting for approval.")
                ctx.record_approval(pending[0].get("node", ""), "resolved", resume)
                graph.invoke(Command(resume=resume), config)
            else:
                before = graph.get_state(config)
                if not before.values and not before.next:
                    graph.invoke({"datasets": {}, "outputs": {}, "visits": {}}, config)
                elif before.next and not _interrupts(before):
                    # A retry of a run that died part-way: continue from the last checkpoint.
                    graph.invoke(None, config)

            snapshot = graph.get_state(config)
            pending = _interrupts(snapshot)
            for p in pending:
                ctx.record_approval(p.get("node", ""), "pending", p)
            status = {
                "status": "waiting_approval" if pending else "completed",
                "pipeline": pipeline,
                "thread_id": thread_id,
                "pending": pending,
                "outputs": snapshot.values.get("outputs", {}),
                "datasets": snapshot.values.get("datasets", {}),
            }
    finally:
        with contextlib.suppress(Exception):
            ctx.flush_events()
        set_context(None)
    print(json.dumps(status, default=str))
    return status


def _interrupts(snapshot) -> list[dict]:
    out = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for it in getattr(task, "interrupts", ()) or ():
            out.append(it.value if isinstance(it.value, dict) else {"value": it.value})
    return out
