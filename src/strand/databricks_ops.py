"""Workspace calls the editor makes: start a run, list pending approvals, resume a run.

NOT exercised by the test suite - it needs a real workspace. Inside a
Databricks App the SDK picks up the app's credentials on its own; locally it
uses your default profile. Requires `pip install databricks-sdk` and, for
approvals, DATABRICKS_WAREHOUSE_ID.
"""
from __future__ import annotations

import json
import os

from .spec import Pipeline


def _client():
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as e:
        raise RuntimeError("databricks-sdk is not installed, so the editor cannot reach a workspace.") from e
    return WorkspaceClient()


def _job_id(w, p: Pipeline) -> int:
    wanted = f"strand · {p.name}"
    for job in w.jobs.list():
        if job.settings and job.settings.name and job.settings.name.endswith(wanted):
            return job.job_id
    raise RuntimeError(f"No deployed job for '{p.name}'. Run `databricks bundle deploy` in build/{p.name} first.")


def start_run(p: Pipeline) -> dict:
    w = _client()
    run = w.jobs.run_now(job_id=_job_id(w, p))
    return {"run_id": run.run_id}


def resolve_approval(p: Pipeline, thread_id: str, decision: dict) -> dict:
    w = _client()
    run = w.jobs.run_now(job_id=_job_id(w, p), job_parameters={"thread_id": thread_id, "resume": json.dumps(decision)})
    return {"run_id": run.run_id}


def list_approvals(p: Pipeline) -> list[dict]:
    """Approvals whose latest record is still 'pending'."""
    warehouse = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse:
        raise RuntimeError("Set DATABRICKS_WAREHOUSE_ID so the editor can read the approvals table.")
    s = p.settings
    table = f"{s.catalog}.{s.staging_schema}.strand_approvals" if s.catalog else f"{s.staging_schema}.strand_approvals"
    w = _client()
    res = w.statement_execution.execute_statement(
        warehouse_id=warehouse,
        statement=(
            f"SELECT thread_id, node, status, ts, payload FROM {table} WHERE pipeline = :pipeline "
            "QUALIFY row_number() OVER (PARTITION BY thread_id, node ORDER BY ts DESC) = 1"
        ),
        parameters=[{"name": "pipeline", "value": p.name}],
        wait_timeout="30s",
    )
    rows = (res.result.data_array or []) if res.result else []
    out = []
    for thread_id, node, status, ts, payload in rows:
        if status == "pending":
            out.append({"thread_id": thread_id, "node": node, "ts": float(ts), **json.loads(payload or "{}")})
    return out
