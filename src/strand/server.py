"""Backend of the visual editor. Runs locally or as a Databricks App."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import databricks_ops
from .codegen import compile_pipeline, gen_preview, write_files
from .spec import NODE_TYPES, Pipeline, dump, load
from .validate import NAME_RE, validate

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = Path(os.environ.get("STRAND_PIPELINES_DIR", "pipelines"))
BUILD = Path(os.environ.get("STRAND_BUILD_DIR", "build"))
WEB = Path(os.environ.get("STRAND_WEB_DIR", ROOT / "web" / "dist"))

app = FastAPI(title="strand")


def _seed() -> None:
    PIPELINES.mkdir(parents=True, exist_ok=True)
    if not any(PIPELINES.glob("*.yaml")):
        for f in (ROOT / "examples").glob("*.yaml"):
            shutil.copy(f, PIPELINES / f.name)


_seed()


def _path(name: str) -> Path:
    if not NAME_RE.match(name):
        raise HTTPException(400, "Invalid pipeline name.")
    return PIPELINES / f"{name}.yaml"


def _parse(data: dict) -> Pipeline:
    try:
        return Pipeline.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False, include_input=False)))


class Body(BaseModel):
    pipeline: dict[str, Any]


class PreviewBody(Body):
    node_id: str
    limit: int = 1000


class ResolveBody(Body):
    thread_id: str
    approved: bool
    comment: str = ""


@app.get("/api/node-types")
def node_types() -> list[dict]:
    return [t.to_json() for t in NODE_TYPES.values()]


@app.get("/api/pipelines")
def list_pipelines() -> list[str]:
    return sorted(f.stem for f in PIPELINES.glob("*.yaml"))


@app.get("/api/pipelines/{name}")
def get_pipeline(name: str) -> dict:
    path = _path(name)
    if not path.exists():
        raise HTTPException(404, "No such pipeline.")
    return load(path.read_text()).model_dump(exclude_none=True)


@app.put("/api/pipelines/{name}")
def put_pipeline(name: str, body: Body) -> dict:
    p = _parse(body.pipeline)
    if p.name != name:
        raise HTTPException(400, "Pipeline name does not match the URL.")
    _path(name).write_text(dump(p))
    return {"saved": str(_path(name))}


@app.delete("/api/pipelines/{name}")
def delete_pipeline(name: str) -> dict:
    _path(name).unlink(missing_ok=True)
    return {"deleted": name}


@app.post("/api/check")
def check(body: Body) -> dict:
    issues, plan = validate(_parse(body.pipeline))
    return {"issues": [i.to_json() for i in issues], "plan": plan.to_json() if plan else None}


@app.post("/api/compile")
def compile_(body: Body, write: bool = False) -> dict:
    p = _parse(body.pipeline)
    issues, plan = validate(p)
    out = {"issues": [i.to_json() for i in issues], "plan": plan.to_json() if plan else None, "files": {}, "written": None}
    if plan is None or any(i.level == "error" for i in issues):
        return out
    files = compile_pipeline(p, plan)
    # the vendored runtime is the same for every pipeline; the editor does not need to show it
    out["files"] = {k: v for k, v in files.items() if not k.startswith("src/strand_runtime/")}
    if write:
        write_files(files, BUILD / p.name)
        out["written"] = str(BUILD / p.name)
    return out


_spark = None


def _preview_spark():
    global _spark
    if _spark is not None:
        return _spark
    try:
        from databricks.connect import DatabricksSession

        _spark = DatabricksSession.builder.serverless(True).getOrCreate()
    except ImportError:
        try:
            from pyspark.sql import SparkSession

            _spark = SparkSession.builder.master("local[2]").appName("strand-preview").getOrCreate()
        except Exception as e:
            raise HTTPException(
                503,
                "Preview needs Spark. Install databricks-connect and configure a workspace, "
                f"or install Java for a local Spark session. ({type(e).__name__})",
            )
    return _spark


@app.post("/api/preview")
def preview(body: PreviewBody) -> dict:
    p = _parse(body.pipeline)
    issues, _ = validate(p)
    blocking = [i for i in issues if i.level == "error"]
    if blocking:
        raise HTTPException(400, blocking[0].message)
    limit = max(1, min(body.limit, 1000))
    try:
        code = gen_preview(p, body.node_id, limit)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    spark = _preview_spark()
    ns: dict = {}
    exec(code, ns)  # the pipeline author's own code, same as running it as a job
    try:
        df = ns["preview"](spark)
        rows = [json.loads(json.dumps(r.asDict(recursive=True), default=str)) for r in df.collect()]
        columns = [{"name": f.name, "type": f.dataType.simpleString()} for f in df.schema.fields]
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {str(e)[:2000]}")
    return {"columns": columns, "rows": rows, "code": code}


def _workspace(fn, *args):
    try:
        return fn(*args)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {str(e)[:500]}")


@app.post("/api/run")
def run(body: Body) -> dict:
    return _workspace(databricks_ops.start_run, _parse(body.pipeline))


@app.post("/api/approvals")
def approvals(body: Body) -> list[dict]:
    return _workspace(databricks_ops.list_approvals, _parse(body.pipeline))


@app.post("/api/approvals/resolve")
def resolve(body: ResolveBody) -> dict:
    decision = {"approved": body.approved, "comment": body.comment}
    return _workspace(databricks_ops.resolve_approval, _parse(body.pipeline), body.thread_id, decision)


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB / path
        return FileResponse(f if path and f.is_file() and WEB in f.resolve().parents else WEB / "index.html")
