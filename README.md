# strand

Visual pipelines that mix Spark transformations and AI agents, compiled to readable
PySpark + LangGraph and packaged as a Databricks Asset Bundle.

## Run the editor

```bash
uv venv --python 3.12 && uv pip install -e ".[run,dev]"
cd web && npm install && npm run build && cd ..
.venv/bin/strand serve          # http://127.0.0.1:8000
```

Pipelines are saved as YAML in `pipelines/`. **Compile** writes the generated project to `build/<name>/`.

## Command line

```bash
strand check examples/support_triage.yaml     # validate and print the execution plan
strand compile examples/support_triage.yaml   # write build/support_triage/
cd build/support_triage && databricks bundle deploy && databricks bundle run support_triage
```

## How it works

| Part | File | What it does |
|---|---|---|
| Spec | `src/strand/spec.py` | Node types and the pipeline model. A node is `dataflow` (Spark) or `control` (driver). |
| Validation | `src/strand/validate.py` | Ports, branch labels, bounded loops, config checks. |
| Plan | `src/strand/plan.py` | Region fusion: contiguous Spark nodes become one function; outputs are written only where a control node or a later region reads them. |
| Codegen | `src/strand/codegen.py` | `transforms.py` (PySpark), `workflow.py` (LangGraph), `main.py`, bundle YAML. |
| Runtime | `src/strand_runtime/` | Copied into every build. Staging tables, idempotent sinks, agent call, approvals, checkpointing. |
| Editor | `src/strand/server.py`, `web/` | FastAPI backend and React Flow canvas. |

Rules the compiler enforces:

- Agent state holds dataset references (table, columns, row count, version), never rows.
- "LLM over every row" is the LLM map node (`ai_query` on Spark), not an agent in a UDF.
- A human approval ends the job run. Approving starts a new run with the same thread id, which
  resumes from the checkpoint; steps that already ran are not repeated.
- Step outputs go to run-scoped staging tables; appends carry a Delta transaction id so a retry does not duplicate rows.
- Loops must be closed by a router with a maximum number of iterations.

## What has and has not been run

Run here: the compiler, validation, and the generated LangGraph workflows end to end
(approval pause and resume, rejection, branches, bounded loops, parallel fan-in) against a
stand-in Spark session (`tests/fakes.py`), plus the editor in a browser.

Not run yet (needs a workspace or a JVM): the generated PySpark on real Spark, `ai_query`,
Unity Catalog tools, the Postgres/Lakebase checkpointer, the bundle deploy, `app.yaml`,
and `src/strand/databricks_ops.py` (Run and Approvals in the editor). Preview needs
`databricks-connect` or a local Java install.
