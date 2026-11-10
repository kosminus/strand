from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import strand_runtime as rt
from strand.codegen import compile_pipeline, gen_preview, prompt_to_sql, write_files
from strand.spec import dump, load
from strand.validate import validate

from fakes import FakeSpark

EXAMPLE = Path(__file__).parent.parent / "examples" / "support_triage.yaml"


def build(p, tmp_path):
    issues, plan = validate(p)
    assert not [i for i in issues if i.level == "error"], issues
    files = compile_pipeline(p, plan)
    for path, src in files.items():
        if path.endswith(".py"):
            compile(src, path, "exec")  # generated code is at least valid Python
    write_files(files, tmp_path)
    sys.path.insert(0, str(tmp_path / "src"))
    for m in [m for m in sys.modules if m.split(".")[0] == p.name]:
        del sys.modules[m]
    try:
        return importlib.import_module(f"{p.name}.workflow"), plan
    finally:
        sys.path.pop(0)


def model_saying(*answers):
    return lambda endpoint: GenericFakeChatModel(messages=iter([AIMessage(content=a) for a in answers]))


def run(wf, spark, saver, answers=(), **kw):
    model = kw.pop("chat_model", None) or model_saying(*answers)
    return rt.run(
        wf.build_graph, pipeline=wf.PIPELINE, settings={**wf.SETTINGS, "checkpoint": {"kind": "memory"}},
        spark=spark, checkpointer=saver, chat_model=model, tools=kw.pop("tools", lambda uc, vs, mcp: []), **kw,
    )


# ------------------------------------------------------------------ plan


def test_fusion_groups_dataflow_and_marks_materialisation():
    _, plan = validate(load(EXAMPLE.read_text()))
    regions = {u.id: u for u in plan.units if u.kind == "region"}
    assert set(regions["region_summary"].nodes) == {"tickets", "customers", "recent", "enriched", "classified", "summary"}
    # only what crosses a boundary is written
    assert set(regions["region_summary"].materialize) == {"classified", "summary"}
    assert regions["region_save_escalations"].nodes == ["urgent", "save_escalations"]
    assert regions["region_save_escalations"].reads == ["classified"]
    # the data dependency region_summary -> region_save_escalations is implied by the
    # path through the approval and must not trigger the write early
    assert ("region_summary", "region_save_escalations") not in {(e.source, e.target) for e in plan.edges}


def test_yaml_round_trip():
    p = load(EXAMPLE.read_text())
    assert load(dump(p)) == p


def test_prompt_template():
    assert prompt_to_sql("Hi {a}, it's {b}") == "concat('Hi ', CAST(`a` AS STRING), ', it\\'s ', CAST(`b` AS STRING))"


# ------------------------------------------------------------------ validation


def errors(yaml_text):
    return [i.message for i in validate(load(yaml_text))[0] if i.level == "error"]


def test_validation_catches_common_mistakes():
    assert any("missing its 'right' input" in e for e in errors("""
name: p
nodes:
  - {id: a, type: source, config: {table: m.s.a}}
  - {id: j, type: join, config: {"on": [id]}}
edges: [{from: a, to: j, port: left}]
"""))
    assert any("needs a branch label" in e for e in errors("""
name: p
nodes:
  - {id: r, type: router, config: {expr: "'x'"}}
  - {id: a, type: approval, config: {message: ok}}
edges: [{from: r, to: a}]
"""))
    assert any("closes a loop" in e for e in errors("""
name: p
nodes:
  - {id: a, type: agent, config: {endpoint: e, instructions: x}}
  - {id: b, type: agent, config: {endpoint: e, instructions: x}}
edges: [{from: a, to: b}, {from: b, to: a}]
"""))
    assert any("syntax error" in e for e in errors("""
name: p
nodes: [{id: r, type: router, config: {expr: "1 +"}}]
"""))


# ------------------------------------------------------------------ end to end


def test_escalate_pauses_for_approval_then_resumes_without_redoing_work(tmp_path):
    wf, _ = build(load(EXAMPLE.read_text()), tmp_path)
    spark, saver = FakeSpark(), InMemorySaver()

    first = run(wf, spark, saver, ["Urgent volume is double the baseline.\nDECISION: escalate"], thread_id="run1")
    assert first["status"] == "waiting_approval"
    assert first["pending"][0]["node"] == "approve"
    assert first["outputs"]["triage"]["choice"] == "escalate"
    # agent state holds references, never rows
    ref = first["datasets"]["summary"]
    assert set(ref) == {"node", "table", "columns", "row_count", "version"}
    assert ref["table"] == "main.strand_staging.support_triage__summary__run1"
    assert spark.user_writes() == []  # nothing written to user tables before approval
    staged = len(spark.writes)

    second = run(wf, spark, saver, thread_id="run1", resume={"approved": True, "by": "reviewer"})
    assert second["status"] == "completed"
    assert second["outputs"]["approve"]["branch"] == "approved"
    (table, mode, options, lineage), = spark.user_writes()
    assert (table, mode) == ("main.support.escalations", "append")
    assert options["txnAppId"] == "strand:support_triage:run1:save_escalations"
    # the write reads the staged table by reference; the LLM batch was not run again
    assert "support_triage__classified__run1" in lineage and "ai_query" not in lineage
    assert not [w for w in spark.writes[staged:] if "__classified__" in w[0] or "__summary__" in w[0]]


def test_rejected_and_ignore_paths(tmp_path):
    wf, _ = build(load(EXAMPLE.read_text()), tmp_path)

    spark, saver = FakeSpark(), InMemorySaver()
    run(wf, spark, saver, ["DECISION: escalate"], thread_id="a")
    done = run(wf, spark, saver, thread_id="a", resume={"approved": False})
    assert done["status"] == "completed" and spark.user_writes() == []

    spark, saver = FakeSpark(), InMemorySaver()
    done = run(wf, spark, saver, ["Looks normal.\nDECISION: ignore"], thread_id="b")
    assert done["status"] == "completed"
    assert [w[0] for w in spark.user_writes()] == ["main.support.weekly_summary"]


LOOP = """
name: fix_loop
nodes:
  - {id: raw, type: source, config: {table: m.s.raw}}
  - {id: checked, type: sql, config: {query: "SELECT * FROM {raw} WHERE ok"}}
  - {id: log, type: sink, config: {table: m.s.attempts, mode: append}}
  - {id: judge, type: agent, config: {endpoint: e, instructions: Good enough, choices: [retry, done]}}
  - {id: again, type: router, config: {expr: "outputs['judge']['choice']", max_iterations: 2}}
edges:
  - {from: raw, to: checked}
  - {from: checked, to: log}
  - {from: checked, to: judge}
  - {from: judge, to: again}
  - {from: again, to: raw, label: retry}
"""


def test_loops_are_bounded_and_appends_are_idempotent_per_pass(tmp_path):
    wf, plan = build(load(LOOP), tmp_path)
    assert [e for e in plan.edges if e.back] and len([u for u in plan.units if u.kind == "region"]) == 1
    spark = FakeSpark()
    done = run(wf, spark, InMemorySaver(), ["DECISION: retry"] * 10, thread_id="t")
    assert done["status"] == "completed"
    assert done["outputs"]["again"]["branch"] == "exhausted"
    appends = [w for w in spark.user_writes()]
    assert [w[2]["txnVersion"] for w in appends] == ["1", "2", "3"]  # one append per pass, each with its own version


def test_parallel_branches_join(tmp_path):
    wf, plan = build(load("""
name: fan_in
nodes:
  - {id: a, type: source, config: {table: m.s.a}}
  - {id: b, type: source, config: {table: m.s.b}}
  - {id: one, type: agent, config: {endpoint: e, instructions: x}}
  - {id: two, type: agent, config: {endpoint: e, instructions: y}}
  - {id: both, type: agent, config: {endpoint: e, instructions: z}}
edges:
  - {from: a, to: one}
  - {from: b, to: two}
  - {from: one, to: both}
  - {from: two, to: both}
"""), tmp_path)
    assert plan.unit("both").join == "all"
    done = run(wf, FakeSpark(), InMemorySaver(), ["ok"] * 3, thread_id="t")
    assert set(done["outputs"]) == {"one", "two", "both"}


def test_preview_code_limits_sources():
    code = gen_preview(load(EXAMPLE.read_text()), "enriched", 1000)
    ns: dict = {}
    exec(code, ns)
    df = ns["preview"](FakeSpark())
    assert df.lineage.count("limit(") == 3 and "ai_query" not in df.lineage


def test_runtime_refuses_to_resume_a_run_that_is_not_waiting(tmp_path):
    wf, _ = build(load(EXAMPLE.read_text()), tmp_path)
    with pytest.raises(RuntimeError, match="not waiting"):
        run(wf, FakeSpark(), InMemorySaver(), thread_id="nope", resume={"approved": True})


# ------------------------------------------------------------------ AI nodes and tools

AI_NODES = """
name: ai_nodes
nodes:
  - {id: docs, type: source, config: {table: m.s.docs}}
  - id: tagged
    type: llm_map
    config: {endpoint: ep, prompt: "Tag: {body}", output_column: tags, output_schema: "topic STRING, entities ARRAY<STRING>"}
  - {id: kind, type: ai_task, config: {function: classify, column: body, labels: [bug, question], output_column: kind}}
  - {id: short, type: ai_task, config: {function: summarize, column: body, max_words: 20, output_column: short}}
  - {id: vec, type: embed, config: {endpoint: databricks-gte-large-en, column: body}}
  - {id: save, type: sink, config: {table: m.s.docs_ai}}
edges:
  - {from: docs, to: tagged}
  - {from: tagged, to: kind}
  - {from: kind, to: short}
  - {from: short, to: vec}
  - {from: vec, to: save}
"""


def test_ai_nodes_generate_the_expected_sql(tmp_path):
    wf, _ = build(load(AI_NODES), tmp_path)
    spark = FakeSpark()
    run(wf, spark, InMemorySaver(), thread_id="t")
    lineage = spark.user_writes()[0][3]
    assert ("from_json(ai_query('ep', concat('Tag: ', CAST(`body` AS STRING)), responseFormat => "
            "'STRUCT<result: STRUCT<topic: STRING, entities: ARRAY<STRING>>>'), "
            "'topic STRING, entities ARRAY<STRING>') AS `tags`") in lineage
    assert "ai_classify(`body`, array('bug', 'question')) AS `kind`" in lineage
    assert "ai_summarize(`body`, 20) AS `short`" in lineage
    assert "ai_query('databricks-gte-large-en', CAST(`body` AS STRING)) AS `embedding`" in lineage
    assert any("needs labels" in e for e in errors("""
name: p
nodes:
  - {id: a, type: source, config: {table: m.s.a}}
  - {id: b, type: ai_task, config: {function: extract, column: x}}
  - {id: c, type: sink, config: {table: m.s.c}}
edges: [{from: a, to: b}, {from: b, to: c}]
"""))


TOOLS = """
name: with_tools
nodes:
  - {id: orders, type: source, config: {table: m.s.orders}}
  - {id: lookup, type: source, config: {table: m.s.regions}}
  - {id: stats, type: sql, config: {query: "SELECT count(*) AS n FROM {orders}"}}
  - id: analyst
    type: agent
    config: {endpoint: e, instructions: Investigate, mcp_servers: ["https://example.test/api/2.0/mcp/functions/m/s"]}
  - {id: by_region, type: sql, config: {query: "SELECT * FROM {orders} JOIN {lookup} USING (region) WHERE region = :region AND day >= :since::date"}}
  - {id: flagged, type: sink, config: {table: m.s.flagged, mode: append}}
  - {id: report, type: sink, config: {table: m.s.report}}
edges:
  - {from: orders, to: stats}
  - {from: stats, to: analyst}
  - {from: analyst, to: by_region, tool: true, description: Orders for one region since a date.}
  - {from: orders, to: by_region}
  - {from: lookup, to: by_region}
  - {from: by_region, to: flagged}
  - {from: analyst, to: report}
  - {from: stats, to: report}
"""


class ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_can_run_part_of_the_pipeline_as_a_tool(tmp_path):
    from langchain_core.tools import StructuredTool

    wf, plan = build(load(TOOLS), tmp_path)
    tool = plan.unit("region_flagged")
    assert (tool.tool_of, tool.params) == ("analyst", ["region", "since"])
    assert plan.unit("analyst").tools == ["region_flagged"]
    # a tool is not a scheduled step, and what it reads is prepared before the agent starts
    assert "region_flagged" not in {e.source for e in plan.edges} | {e.target for e in plan.edges}
    assert ("region_lookup", "analyst") in {(e.source, e.target) for e in plan.edges}
    src = (tmp_path / "src/with_tools/workflow.py").read_text()
    assert 'g.add_node("region_flagged"' not in src and "rt.PipelineTool(\"run_flagged\"" in src

    seen = {}

    async def remote(q: str) -> str:
        return "ok"

    def tools(uc, vs, mcp):
        seen["mcp"] = mcp
        return [StructuredTool.from_function(coroutine=remote, name="remote", description="An MCP tool.")]

    call = {"name": "run_flagged", "args": {"region": "emea'; DROP TABLE x; --", "since": "2026-01-01"}, "id": "c1"}
    model = lambda endpoint: ToolCallingModel(messages=iter([  # noqa: E731
        AIMessage(content="", tool_calls=[call]),
        AIMessage(content="", tool_calls=[{**call, "id": "c2"}]),
        AIMessage(content="Flagged."),
    ]))
    spark = FakeSpark()
    done = run(wf, spark, InMemorySaver(), thread_id="t", chat_model=model, tools=tools)

    assert seen["mcp"] == ["https://example.test/api/2.0/mcp/functions/m/s"]
    assert done["outputs"]["analyst"]["tool_calls"] == ["run_flagged", "run_flagged"]
    assert done["datasets"]["flagged"]["table"] == "m.s.flagged"
    appends = [w for w in spark.user_writes() if w[0] == "m.s.flagged"]
    assert [w[2]["txnVersion"] for w in appends] == ["1", "2"]  # two calls, two appends
    # the model's argument is bound as a parameter, not pasted into the SQL text
    assert "DROP TABLE" not in appends[0][3] and "args[region, since]" in appends[0][3]
    assert [w[0] for w in spark.user_writes()][-1] == "m.s.report"  # the normal step after the agent still ran


def test_a_tool_cannot_also_be_a_normal_step():
    # `report` would be both part of the tool (it reads by_region) and a step after the agent
    assert any("cannot also be triggered by 'analyst'" in e for e in errors(TOOLS.replace(
        "  - {from: stats, to: report}", "  - {from: by_region, to: report}")))
