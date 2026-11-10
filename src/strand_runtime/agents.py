"""Agent nodes. The tool loop itself is LangGraph's; this only builds the prompt
from dataset references, wires up the tools, and records what went in and out."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import State, current


@dataclass
class PipelineTool:
    """A fused Spark region the agent may run. Arguments are bound as SQL parameters."""

    name: str
    description: str
    fn: Callable[..., dict]
    params: list[str] = field(default_factory=list)


def _summarise(ref: dict, sample: list[dict] | None = None) -> str:
    cols = ", ".join(f"{c['name']} {c['type']}" for c in ref.get("columns", []))
    lines = [f"- {ref['node']}: table `{ref['table']}`, {ref.get('row_count', '?')} rows", f"  columns: {cols}"]
    if sample is not None:
        lines.append("  sample rows: " + json.dumps(sample, default=str))
    return "\n".join(lines)


def _pipeline_tool(spec: PipelineTool, datasets: dict, produced: dict, calls: list):
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    ctx = current()

    def call(**kwargs) -> str:
        calls.append(spec.name)
        # each call is its own pass, so an append written by call 2 is not mistaken for call 1
        ctx.iteration = len(calls)
        refs = spec.fn(ctx, {**datasets, **produced}, **{k: str(kwargs[k]) for k in spec.params})
        produced.update(refs)
        ctx.log_event("tool", spec.name, {"args": kwargs, "datasets": refs})
        return "Done. Results (references, not rows):\n" + "\n".join(_summarise(r) for r in refs.values())

    args = create_model(f"{spec.name}_args", **{p: (str, ...) for p in spec.params})
    return StructuredTool.from_function(func=call, name=spec.name, description=spec.description, args_schema=args)


def _build_agent(model, tools, system_prompt: str):
    try:
        from langchain.agents import create_agent

        return create_agent(model, tools, system_prompt=system_prompt)
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools, prompt=system_prompt)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def run_agent(
    state: State,
    node_id: str,
    *,
    endpoint: str,
    instructions: str,
    system_prompt: str = "",
    reads: list[str] = (),
    uc_functions: list[str] = (),
    vector_indexes: list[str] = (),
    mcp_servers: list[str] = (),
    pipeline_tools: list[PipelineTool] = (),
    choices: list[str] = (),
    include_sample: bool = False,
    max_steps: int = 8,
) -> dict:
    ctx = current()
    datasets = state.get("datasets") or {}
    refs = [datasets[r] for r in reads if r in datasets]

    parts = [instructions.strip()]
    if refs:
        parts.append("Datasets available to you (query them with your tools; you are not shown the rows):\n"
                     + "\n".join(_summarise(r, ctx.sample(r) if include_sample else None) for r in refs))
    earlier = {k: v for k, v in (state.get("outputs") or {}).items() if k != node_id}
    if earlier:
        parts.append("Results of earlier steps:\n" + json.dumps(earlier, default=str, indent=1))
    if choices:
        parts.append("Finish with a final line of exactly this form:\nDECISION: <" + " | ".join(choices) + ">")
    prompt = "\n\n".join(parts)

    produced: dict[str, dict] = {}
    calls: list[str] = []
    tools = ctx.tools(list(uc_functions), list(vector_indexes), list(mcp_servers))
    tools += [_pipeline_tool(t, datasets, produced, calls) for t in pipeline_tools]
    agent = _build_agent(ctx.chat_model(endpoint), tools, system_prompt)
    request, config = {"messages": [("user", prompt)]}, {"recursion_limit": 2 * int(max_steps) + 2}
    if any(getattr(t, "func", True) is None and getattr(t, "coroutine", None) for t in tools):
        result = asyncio.run(agent.ainvoke(request, config))  # MCP tools are async-only
    else:
        result = agent.invoke(request, config)
    messages = result["messages"]
    text = _text(messages[-1].content)

    choice = None
    if choices:
        m = re.findall(r"DECISION:\s*<?\s*([^\n>]+?)\s*>?\s*$", text, flags=re.M)
        if m and m[-1] in choices:
            choice = m[-1]
        else:  # the model ignored the format; accept an unambiguous mention
            named = [c for c in choices if re.search(rf"\b{re.escape(c)}\b", text)]
            choice = named[0] if len(named) == 1 else None

    output = {"text": text, "choice": choice, "steps": max(0, len(messages) - 2), "tool_calls": calls}
    # Agents do not give the same answer twice: keep the exact input and output.
    ctx.log_event("agent", node_id, {"endpoint": endpoint, "prompt": prompt, "output": output})
    return {"outputs": {node_id: output}, "datasets": produced}
