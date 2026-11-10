"""Agent nodes. The tool loop itself is LangGraph's; this only builds the prompt
from dataset references and records what went in and what came out."""
from __future__ import annotations

import json
import re
from typing import Any

from .core import State, current


def _summarise(ref: dict, sample: list[dict] | None) -> str:
    cols = ", ".join(f"{c['name']} {c['type']}" for c in ref.get("columns", []))
    lines = [f"- {ref['node']}: table `{ref['table']}`, {ref.get('row_count', '?')} rows", f"  columns: {cols}"]
    if sample is not None:
        lines.append("  sample rows: " + json.dumps(sample, default=str))
    return "\n".join(lines)


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

    agent = _build_agent(ctx.chat_model(endpoint), ctx.tools(list(uc_functions), list(vector_indexes)), system_prompt)
    result = agent.invoke({"messages": [("user", prompt)]}, {"recursion_limit": 2 * int(max_steps) + 2})
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

    output = {"text": text, "choice": choice, "steps": max(0, len(messages) - 2)}
    # Agents do not give the same answer twice: keep the exact input and output.
    ctx.log_event("agent", node_id, {"endpoint": endpoint, "prompt": prompt, "output": output})
    return {"outputs": {node_id: output}}
