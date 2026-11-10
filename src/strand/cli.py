"""strand check | compile | serve"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .codegen import compile_pipeline, write_files
from .spec import load
from .validate import validate


def _check(path: str):
    p = load(Path(path).read_text())
    issues, plan = validate(p)
    for i in issues:
        where = f" [{i.node}]" if i.node else ""
        print(f"{i.level.upper():7}{where} {i.message}", file=sys.stderr)
    return p, plan, any(i.level == "error" for i in issues)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="strand", description="Visual agentic data pipelines for Databricks.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="Validate a pipeline and print its execution plan.")
    c.add_argument("pipeline")
    c = sub.add_parser("compile", help="Generate PySpark, LangGraph and a Databricks Asset Bundle.")
    c.add_argument("pipeline")
    c.add_argument("-o", "--out", default=None, help="Output directory (default: build/<name>)")
    c = sub.add_parser("serve", help="Run the visual editor.")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("strand.server:app", host=args.host, port=args.port)
        return 0

    p, plan, failed = _check(args.pipeline)
    if failed or plan is None:
        return 1
    if args.cmd == "check":
        for u in plan.units:
            if u.kind == "region":
                print(f"{u.id}  [Spark]  {' -> '.join(u.nodes)}")
                for n, why in u.materialize.items():
                    print(f"    writes {n}: {why}")
            else:
                print(f"{u.id}  [{p.node(u.id).type}]")
        for e in plan.edges:
            print(f"  {e.source} -> {e.target}" + (f"  [{e.label}]" if e.label else "") + ("  (loop)" if e.back else ""))
        return 0
    out = Path(args.out or f"build/{p.name}")
    files = compile_pipeline(p, plan)
    write_files(files, out)
    print(f"Wrote {len(files)} files to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
