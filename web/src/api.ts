export type FieldDef = {
  name: string
  label: string
  type: 'text' | 'textarea' | 'code' | 'select' | 'list' | 'number' | 'bool'
  required: boolean
  default: unknown
  options: string[]
  help: string
  language: string
}

export type NodeTypeDef = {
  type: string
  kind: 'dataflow' | 'control'
  label: string
  description: string
  fields: FieldDef[]
  inputs: { name: string; max: number | null; required: boolean }[]
  branches: string[]
  free_branches: boolean
}

export type Issue = { level: 'error' | 'warning'; message: string; node: string | null }

export type Unit = {
  id: string
  kind: 'region' | 'control'
  nodes: string[]
  materialize: Record<string, string>
  reads: string[]
  join: string
}

export type Plan = {
  units: Unit[]
  edges: { source: string; target: string; label: string | null; back: boolean }[]
  node_unit: Record<string, string>
}

export type PipelineJson = {
  name: string
  description: string
  settings: {
    catalog: string
    staging_schema: string
    table_format?: string
    sample_rows?: number
    checkpoint: { kind: string; dsn_secret?: string | null; path?: string | null }
  }
  nodes: { id: string; type: string; label?: string; config: Record<string, unknown>; ui?: { x: number; y: number } }[]
  edges: { source: string; target: string; port?: string; label?: string }[]
}

export type CheckResult = { issues: Issue[]; plan: Plan | null }
export type CompileResult = CheckResult & { files: Record<string, string>; written: string | null }
export type PreviewResult = { columns: { name: string; type: string }[]; rows: Record<string, unknown>[]; code: string }
export type Approval = { thread_id: string; node: string; ts: number; message?: string }

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    let detail: unknown = r.statusText
    try {
      detail = (await r.json()).detail
    } catch {
      /* keep status text */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json() as Promise<T>
}

export const api = {
  nodeTypes: () => call<NodeTypeDef[]>('GET', '/api/node-types'),
  list: () => call<string[]>('GET', '/api/pipelines'),
  get: (name: string) => call<PipelineJson>('GET', `/api/pipelines/${name}`),
  save: (p: PipelineJson) => call<{ saved: string }>('PUT', `/api/pipelines/${p.name}`, { pipeline: p }),
  check: (p: PipelineJson) => call<CheckResult>('POST', '/api/check', { pipeline: p }),
  compile: (p: PipelineJson) => call<CompileResult>('POST', '/api/compile?write=true', { pipeline: p }),
  preview: (p: PipelineJson, node_id: string) => call<PreviewResult>('POST', '/api/preview', { pipeline: p, node_id }),
  run: (p: PipelineJson) => call<{ run_id: number }>('POST', '/api/run', { pipeline: p }),
  approvals: (p: PipelineJson) => call<Approval[]>('POST', '/api/approvals', { pipeline: p }),
  resolve: (p: PipelineJson, thread_id: string, approved: boolean) =>
    call<{ run_id: number }>('POST', '/api/approvals/resolve', { pipeline: p, thread_id, approved }),
}
