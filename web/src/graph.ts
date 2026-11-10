import { createContext } from 'react'
import type { Edge, Node } from '@xyflow/react'
import type { Issue, NodeTypeDef, PipelineJson, Plan } from './api'

export type NData = { type: string; label?: string; config: Record<string, unknown> }
export type SNode = Node<NData>
export type SEdge = Edge<{ label?: string }>
export type Defs = Record<string, NodeTypeDef>
export type Meta = Pick<PipelineJson, 'name' | 'description' | 'settings'>

export const EMPTY_META: Meta = {
  name: 'untitled_pipeline',
  description: '',
  settings: { catalog: 'main', staging_schema: 'strand_staging', checkpoint: { kind: 'postgres', dsn_secret: '' } },
}

export const Ctx = createContext<{ defs: Defs; plan: Plan | null; issues: Issue[]; regionIndex: Record<string, number> }>({
  defs: {},
  plan: null,
  issues: [],
  regionIndex: {},
})

const isEmpty = (v: unknown) => v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0)

export function toPipeline(meta: Meta, nodes: SNode[], edges: SEdge[], defs: Defs): PipelineJson {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
  return {
    ...meta,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.type,
      ...(n.data.label ? { label: n.data.label } : {}),
      config: Object.fromEntries(Object.entries(n.data.config).filter(([, v]) => !isEmpty(v))),
      ui: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
    })),
    edges: edges
      .filter((e) => byId[e.source] && byId[e.target])
      .map((e) => {
        const sd = defs[byId[e.source].data.type]
        const td = defs[byId[e.target].data.type]
        const out: PipelineJson['edges'][number] = { source: e.source, target: e.target }
        if (sd?.kind === 'dataflow' && td?.kind === 'dataflow' && td.inputs.length > 1 && e.targetHandle) out.port = e.targetHandle
        if (sd?.branches.length) out.label = e.sourceHandle ?? ''
        else if (sd?.free_branches) out.label = e.data?.label ?? ''
        return out
      }),
  }
}

export function fromPipeline(p: PipelineJson, defs: Defs): { meta: Meta; nodes: SNode[]; edges: SEdge[] } {
  const type = Object.fromEntries(p.nodes.map((n) => [n.id, defs[n.type]]))
  return {
    meta: { name: p.name, description: p.description ?? '', settings: p.settings },
    nodes: p.nodes.map((n, i) => ({
      id: n.id,
      type: 'strand',
      position: n.ui ?? { x: 80 + i * 240, y: 120 },
      data: { type: n.type, label: n.label, config: n.config ?? {} },
    })),
    edges: p.edges.map((e, i) => {
      const sd = type[e.source]
      const td = type[e.target]
      const targetHandle = td?.kind === 'control' ? 'in' : sd?.kind === 'control' ? 'after' : (e.port ?? td?.inputs[0]?.name ?? 'in')
      return {
        id: `e${i}_${e.source}_${e.target}`,
        source: e.source,
        target: e.target,
        sourceHandle: sd?.branches.length ? (e.label ?? sd.branches[0]) : 'out',
        targetHandle,
        data: sd?.free_branches ? { label: e.label ?? '' } : {},
      }
    }),
  }
}

export function defaults(def: NodeTypeDef): Record<string, unknown> {
  return Object.fromEntries(def.fields.filter((f) => !isEmpty(f.default)).map((f) => [f.name, f.default]))
}

export function freshId(type: string, nodes: SNode[]): string {
  const taken = new Set(nodes.map((n) => n.id))
  const base = type === 'llm_map' ? 'llm' : type
  for (let i = 1; ; i++) if (!taken.has(`${base}_${i}`)) return `${base}_${i}`
}

export const REGION_COLORS = ['#0f9d8a', '#2f7de1', '#c2410c', '#7c8a12', '#b8317f', '#5b6ee1']
