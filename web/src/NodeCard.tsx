import { useContext } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Ctx, REGION_COLORS, type SNode } from './graph'

const SUMMARY: Record<string, string[]> = {
  source: ['table', 'path'],
  filter: ['condition'],
  sql: ['query'],
  python: ['code'],
  join: ['on'],
  llm_map: ['prompt'],
  ai_task: ['function'],
  embed: ['column'],
  sink: ['table'],
  agent: ['instructions'],
  router: ['expr'],
  approval: ['message'],
}

export function NodeCard({ id, data, selected }: NodeProps<SNode>) {
  const { defs, plan, issues, regionIndex } = useContext(Ctx)
  const def = defs[data.type]
  if (!def) return <div className="node">unknown: {data.type}</div>

  const mine = issues.filter((i) => i.node === id)
  const level = mine.some((i) => i.level === 'error') ? 'error' : mine.length ? 'warning' : ''
  const unitId = plan?.node_unit[id]
  const unit = plan?.units.find((u) => u.id === unitId)
  const region = unit?.kind === 'region' ? unit : null
  const color = region ? REGION_COLORS[regionIndex[region.id] % REGION_COLORS.length] : undefined
  const written = region?.materialize[id]
  const summary = SUMMARY[data.type]?.map((k) => data.config[k]).find((v) => v && String(v).length)
  const dataflow = def.kind === 'dataflow'

  return (
    <div className={`node ${def.kind} ${selected ? 'selected' : ''} ${level}`} style={color ? { ['--region' as string]: color } : undefined}>
      {dataflow ? (
        <>
          <Handle id="after" type="target" position={Position.Top} className="h-ctl" title="Run after (control)" />
          {def.inputs.map((p, i) => (
            <Handle
              key={p.name}
              id={p.name}
              type="target"
              position={Position.Left}
              className="h-data"
              style={{ top: `${((i + 1) * 100) / (def.inputs.length + 1)}%` }}
              title={`${p.name} input`}
            />
          ))}
          {def.inputs.length > 1 &&
            def.inputs.map((p, i) => (
              <span key={p.name} className="port" style={{ top: `${((i + 1) * 100) / (def.inputs.length + 1)}%` }}>
                {p.name}
              </span>
            ))}
        </>
      ) : (
        <Handle id="in" type="target" position={Position.Left} className="h-any" />
      )}

      <div className="node-head">
        <span className="kind">{def.label}</span>
        {mine.length > 0 && (
          <span className={`dot ${level}`} title={mine.map((i) => i.message).join('\n')}>
            {mine.length}
          </span>
        )}
      </div>
      <div className="node-id">{data.label || id}</div>
      {data.label && <div className="node-sub">{id}</div>}
      {summary != null && <div className="node-summary">{Array.isArray(summary) ? summary.join(', ') : String(summary)}</div>}
      {region?.tool_of && <div className="toolof">tool of {region.tool_of}</div>}
      {written && (
        <div className="written" title={written}>
          {written.startsWith('no copy') ? '↗ passed by reference' : '▣ written to a table'}
        </div>
      )}

      {def.branches.length > 0 ? (
        def.branches.map((b, i) => (
          <div key={b}>
            <span className="branch" style={{ top: `${((i + 1) * 100) / (def.branches.length + 1)}%` }}>
              {b}
            </span>
            <Handle
              id={b}
              type="source"
              position={Position.Right}
              className="h-ctl"
              style={{ top: `${((i + 1) * 100) / (def.branches.length + 1)}%` }}
            />
          </div>
        ))
      ) : (
        <Handle id="out" type="source" position={Position.Right} className={dataflow ? 'h-data' : 'h-ctl'} />
      )}
      {data.type === 'agent' && (
        <>
          <span className="tool-port">tools</span>
          <Handle id="tool" type="source" position={Position.Bottom} className="h-tool" title="Connect to a Spark node the agent may run as a tool" />
        </>
      )}
    </div>
  )
}
