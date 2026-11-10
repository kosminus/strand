import { useContext, useEffect, useState } from 'react'
import type { FieldDef } from './api'
import { Ctx, type Meta, type SEdge, type SNode } from './graph'

function FieldInput({ f, value, onChange }: { f: FieldDef; value: unknown; onChange: (v: unknown) => void }) {
  const v = value ?? ''
  if (f.type === 'bool')
    return <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
  if (f.type === 'select')
    return (
      <select value={String(v || f.default || '')} onChange={(e) => onChange(e.target.value)}>
        {f.options.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
    )
  if (f.type === 'textarea' || f.type === 'code')
    return (
      <textarea
        className={f.type === 'code' ? 'mono' : ''}
        rows={f.type === 'code' ? 6 : 4}
        spellCheck={f.type !== 'code'}
        value={String(v)}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  if (f.type === 'number')
    return <input type="number" value={String(v)} onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))} />
  if (f.type === 'list')
    return (
      <input
        value={Array.isArray(value) ? value.join(', ') : String(v)}
        placeholder="comma separated"
        onChange={(e) => onChange(e.target.value.split(',').map((s) => s.trimStart()))}
        onBlur={(e) => onChange(e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
      />
    )
  return <input value={String(v)} onChange={(e) => onChange(e.target.value)} />
}

function Row({ label, help, required, children }: { label: string; help?: string; required?: boolean; children: React.ReactNode }) {
  return (
    <label className="row">
      <span className="row-label">
        {label}
        {required && <em> *</em>}
      </span>
      {children}
      {help && <span className="help">{help}</span>}
    </label>
  )
}

function IdField({ id, onRename }: { id: string; onRename: (to: string) => string | null }) {
  const [draft, setDraft] = useState(id)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setDraft(id)
    setError(null)
  }, [id])
  return (
    <Row label="Id" help={error ?? 'Used as the variable name in generated code.'}>
      <input
        className={`mono ${error ? 'bad' : ''}`}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => setError(draft === id ? null : onRename(draft))}
        onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
      />
    </Row>
  )
}

type Props = {
  node: SNode | null
  edge: SEdge | null
  meta: Meta
  nodes: SNode[]
  setMeta: (m: Meta) => void
  patchNode: (id: string, patch: Partial<SNode['data']>) => void
  renameNode: (from: string, to: string) => string | null
  patchEdge: (id: string, patch: { label?: string; description?: string }) => void
  remove: () => void
  onPreview: (id: string) => void
}

export function Inspector({ node, edge, meta, nodes, setMeta, patchNode, renameNode, patchEdge, remove, onPreview }: Props) {
  const { defs, issues } = useContext(Ctx)

  if (node) {
    const def = defs[node.data.type]
    const mine = issues.filter((i) => i.node === node.id)
    return (
      <aside className="inspector">
        <h2>
          {def.label} <span className={`tag ${def.kind}`}>{def.kind === 'dataflow' ? 'Spark' : 'driver'}</span>
        </h2>
        <p className="muted">{def.description}</p>
        {mine.map((i, k) => (
          <div key={k} className={`issue ${i.level}`}>
            {i.message}
          </div>
        ))}
        <IdField id={node.id} onRename={(to) => renameNode(node.id, to)} />
        <Row label="Label">
          <input value={node.data.label ?? ''} onChange={(e) => patchNode(node.id, { label: e.target.value })} />
        </Row>
        {def.fields.map((f) => (
          <Row key={f.name} label={f.label} help={f.help} required={f.required}>
            <FieldInput
              f={f}
              value={node.data.config[f.name]}
              onChange={(v) => patchNode(node.id, { config: { ...node.data.config, [f.name]: v } })}
            />
          </Row>
        ))}
        <div className="actions">
          {def.kind === 'dataflow' && <button onClick={() => onPreview(node.id)}>Preview 1,000 rows</button>}
          <button className="danger" onClick={remove}>
            Delete
          </button>
        </div>
      </aside>
    )
  }

  if (edge) {
    const src = nodes.find((n) => n.id === edge.source)
    const sd = src ? defs[src.data.type] : undefined
    return (
      <aside className="inspector">
        <h2>Connection</h2>
        <p className="muted mono">
          {edge.source} → {edge.target}
        </p>
        <p className="muted">
          {edge.sourceHandle === 'tool'
            ? 'Tool: the agent may run the Spark steps starting here, as often as it needs. They do not run unless it calls them. Use :name in their SQL to give the tool arguments.'
            : sd?.kind === 'dataflow'
            ? 'Carries data. If the two ends run in different steps, the data is written to a table and passed by reference.'
            : 'Controls order only: the target runs after the source.'}
        </p>
        {sd?.free_branches && (
          <Row label="Branch label" required help="The router's expression must evaluate to this value to take this branch.">
            <input autoFocus className="mono" value={edge.data?.label ?? ''} onChange={(e) => patchEdge(edge.id, { label: e.target.value })} />
          </Row>
        )}
        {edge.sourceHandle === 'tool' && (
          <Row label="Tool description" help="Tells the model what this tool does and when to use it.">
            <textarea autoFocus rows={3} value={edge.data?.description ?? ''} onChange={(e) => patchEdge(edge.id, { description: e.target.value })} />
          </Row>
        )}
        {sd && sd.branches.length > 0 && <Row label="Branch">{<input disabled value={edge.sourceHandle ?? ''} />}</Row>}
        <div className="actions">
          <button className="danger" onClick={remove}>
            Delete
          </button>
        </div>
      </aside>
    )
  }

  const s = meta.settings
  const set = (patch: Partial<Meta['settings']>) => setMeta({ ...meta, settings: { ...s, ...patch } })
  return (
    <aside className="inspector">
      <h2>Pipeline</h2>
      <p className="muted">Select a node or a connection to edit it.</p>
      {issues
        .filter((i) => !i.node)
        .map((i, k) => (
          <div key={k} className={`issue ${i.level}`}>
            {i.message}
          </div>
        ))}
      <Row label="Name" required help="Becomes the Python package and the job name.">
        <input className="mono" value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} />
      </Row>
      <Row label="Description">
        <textarea rows={3} value={meta.description} onChange={(e) => setMeta({ ...meta, description: e.target.value })} />
      </Row>
      <Row label="Catalog">
        <input className="mono" value={s.catalog} onChange={(e) => set({ catalog: e.target.value })} />
      </Row>
      <Row label="Staging schema" help="Where step outputs, the event log and approvals are written.">
        <input className="mono" value={s.staging_schema} onChange={(e) => set({ staging_schema: e.target.value })} />
      </Row>
      <Row label="Checkpoint store" help="Where a paused run keeps its state. Postgres means Lakebase on Databricks.">
        <select value={s.checkpoint.kind} onChange={(e) => set({ checkpoint: { ...s.checkpoint, kind: e.target.value } })}>
          <option>postgres</option>
          <option>sqlite</option>
          <option>memory</option>
        </select>
      </Row>
      {s.checkpoint.kind === 'postgres' && (
        <Row label="Connection secret" help="scope/key of a secret holding the connection string.">
          <input
            className="mono"
            value={s.checkpoint.dsn_secret ?? ''}
            onChange={(e) => set({ checkpoint: { ...s.checkpoint, dsn_secret: e.target.value } })}
          />
        </Row>
      )}
    </aside>
  )
}
