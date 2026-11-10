import { useContext, useState } from 'react'
import type { Approval, CompileResult, PreviewResult } from './api'
import { Ctx, REGION_COLORS } from './graph'

export type Tab = 'plan' | 'issues' | 'code' | 'preview' | 'approvals'

type Props = {
  tab: Tab
  setTab: (t: Tab) => void
  compiled: CompileResult | null
  preview: { node: string; loading: boolean; result?: PreviewResult; error?: string } | null
  approvals: { loading: boolean; items?: Approval[]; error?: string } | null
  loadApprovals: () => void
  resolve: (thread: string, approved: boolean) => void
  select: (id: string) => void
}

export function Panel({ tab, setTab, compiled, preview, approvals, loadApprovals, resolve, select }: Props) {
  const { plan, issues, regionIndex } = useContext(Ctx)
  const [file, setFile] = useState<string | null>(null)
  const files = compiled?.files ?? {}
  const names = Object.keys(files).sort((a, b) => Number(b.endsWith('.py')) - Number(a.endsWith('.py')) || a.localeCompare(b))
  const current = file && files[file] !== undefined ? file : (names.find((n) => n.endsWith('transforms.py')) ?? names[0])
  const errors = issues.filter((i) => i.level === 'error').length

  const tabs: [Tab, string][] = [
    ['plan', 'Execution plan'],
    ['issues', `Issues${issues.length ? ` (${issues.length})` : ''}`],
    ['code', 'Generated code'],
    ['preview', 'Preview'],
    ['approvals', 'Approvals'],
  ]

  return (
    <section className="panel">
      <nav className="tabs">
        {tabs.map(([id, label]) => (
          <button key={id} className={`${tab === id ? 'active' : ''} ${id === 'issues' && errors ? 'has-errors' : ''}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>
      <div className="panel-body">
        {tab === 'plan' &&
          (plan ? (
            <ol className="plan">
              {plan.units.map((u) => {
                const outgoing = plan.edges.filter((e) => e.source === u.id)
                return (
                  <li key={u.id}>
                    <span
                      className="swatch"
                      style={{ background: u.kind === 'region' ? REGION_COLORS[regionIndex[u.id] % REGION_COLORS.length] : 'var(--control)' }}
                    />
                    <div>
                      <strong className="mono">{u.id}</strong>{' '}
                      <span className="muted">
                        {u.kind === 'region' ? 'one Spark plan' : 'runs on the driver'}
                      </span>
                      {u.kind === 'region' && (
                        <div className="chain">
                          {u.nodes.map((n, i) => (
                            <span key={n}>
                              {i > 0 && ' → '}
                              <a onClick={() => select(n)}>{n}</a>
                            </span>
                          ))}
                        </div>
                      )}
                      {Object.entries(u.materialize).map(([n, why]) => (
                        <div key={n} className="muted small">
                          ▣ writes <span className="mono">{n}</span>: {why}
                        </div>
                      ))}
                      {outgoing.length > 0 && (
                        <div className="muted small">
                          then{' '}
                          {outgoing.map((e, i) => (
                            <span key={i}>
                              {i > 0 && ', '}
                              {e.label && <em>[{e.label}] </em>}
                              <span className="mono">{e.target}</span>
                              {e.back && ' (loop)'}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </li>
                )
              })}
            </ol>
          ) : (
            <p className="muted pad">Fix the errors to see how this pipeline will run.</p>
          ))}

        {tab === 'issues' &&
          (issues.length ? (
            <ul className="issues">
              {issues.map((i, k) => (
                <li key={k} className={i.level} onClick={() => i.node && select(i.node)}>
                  <b>{i.level}</b> {i.node && <span className="mono">{i.node}</span>} {i.message}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted pad">No issues.</p>
          ))}

        {tab === 'code' &&
          (names.length ? (
            <div className="code">
              <ul>
                {names.map((n) => (
                  <li key={n} className={n === current ? 'active' : ''} onClick={() => setFile(n)}>
                    {n.replace(/^src\//, '')}
                  </li>
                ))}
                {compiled?.written && <li className="note">Written to {compiled.written}/</li>}
              </ul>
              <pre>{current ? files[current] : ''}</pre>
            </div>
          ) : (
            <p className="muted pad">Press Compile to generate PySpark, the LangGraph workflow and a Databricks Asset Bundle.</p>
          ))}

        {tab === 'preview' &&
          (!preview ? (
            <p className="muted pad">Select a Spark node and press “Preview 1,000 rows”.</p>
          ) : preview.loading ? (
            <p className="muted pad">Running {preview.node} on a sample…</p>
          ) : preview.error ? (
            <p className="issue error pad">{preview.error}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {preview.result!.columns.map((c) => (
                      <th key={c.name}>
                        {c.name} <span className="muted">{c.type}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.result!.rows.map((r, i) => (
                    <tr key={i}>
                      {preview.result!.columns.map((c) => (
                        <td key={c.name}>{r[c.name] === null ? <span className="muted">null</span> : String(r[c.name])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

        {tab === 'approvals' && (
          <div className="pad">
            <button onClick={loadApprovals} disabled={approvals?.loading}>
              {approvals?.loading ? 'Loading…' : 'Refresh'}
            </button>
            {approvals?.error && <p className="issue error">{approvals.error}</p>}
            {approvals?.items?.length === 0 && <p className="muted">No runs are waiting for approval.</p>}
            {approvals?.items?.map((a) => (
              <div key={a.thread_id + a.node} className="approval">
                <div>
                  <strong>{a.message ?? a.node}</strong>
                  <div className="muted small mono">
                    run {a.thread_id} · {new Date(a.ts * 1000).toLocaleString()}
                  </div>
                </div>
                <button onClick={() => resolve(a.thread_id, true)}>Approve</button>
                <button className="danger" onClick={() => resolve(a.thread_id, false)}>
                  Reject
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
