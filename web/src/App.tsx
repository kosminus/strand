import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './styles.css'
import { api, type Approval, type CheckResult, type CompileResult, type PreviewResult } from './api'
import { Ctx, EMPTY_META, defaults, freshId, fromPipeline, toPipeline, type Defs, type Meta, type SEdge, type SNode } from './graph'
import { Inspector } from './Inspector'
import { NodeCard } from './NodeCard'
import { Panel, type Tab } from './Panel'

const nodeTypes = { strand: NodeCard }
const ID_RE = /^[a-z][a-z0-9_]*$/

function Editor() {
  const [defs, setDefs] = useState<Defs>({})
  const [saved, setSaved] = useState<string[]>([])
  const [meta, setMeta] = useState<Meta>(EMPTY_META)
  const [nodes, setNodes, onNodesChange] = useNodesState<SNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<SEdge>([])
  const [check, setCheck] = useState<CheckResult>({ issues: [], plan: null })
  const [compiled, setCompiled] = useState<CompileResult | null>(null)
  const [preview, setPreview] = useState<{ node: string; loading: boolean; result?: PreviewResult; error?: string } | null>(null)
  const [approvals, setApprovals] = useState<{ loading: boolean; items?: Approval[]; error?: string } | null>(null)
  const [tab, setTab] = useState<Tab>('plan')
  const [status, setStatus] = useState<{ text: string; bad?: boolean } | null>(null)
  const [dirty, setDirty] = useState(false)
  const flow = useReactFlow<SNode, SEdge>()
  const wrap = useRef<HTMLDivElement>(null)

  const pipeline = useMemo(() => toPipeline(meta, nodes, edges, defs), [meta, nodes, edges, defs])
  const pipelineKey = JSON.stringify(pipeline)
  const selectedNode = nodes.filter((n) => n.selected).length === 1 ? nodes.find((n) => n.selected)! : null
  const selectedEdge = !selectedNode ? (edges.find((e) => e.selected) ?? null) : null

  const say = (text: string, bad = false) => {
    setStatus({ text, bad })
    if (!bad) setTimeout(() => setStatus((s) => (s?.text === text ? null : s)), 4000)
  }

  const open = useCallback(
    async (name: string, d: Defs) => {
      const g = fromPipeline(await api.get(name), d)
      setMeta(g.meta)
      setNodes(g.nodes)
      setEdges(g.edges)
      setCompiled(null)
      setPreview(null)
      setDirty(false)
      setTimeout(() => flow.fitView({ padding: 0.15 }), 50)
    },
    [flow, setEdges, setNodes],
  )

  useEffect(() => {
    ;(async () => {
      try {
        const d = Object.fromEntries((await api.nodeTypes()).map((t) => [t.type, t]))
        setDefs(d)
        const names = await api.list()
        setSaved(names)
        // ?pipeline=<name> opens that pipeline directly
        const wanted = new URLSearchParams(location.search).get('pipeline')
        if (names.length) await open(wanted && names.includes(wanted) ? wanted : names[0], d)
      } catch (e) {
        say(`Cannot reach the strand server: ${(e as Error).message}`, true)
      }
    })()
  }, [])

  // validate and re-plan shortly after every change
  useEffect(() => {
    if (!Object.keys(defs).length) return
    const t = setTimeout(() => api.check(JSON.parse(pipelineKey)).then(setCheck).catch(() => {}), 300)
    return () => clearTimeout(t)
  }, [pipelineKey, defs])

  const regionIndex = useMemo(() => {
    const out: Record<string, number> = {}
    check.plan?.units.filter((u) => u.kind === 'region').forEach((u, i) => (out[u.id] = i))
    return out
  }, [check.plan])

  const kindOf = useCallback((id: string) => defs[nodes.find((n) => n.id === id)?.data.type ?? '']?.kind, [defs, nodes])

  const isValidConnection = useCallback(
    (c: Connection | Edge) => {
      if (c.source === c.target) return false
      const s = nodes.find((n) => n.id === c.source)
      const t = nodes.find((n) => n.id === c.target)
      if (!s || !t) return false
      const sd = defs[s.data.type]
      const td = defs[t.data.type]
      if (edges.some((e) => e.source === c.source && e.target === c.target && e.sourceHandle === c.sourceHandle)) return false
      if (c.sourceHandle === 'tool') return td.kind === 'dataflow' && c.targetHandle === 'after'
      if (td.kind === 'control') return true
      if (sd.kind === 'control') return c.targetHandle === 'after'
      // data into a Spark node: must land on a free data port
      if (sd.type === 'sink') return false
      const port = td.inputs.find((p) => p.name === c.targetHandle)
      if (!port) return false
      const used = edges.filter((e) => e.target === c.target && e.targetHandle === c.targetHandle && kindOf(e.source) === 'dataflow').length
      return port.max === null || used < port.max
    },
    [defs, edges, nodes, kindOf],
  )

  const onConnect = useCallback(
    (c: Connection) => {
      const sd = defs[nodes.find((n) => n.id === c.source)!.data.type]
      const id = `e_${c.source}_${c.sourceHandle}_${c.target}_${c.targetHandle}`
      const taken = edges.filter((e) => e.source === c.source).length
      const edge: SEdge = { ...c, id, data: sd.free_branches ? { label: `branch_${taken + 1}` } : {}, selected: sd.free_branches }
      if (sd.free_branches) setNodes((ns) => ns.map((n) => ({ ...n, selected: false })))
      setEdges((es) => addEdge(edge, sd.free_branches ? es.map((e) => ({ ...e, selected: false })) : es))
      setDirty(true)
    },
    [defs, edges, nodes, setEdges, setNodes],
  )

  const addNode = (type: string, position?: { x: number; y: number }) => {
    const def = defs[type]
    const rect = wrap.current?.getBoundingClientRect()
    // clicked (not dropped): place left to right so new nodes do not stack
    const step = (nodes.length % 5) * 230
    const pos =
      position ??
      flow.screenToFlowPosition({ x: (rect?.left ?? 0) + 60 + step, y: (rect?.top ?? 0) + 80 + Math.floor(nodes.length / 5) * 130 })
    const id = freshId(type, nodes)
    setNodes((ns) => [
      ...ns.map((n) => ({ ...n, selected: false })),
      { id, type: 'strand', position: pos, selected: true, data: { type, config: defaults(def) } },
    ])
    setDirty(true)
  }

  const patchNode = (id: string, patch: Partial<SNode['data']>) => {
    setNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)))
    setDirty(true)
  }

  const renameNode = (from: string, to: string): string | null => {
    if (!ID_RE.test(to)) return 'Lowercase letters, digits and underscores, starting with a letter.'
    if (nodes.some((n) => n.id === to)) return 'Another node already has this id.'
    setNodes((ns) => ns.map((n) => (n.id === from ? { ...n, id: to } : n)))
    setEdges((es) => es.map((e) => ({ ...e, source: e.source === from ? to : e.source, target: e.target === from ? to : e.target })))
    setDirty(true)
    return null
  }

  const remove = () => {
    if (selectedNode) flow.deleteElements({ nodes: [{ id: selectedNode.id }] })
    else if (selectedEdge) flow.deleteElements({ edges: [{ id: selectedEdge.id }] })
    setDirty(true)
  }

  const select = (id: string) => {
    setNodes((ns) => ns.map((n) => ({ ...n, selected: n.id === id })))
    setEdges((es) => es.map((e) => ({ ...e, selected: false })))
    const n = nodes.find((x) => x.id === id)
    if (n) flow.setCenter(n.position.x + 100, n.position.y + 40, { zoom: 1, duration: 300 })
  }

  const save = async () => {
    try {
      await api.save(pipeline)
      setSaved(await api.list())
      setDirty(false)
      say(`Saved ${pipeline.name}`)
    } catch (e) {
      say((e as Error).message, true)
    }
  }

  const compile = async () => {
    try {
      const res = await api.compile(pipeline)
      setCheck({ issues: res.issues, plan: res.plan })
      if (res.written) {
        setCompiled(res)
        setTab('code')
        say(`Compiled to ${res.written}/`)
      } else {
        setTab('issues')
        say('Fix the errors before compiling.', true)
      }
    } catch (e) {
      say((e as Error).message, true)
    }
  }

  const runPreview = async (id: string) => {
    setTab('preview')
    setPreview({ node: id, loading: true })
    try {
      setPreview({ node: id, loading: false, result: await api.preview(pipeline, id) })
    } catch (e) {
      setPreview({ node: id, loading: false, error: (e as Error).message })
    }
  }

  const run = async () => {
    try {
      const r = await api.run(pipeline)
      say(`Started job run ${r.run_id}`)
    } catch (e) {
      say((e as Error).message, true)
    }
  }

  const loadApprovals = async () => {
    setApprovals({ loading: true })
    try {
      setApprovals({ loading: false, items: await api.approvals(pipeline) })
    } catch (e) {
      setApprovals({ loading: false, error: (e as Error).message })
    }
  }

  const resolve = async (thread: string, approved: boolean) => {
    try {
      const r = await api.resolve(pipeline, thread, approved)
      say(`${approved ? 'Approved' : 'Rejected'}; resuming as job run ${r.run_id}`)
      setApprovals((a) => (a ? { ...a, items: a.items?.filter((i) => i.thread_id !== thread) } : a))
    } catch (e) {
      say((e as Error).message, true)
    }
  }

  const shownEdges = useMemo(
    () =>
      edges.map((e) => {
        const control = kindOf(e.source) === 'control'
        const tool = e.sourceHandle === 'tool'
        return {
          ...e,
          label: tool ? 'tool' : e.data?.label,
          className: tool ? 'edge-tool' : control ? 'edge-control' : 'edge-data',
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        }
      }),
    [edges, kindOf],
  )

  const errors = check.issues.filter((i) => i.level === 'error').length
  const groups: [string, string][] = [
    ['dataflow', 'Spark'],
    ['control', 'Agents and control'],
  ]

  return (
    <Ctx.Provider value={{ defs, plan: check.plan, issues: check.issues, regionIndex }}>
      <div className="app">
        <header>
          <span className="logo">strand</span>
          <select
            value={saved.includes(meta.name) ? meta.name : ''}
            onChange={(e) => {
              if (e.target.value && (!dirty || confirm('Discard unsaved changes?'))) open(e.target.value, defs)
            }}
          >
            {!saved.includes(meta.name) && <option value="">{meta.name} (unsaved)</option>}
            {saved.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
          <button
            onClick={() => {
              if (dirty && !confirm('Discard unsaved changes?')) return
              setMeta({ ...EMPTY_META, name: `pipeline_${saved.length + 1}` })
              setNodes([])
              setEdges([])
              setCompiled(null)
              setDirty(false)
              setPreview(null)
              setTab('plan')
            }}
          >
            New
          </button>
          <span className={`status ${status?.bad ? 'bad' : ''}`}>{status?.text ?? (dirty ? 'Unsaved changes' : '')}</span>
          <span className={`pill ${errors ? 'bad' : 'ok'}`} onClick={() => setTab('issues')}>
            {errors ? `${errors} error${errors > 1 ? 's' : ''}` : 'Valid'}
          </span>
          <button onClick={save}>Save</button>
          <button className="primary" onClick={compile}>
            Compile
          </button>
          <button onClick={run} title="Start the deployed job in your Databricks workspace">
            Run
          </button>
        </header>

        <aside className="palette">
          {groups.map(([kind, title]) => (
            <div key={kind}>
              <h3>{title}</h3>
              {Object.values(defs)
                .filter((d) => d.kind === kind)
                .map((d) => (
                  <div
                    key={d.type}
                    className={`chip ${d.kind}`}
                    draggable
                    title={d.description}
                    onDragStart={(e) => e.dataTransfer.setData('application/strand', d.type)}
                    onClick={() => addNode(d.type)}
                  >
                    {d.label}
                  </div>
                ))}
            </div>
          ))}
          <div className="legend">
            <div>
              <i className="line data" /> data
            </div>
            <div>
              <i className="line control" /> order only
            </div>
            <div>
              <i className="line tool" /> agent tool
            </div>
            <div>▣ written to a table</div>
            <p>Spark nodes that share a colour run as one Spark plan.</p>
          </div>
        </aside>

        <main
          ref={wrap}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            const type = e.dataTransfer.getData('application/strand')
            if (type) addNode(type, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }))
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={shownEdges}
            nodeTypes={nodeTypes}
            onNodesChange={(c) => {
              onNodesChange(c)
              if (c.some((x) => x.type === 'remove' || (x.type === 'position' && x.dragging))) setDirty(true)
            }}
            onEdgesChange={(c) => {
              onEdgesChange(c as Parameters<typeof onEdgesChange>[0])
              if (c.some((x) => x.type === 'remove')) setDirty(true)
            }}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            minZoom={0.2}
            colorMode="system"
            fitView
            proOptions={{ hideAttribution: false }}
            deleteKeyCode={['Backspace', 'Delete']}
          >
            <Background gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
          {nodes.length === 0 && <div className="empty">Drag a node from the left, or click one, to start.</div>}
        </main>

        <Inspector
          node={selectedNode}
          edge={selectedEdge}
          meta={meta}
          nodes={nodes}
          setMeta={(m) => {
            setMeta(m)
            setDirty(true)
          }}
          patchNode={patchNode}
          renameNode={renameNode}
          patchEdge={(id, patch) => {
            setEdges((es) => es.map((e) => (e.id === id ? { ...e, data: { ...e.data, ...patch } } : e)))
            setDirty(true)
          }}
          remove={remove}
          onPreview={runPreview}
        />

        <Panel
          tab={tab}
          setTab={(t) => {
            setTab(t)
            if (t === 'approvals' && !approvals) loadApprovals()
          }}
          compiled={compiled}
          preview={preview}
          approvals={approvals}
          loadApprovals={loadApprovals}
          resolve={resolve}
          select={select}
        />
      </div>
    </Ctx.Provider>
  )
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Editor />
    </ReactFlowProvider>
  )
}
