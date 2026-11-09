// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { PayPalButtons, PayPalScriptProvider } from '@paypal/react-paypal-js'
import { AllCommunityModule, ModuleRegistry, themeQuartz, colorSchemeDarkBlue, type ColDef } from 'ag-grid-community'
import { AgGridReact } from 'ag-grid-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, console_, operatorKey, pct, pub, usd, type Experiment, type JobDetail, type JobEvent, type Job, type Me } from './api'
import { LineChart, type Series } from './Chart'
import { describe, type FeedLine } from './feed'
import { MoneyCheck } from './MoneyCheck'

ModuleRegistry.registerModules([AllCommunityModule])
const gridTheme = themeQuartz.withPart(colorSchemeDarkBlue).withParams({
  backgroundColor: '#111a2e',
  headerBackgroundColor: '#0d1526',
  borderColor: '#1f2b45',
  fontSize: 12,
  rowHeight: 34,
  headerHeight: 34,
})

const TONE: Record<FeedLine['tone'], string> = {
  agent: 'border-l-blue-400',
  money: 'border-l-emerald-400',
  good: 'border-l-green-500',
  warn: 'border-l-amber-400',
  bad: 'border-l-red-500',
  info: 'border-l-slate-500',
}
const TONE_LABEL: Record<FeedLine['tone'], string> = { agent: 'AGENT', money: 'PAYPAL', good: 'OK', warn: 'CHECK', bad: 'STOP', info: 'INFO' }

function StatusPill({ status }: { status: string }) {
  const c =
    status === 'ACTIVE' ? 'bg-green-500/15 text-green-300' : status === 'CLOSED' ? 'bg-slate-500/20 text-slate-300' : status === 'PAYMENT_PROBLEM' ? 'bg-red-500/15 text-red-300' : 'bg-amber-500/15 text-amber-300'
  return <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${c}`}>{status.replace('_', ' ')}</span>
}

function KeyGate({ onReady }: { onReady: () => void }) {
  const [k, setK] = useState('')
  return (
    <div className="flex h-full items-center justify-center p-6">
      <form
        className="card w-full max-w-sm space-y-3 p-6"
        onSubmit={(e) => {
          e.preventDefault()
          operatorKey.set(k)
          onReady()
        }}
      >
        <h1 className="text-lg font-semibold">Blindspot console</h1>
        <p className="text-sm text-mute">Enter the operator key to manage learning jobs.</p>
        <input className="input" type="password" value={k} onChange={(e) => setK(e.target.value)} placeholder="Operator key" autoFocus />
        <button className="btn-primary w-full">Open console</button>
        <a className="block text-center text-xs text-mute underline" href="/label">
          I'm an annotator
        </a>
      </form>
    </div>
  )
}

function NewJob({ onCreated }: { onCreated: (id: string) => void }) {
  const [f, setF] = useState({ name: 'Support intent classifier', budget: '60.00', price: '0.50', worker_pay: '0.40', round_size: 40, target_accuracy: 0.8, strategy: 'blindspot' })
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const set = (k: string, v: string | number) => setF((x) => ({ ...x, [k]: v }))
  return (
    <form
      className="space-y-3"
      onSubmit={async (e) => {
        e.preventDefault()
        setBusy(true)
        setErr('')
        try {
          const r = await console_.createJob({ ...f, seed_count: 200, min_gain_per_10: 0.25 })
          onCreated(r.job_id)
        } catch (x) {
          setErr((x as Error).message)
        } finally {
          setBusy(false)
        }
      }}
    >
      <div>
        <div className="label mb-1">Model to improve</div>
        <input className="input" value={f.name} onChange={(e) => set('name', e.target.value)} />
        <p className="mt-1 text-xs text-mute">Customer-support intent classifier · 77 intents · starts from 200 labelled examples</p>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <label>
          <div className="label mb-1">Budget</div>
          <input className="input mono" value={f.budget} onChange={(e) => set('budget', e.target.value)} />
        </label>
        <label>
          <div className="label mb-1">Per label</div>
          <input className="input mono" value={f.price} onChange={(e) => set('price', e.target.value)} />
        </label>
        <label>
          <div className="label mb-1">To worker</div>
          <input className="input mono" value={f.worker_pay} onChange={(e) => set('worker_pay', e.target.value)} />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <label>
          <div className="label mb-1">Round size</div>
          <input className="input mono" type="number" value={f.round_size} onChange={(e) => set('round_size', Number(e.target.value))} />
        </label>
        <label>
          <div className="label mb-1">Target</div>
          <input className="input mono" type="number" step="0.01" value={f.target_accuracy} onChange={(e) => set('target_accuracy', Number(e.target.value))} />
        </label>
        <label>
          <div className="label mb-1">Strategy</div>
          <select className="input" value={f.strategy} onChange={(e) => set('strategy', e.target.value)}>
            <option value="blindspot">Blind spots</option>
            <option value="random">Random (baseline)</option>
          </select>
        </label>
      </div>
      {err && <p className="text-sm text-red-300">{err}</p>}
      <button className="btn-primary w-full" disabled={busy}>
        {busy ? 'Measuring the starting model…' : 'Create learning job'}
      </button>
    </form>
  )
}

function Funding({ job, clientId, onDone }: { job: Job; clientId: string | null; onDone: () => void }) {
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-300">
        PayPal places a <b>hold</b> of {usd(job.budget_cents)}. You are charged only for labels that pass the quality check; the rest is released when the job ends.
      </p>
      {clientId && (
        <PayPalScriptProvider options={{ clientId, intent: 'authorize', currency: 'USD', components: 'buttons' }}>
          <div className="rounded-lg bg-white p-2">
            <PayPalButtons
              style={{ layout: 'horizontal', tagline: false, height: 40 }}
              fundingSource="paypal"
              createOrder={async () => (await console_.walletOrder(job.id)).order_id}
              onApprove={async () => {
                try {
                  await console_.walletAuthorize(job.id)
                  onDone()
                } catch (x) {
                  setErr((x as Error).message)
                }
              }}
              onError={(e) => setErr(String(e))}
            />
          </div>
        </PayPalScriptProvider>
      )}
      <button
        className="btn-ghost w-full"
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          setErr('')
          try {
            await console_.fundCard(job.id)
            onDone()
          } catch (x) {
            setErr((x as Error).message)
          } finally {
            setBusy(false)
          }
        }}
      >
        {busy ? 'Placing hold…' : 'Use a sandbox test card instead'}
      </button>
      {err && <p className="text-sm text-red-300">{err}</p>}
    </div>
  )
}

function AgentControls({ d, onChange }: { d: JobDetail; onChange: () => void }) {
  const [rounds, setRounds] = useState(6)
  const [wait, setWait] = useState(0)
  const [err, setErr] = useState('')
  const act = async (fn: () => Promise<unknown>) => {
    setErr('')
    try {
      await fn()
    } catch (x) {
      setErr((x as Error).message)
    }
    onChange()
  }
  const j = d.job
  if (j.status === 'CLOSED') {
    return <p className="text-sm text-mute">Job closed. The unused hold was released to the customer.</p>
  }
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <label>
          <div className="label mb-1">Max rounds</div>
          <input className="input mono" type="number" min={1} max={50} value={rounds} onChange={(e) => setRounds(Number(e.target.value))} />
        </label>
        <label>
          <div className="label mb-1">Time for people (s)</div>
          <input className="input mono" type="number" min={0} max={1800} value={wait} onChange={(e) => setWait(Number(e.target.value))} />
        </label>
      </div>
      <p className="text-xs text-mute">{wait > 0 ? 'People on the annotator page get tasks too; unanswered tasks expire unpaid and uncharged.' : 'Only simulated annotators take part (they answer instantly).'}</p>
      <div className="flex gap-2">
        {d.autopilot ? (
          <button className="btn-ghost flex-1" onClick={() => act(() => console_.stop(j.id))}>
            Pause agent
          </button>
        ) : (
          <button className="btn-primary flex-1" disabled={j.status !== 'ACTIVE'} onClick={() => act(() => console_.autopilot(j.id, { rounds, wait_seconds: wait, close_when_done: true }))}>
            Run buyer agent
          </button>
        )}
        <button className="btn-danger" disabled={j.status !== 'ACTIVE' && j.status !== 'PAYMENT_PROBLEM'} onClick={() => act(() => console_.close(j.id))}>
          Close &amp; release
        </button>
      </div>
      <button className="w-full text-xs text-mute underline" onClick={() => act(() => console_.reconcile(j.id))}>
        Check payment status with PayPal
      </button>
      {err && <p className="text-sm text-red-300">{err}</p>}
    </div>
  )
}

function Money({ d }: { d: JobDetail }) {
  const j = d.job
  const cents = (states: string[]) =>
    d.settlements.filter((s) => s.kind === 'round' && states.includes(s.payout_state)).reduce((a, s) => a + Math.round(Number(s.payout) * 100), 0)
  const sent = cents(['SUBMITTED', 'ITEMS_PENDING', 'ITEM_FAILED', 'DONE'])
  const received = cents(['DONE'])
  const released = j.status === 'CLOSED' ? j.budget_cents - j.captured_cents : 0
  const cells: [string, string, string][] = [
    ['Held by PayPal', j.authorization_id ? usd(j.budget_cents) : '—', 'text-slate-100'],
    ['Charged', usd(j.captured_cents), 'text-emerald-300'],
    ['Refunded', usd(j.refunded_cents), 'text-amber-300'],
    [`Sent to people · ${usd(received)} received`, usd(sent), 'text-sky-300'],
    [j.status === 'CLOSED' ? 'Released' : 'Still held', usd(j.status === 'CLOSED' ? released : j.authorization_id ? j.budget_cents - j.captured_cents : 0), 'text-slate-300'],
  ]
  return (
    <div className="grid grid-cols-5 divide-x divide-line">
      {cells.map(([k, v, c]) => (
        <div key={k} className="px-4 py-3">
          <div className="label">{k}</div>
          <div className={`mono mt-1 text-xl font-semibold ${c}`}>{v}</div>
        </div>
      ))}
    </div>
  )
}

function Feed({ lines }: { lines: FeedLine[] }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'nearest' })
  }, [lines.length])
  return (
    <div className="h-full space-y-2 overflow-y-auto p-3">
      {lines.length === 0 && <p className="p-4 text-center text-sm text-mute">The agent's decisions and every PayPal step appear here.</p>}
      {lines.map((l) => (
        <div key={l.seq} className={`rounded-md border border-line border-l-4 bg-ink/60 px-3 py-2 ${TONE[l.tone]}`}>
          <div className="flex items-baseline justify-between gap-2">
            <div className="text-sm font-medium">{l.title}</div>
            <div className="shrink-0 text-[10px] text-mute">
              {TONE_LABEL[l.tone]} · {new Date(l.at * 1000).toLocaleTimeString()}
            </div>
          </div>
          {l.body && <div className="mt-1 text-xs leading-relaxed text-slate-300">{l.body}</div>}
          {l.ids && <div className="mono mt-1 text-[10px] text-mute">{l.ids.join(' · ')}</div>}
        </div>
      ))}
      <div ref={end} />
    </div>
  )
}

function JobView({ id, clientId, experiment, onJobChange }: { id: string; clientId: string | null; experiment: Experiment | null; onJobChange: () => void }) {
  const [d, setD] = useState<JobDetail | null>(null)
  const [lines, setLines] = useState<FeedLine[]>([])
  const refresh = useCallback(() => console_.job(id).then(setD), [id])

  useEffect(() => {
    setLines([])
    refresh()
    const seen = new Set<number>()
    const handler = (data: string) => {
      const e = JSON.parse(data) as JobEvent
      if (seen.has(e.seq)) return
      seen.add(e.seq)
      const line = describe(e)
      if (line) setLines((xs) => [...xs, line])
      refresh()
      if (e.kind === 'budget_held' || e.kind === 'budget_released') onJobChange()
    }
    const stop = console_.events(id, 0, handler)
    const t = setInterval(refresh, 4000)
    return () => {
      stop()
      clearInterval(t)
    }
  }, [id, refresh, onJobChange])

  const roundCols = useMemo<ColDef[]>(
    () => [
      { field: 'number', headerName: '#', width: 56 },
      { field: 'status', width: 100 },
      { field: 'labels_planned', headerName: 'Bought', width: 82 },
      { headerName: 'Accepted', width: 90, valueGetter: (p) => p.data.settlement?.accepted ?? '' },
      { headerName: 'Rejected', width: 90, valueGetter: (p) => p.data.settlement?.rejected ?? '' },
      { headerName: 'Accuracy', width: 130, valueGetter: (p) => `${pct(p.data.val_before)} → ${pct(p.data.val_after)}` },
      { headerName: 'Charged', width: 90, valueGetter: (p) => (p.data.settlement ? `$${p.data.settlement.charge}` : '') },
      { headerName: 'Charge', width: 92, valueGetter: (p) => p.data.settlement?.capture_state ?? '' },
      { headerName: 'Paid out', width: 90, valueGetter: (p) => (p.data.settlement ? `$${p.data.settlement.payout}` : '') },
      { headerName: 'Payout', width: 120, valueGetter: (p) => p.data.settlement?.payout_state ?? '' },
      { headerName: 'PayPal capture', flex: 1, minWidth: 150, valueGetter: (p) => p.data.settlement?.capture_id ?? '', cellClass: 'mono' },
      { headerName: 'Payout batch', flex: 1, minWidth: 130, valueGetter: (p) => p.data.settlement?.payout_batch_id ?? '', cellClass: 'mono' },
      { headerName: 'Blind spots targeted', flex: 2, minWidth: 260, valueGetter: (p) => (p.data.confusions ?? []).map((c: { a: string; b: string }) => `${c.a} ↔ ${c.b}`).join('; ') },
    ],
    [],
  )
  const annCols = useMemo<ColDef[]>(
    () => [
      { field: 'name', flex: 1, minWidth: 120, tooltipField: 'name' },
      { field: 'gold', headerName: 'Tests', width: 62 },
      { field: 'earned', headerName: 'Earned', width: 74, valueFormatter: (p) => `$${p.value}` },
      {
        field: 'banned',
        headerName: '',
        cellDataType: 'text',
        width: 72,
        valueFormatter: (p) => (p.value ? 'banned' : 'ok'),
        cellClass: (p) => (p.value ? 'text-red-300 font-semibold' : 'text-green-300'),
      },
    ],
    [],
  )

  if (!d) return <div className="p-8 text-mute">Loading…</div>
  const j = d.job
  const live: Series[] = [
    {
      name: j.strategy === 'random' ? 'This job (random buying)' : 'This job (blind-spot buying)',
      color: '#3b82f6',
      points: d.curve.map((c) => ({ x: Number(c.charged), y: c.validation, note: `${c.labels} labels` })),
    },
  ]
  const bench: Series[] = experiment
    ? [
        { name: 'Blind-spot buying', color: '#3b82f6', points: experiment.points.map((p) => ({ x: p.dollars, y: p.blindspot, note: `${p.labels} labels` })) },
        { name: 'Uncertainty only', color: '#f59e0b', dashed: true, points: experiment.points.map((p) => ({ x: p.dollars, y: p.uncertain, note: `${p.labels} labels` })) },
        { name: 'Random buying', color: '#94a3b8', dashed: true, points: experiment.points.map((p) => ({ x: p.dollars, y: p.random, note: `${p.labels} labels` })) },
      ]
    : []

  return (
    <div className="grid h-full grid-cols-[340px_1fr_380px] gap-4 overflow-hidden p-4">
      <div className="space-y-4 overflow-y-auto">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div className="font-semibold">{j.name}</div>
            <StatusPill status={j.status} />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-y-1 text-xs text-mute">
            <span>Budget</span>
            <span className="mono text-right text-slate-200">{usd(j.budget_cents)}</span>
            <span>Customer pays per label</span>
            <span className="mono text-right text-slate-200">{usd(j.price_cents)}</span>
            <span>Annotator earns per label</span>
            <span className="mono text-right text-slate-200">{usd(j.worker_cents)}</span>
            <span>Target accuracy</span>
            <span className="mono text-right text-slate-200">{pct(j.target_accuracy)}</span>
            <span>Strategy</span>
            <span className="text-right text-slate-200">{j.strategy === 'random' ? 'Random (baseline)' : 'Blind spots'}</span>
          </div>
          {j.authorization_id && <div className="mono mt-3 truncate text-[10px] text-mute">authorization {j.authorization_id}</div>}
        </div>
        <div className="card p-4">
          <div className="label mb-3">{j.status === 'DRAFT' || j.status === 'FUNDING' ? '1 · Fund the learning budget' : '2 · Buyer agent'}</div>
          {j.status === 'DRAFT' || j.status === 'FUNDING' ? <Funding job={j} clientId={clientId} onDone={refresh} /> : <AgentControls d={d} onChange={refresh} />}
        </div>
        <div className="card overflow-hidden">
          <div className="label px-4 pt-3">Annotators</div>
          <div className="h-56 p-2">
            <AgGridReact theme={gridTheme} rowData={d.annotators} columnDefs={annCols} getRowId={(p) => p.data.id} />
          </div>
        </div>
      </div>

      <div className="flex min-w-0 flex-col gap-4 overflow-y-auto">
        <div className="card">
          <Money d={d} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="card p-4">
            <div className="label">This job · accuracy vs dollars charged</div>
            <div className="mt-1 text-xs text-mute">Validation accuracy after each paid round (customer's held-out set)</div>
            <LineChart series={live} xLabel="Dollars charged" yLabel="Accuracy" xFormat={(x) => `$${x.toFixed(0)}`} target={j.target_accuracy} height={240} />
          </div>
          <div className="card p-4">
            <div className="label">Benchmark · same budget, two buyers</div>
            <div className="mt-1 text-xs text-mute">Sealed test set, mean of {experiment?.seeds.length ?? 5} runs. Both targeted buyers beat random buying; blind spots add a little over plain uncertainty.</div>
            <LineChart series={bench} xLabel="Dollars spent on labels" yLabel="Accuracy" xFormat={(x) => `$${x.toFixed(0)}`} height={240} />
          </div>
        </div>
        <div className="card shrink-0 overflow-hidden">
          <div className="label px-4 pt-3">Rounds · every purchase and its PayPal records</div>
          <div className="h-[300px] p-2">
            <AgGridReact theme={gridTheme} rowData={d.rounds} columnDefs={roundCols} getRowId={(p) => p.data.id} />
          </div>
        </div>
        <MoneyCheck jobId={j.id} funded={!!j.paypal_order_id} />
      </div>

      <div className="card flex min-h-0 flex-col">
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <div className="label">Agent feed</div>
          {d.autopilot && <span className="animate-pulse text-xs text-blue-300">● working</span>}
        </div>
        <div className="min-h-0 flex-1">
          <Feed lines={lines} />
        </div>
      </div>
    </div>
  )
}

export function Console() {
  const [ready, setReady] = useState(false)
  const [needKey, setNeedKey] = useState(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const [sel, setSel] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [clientId, setClientId] = useState<string | null>(null)
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [people, setPeople] = useState<Me[]>([])

  const load = useCallback(async () => {
    try {
      const js = await console_.jobs()
      setJobs(js)
      setPeople(await console_.annotators())
      setNeedKey(false)
      setReady(true)
      setSel((s) => s ?? js[0]?.id ?? null)
    } catch (x) {
      if (x instanceof ApiError && x.status === 401) setNeedKey(true)
    }
  }, [])

  useEffect(() => {
    load()
    pub.config().then((c) => setClientId(c.paypal_client_id)).catch(() => undefined)
    pub.experiment().then(setExperiment).catch(() => undefined)
  }, [load])

  if (needKey) return <KeyGate onReady={load} />
  if (!ready) return <div className="p-8 text-mute">Loading…</div>

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-4 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-full bg-gradient-to-br from-blue-400 to-indigo-600 ring-2 ring-blue-300/30" />
          <span className="font-semibold">Blindspot Market</span>
          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">PAYPAL SANDBOX</span>
        </div>
        <select className="input max-w-xs py-1.5" value={sel ?? ''} onChange={(e) => { setSel(e.target.value); setCreating(false) }}>
          {jobs.length === 0 && <option value="">No jobs yet</option>}
          {jobs.map((j) => (
            <option key={j.id} value={j.id}>
              {j.name} · {j.strategy} · {j.status}
            </option>
          ))}
        </select>
        <button className="btn-ghost py-1.5" onClick={() => setCreating(true)}>
          New job
        </button>
        <div className="ml-auto flex items-center gap-4 text-xs text-mute">
          <span>{people.filter((p) => p.kind === 'human').length} people · {people.filter((p) => p.kind === 'simulated').length} simulated annotators</span>
          <a className="underline" href="/label" target="_blank" rel="noreferrer">
            Annotator page ↗
          </a>
        </div>
      </header>
      <main className="min-h-0 flex-1">
        {creating || !sel ? (
          <div className="mx-auto mt-10 max-w-md">
            <div className="card p-6">
              <h2 className="mb-1 text-lg font-semibold">New learning job</h2>
              <p className="mb-4 text-sm text-mute">Set a budget. The buyer agent finds what the model gets wrong, pays people to label exactly those messages, and stops when more labels stop paying off.</p>
              <NewJob
                onCreated={async (id) => {
                  setCreating(false)
                  setSel(id)
                  await load()
                }}
              />
            </div>
          </div>
        ) : (
          <JobView key={sel} id={sel} clientId={clientId} experiment={experiment} onJobChange={load} />
        )}
      </main>
    </div>
  )
}
