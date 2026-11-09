// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { useState } from 'react'
import { console_, plain, type Audit } from './api'

const SUGGESTED = ['Why was I refunded?', 'How much did PayPal charge me, and when?', 'Have the annotators received their pay?']

/** Second pair of eyes: PayPal's own records, read through the PayPal Agent Toolkit, against Blindspot's ledger. */
export function MoneyCheck({ jobId, funded }: { jobId: string; funded: boolean }) {
  const [audit, setAudit] = useState<Audit | null>(null)
  const [busy, setBusy] = useState(false)
  const [q, setQ] = useState('')
  const [chat, setChat] = useState<{ q: string; a: string; tools: string[] }[]>([])
  const [asking, setAsking] = useState(false)
  const [err, setErr] = useState('')

  const runAudit = async () => {
    setBusy(true)
    setErr('')
    try {
      setAudit(await console_.audit(jobId))
    } catch (x) {
      setErr((x as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const ask = async (question: string) => {
    if (!question.trim()) return
    setAsking(true)
    setErr('')
    try {
      const r = await console_.ask(jobId, question)
      setChat((c) => [...c, { q: question, a: r.answer, tools: r.tools }])
      setQ('')
    } catch (x) {
      setErr((x as Error).message)
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="label">Check with PayPal</div>
          <div className="mt-1 text-xs text-mute">PayPal's own records, read through the PayPal Agent Toolkit, compared line by line with Blindspot's ledger.</div>
        </div>
        <button className="btn-ghost shrink-0" disabled={!funded || busy} onClick={runAudit}>
          {busy ? 'Reading PayPal…' : 'Run audit'}
        </button>
      </div>
      {audit && (
        <div className="mt-3 overflow-hidden rounded-lg border border-line">
          <div className={`px-3 py-2 text-sm font-medium ${audit.ok ? 'bg-green-500/10 text-green-300' : 'bg-red-500/10 text-red-300'}`}>
            {audit.ok ? `All ${audit.checks.length} checks match PayPal's records` : 'PayPal and the ledger disagree — see the red lines'}
            <span className="mono ml-2 text-[10px] text-mute">order {audit.order_id}</span>
          </div>
          <table className="w-full text-xs">
            <thead className="text-left text-mute">
              <tr>
                <th className="px-3 py-1.5 font-medium" />
                <th className="px-3 py-1.5 font-medium">Check</th>
                <th className="px-3 py-1.5 font-medium">Blindspot</th>
                <th className="px-3 py-1.5 font-medium">PayPal says</th>
              </tr>
            </thead>
            <tbody>
              {audit.checks.map((c, i) => (
                <tr key={i} className="border-t border-line">
                  <td className={`px-3 py-1.5 ${c.ok ? 'text-green-400' : 'text-red-400'}`}>{c.ok ? '✓' : '✗'}</td>
                  <td className="px-3 py-1.5">{c.check}</td>
                  <td className="mono px-3 py-1.5 text-slate-300">{c.blindspot}</td>
                  <td className="mono px-3 py-1.5 text-slate-300">{c.paypal}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-4 space-y-2">
        <div className="label">Ask about your money</div>
        {chat.map((m, i) => (
          <div key={i} className="space-y-1">
            <div className="text-sm text-slate-300">“{m.q}”</div>
            <div className="whitespace-pre-line rounded-lg border border-line bg-ink/60 p-3 text-sm leading-relaxed">{plain(m.a)}</div>
            <div className="text-[10px] text-mute">Gemini looked up: {m.tools.length ? [...new Set(m.tools)].map((t) => (t === 'paypal_order' ? 'PayPal order (Agent Toolkit)' : 'Blindspot ledger')).join(', ') : 'nothing'} · read-only, cannot move money</div>
          </div>
        ))}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            ask(q)
          }}
        >
          <input className="input" placeholder="e.g. Why was I refunded?" value={q} onChange={(e) => setQ(e.target.value)} disabled={!funded || asking} maxLength={500} />
          <button className="btn-primary shrink-0" disabled={!funded || asking || !q.trim()}>
            {asking ? 'Checking…' : 'Ask'}
          </button>
        </form>
        {chat.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTED.map((s) => (
              <button key={s} className="rounded-full border border-line px-2.5 py-1 text-xs text-mute hover:text-slate-200" disabled={!funded || asking} onClick={() => ask(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
        {err && <p className="text-sm text-red-300">{err}</p>}
      </div>
    </div>
  )
}
