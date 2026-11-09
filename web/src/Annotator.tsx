// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, annotator, annotatorToken, pub, type Me, type Task } from './api'

function Join({ onJoined }: { onJoined: () => void }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [invite, setInvite] = useState(new URLSearchParams(location.search).get('invite') ?? '')
  const [needInvite, setNeedInvite] = useState(false)
  const [err, setErr] = useState('')
  useEffect(() => {
    pub.config().then((c) => setNeedInvite(c.invite_required)).catch(() => undefined)
  }, [])
  return (
    <form
      className="card space-y-4 p-6"
      onSubmit={async (e) => {
        e.preventDefault()
        setErr('')
        try {
          const r = await annotator.join({ name, paypal_email: email, invite })
          annotatorToken.set(r.token)
          onJoined()
        } catch (x) {
          setErr((x as Error).message)
        }
      }}
    >
      <div>
        <h1 className="text-xl font-semibold">Teach an AI, get paid</h1>
        <p className="mt-1 text-sm text-mute">
          Read short customer messages and pick what the customer wants. Accepted answers are paid to your PayPal account after each round. Some questions are hidden checks — careless answers are not paid.
        </p>
      </div>
      <label className="block">
        <div className="label mb-1">Your name</div>
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} required maxLength={60} />
      </label>
      <label className="block">
        <div className="label mb-1">PayPal email (sandbox account)</div>
        <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
      </label>
      {needInvite && (
        <label className="block">
          <div className="label mb-1">Invite code</div>
          <input className="input" value={invite} onChange={(e) => setInvite(e.target.value)} required />
        </label>
      )}
      {err && <p className="text-sm text-red-300">{err}</p>}
      <button className="btn-primary w-full py-3 text-base">Start labelling</button>
      <p className="text-center text-[11px] text-mute">PayPal sandbox demo — no real money moves.</p>
    </form>
  )
}

function TaskCard({ task, onAnswer, busy }: { task: Task; onAnswer: (intent: number) => void; busy: boolean }) {
  const [q, setQ] = useState('')
  const [searching, setSearching] = useState(false)
  useEffect(() => {
    setQ('')
    setSearching(false)
  }, [task.task_id])
  const matches = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? task.all_intents.filter((i) => i.label.toLowerCase().includes(s)).slice(0, 8) : []
  }, [q, task.all_intents])
  return (
    <div className="card space-y-4 p-5">
      <div className="label">What does this customer want?</div>
      <blockquote className="rounded-lg border border-line bg-ink p-4 text-lg leading-snug">“{task.text}”</blockquote>
      <div className="space-y-2">
        {task.options.map((o) => (
          <button key={o.id} disabled={busy} className="btn-ghost w-full justify-start py-3 text-left text-[15px]" onClick={() => onAnswer(o.id)}>
            {o.label}
          </button>
        ))}
      </div>
      {searching ? (
        <div className="space-y-2">
          <input className="input" autoFocus placeholder="Search all 77 intents…" value={q} onChange={(e) => setQ(e.target.value)} />
          {matches.map((o) => (
            <button key={o.id} disabled={busy} className="btn-ghost w-full justify-start text-left" onClick={() => onAnswer(o.id)}>
              {o.label}
            </button>
          ))}
        </div>
      ) : (
        <div className="flex gap-2">
          <button className="btn-ghost flex-1 text-xs" onClick={() => setSearching(true)}>
            Something else…
          </button>
          <button className="btn-ghost flex-1 text-xs" disabled={busy} onClick={() => onAnswer(-1)}>
            Can't tell
          </button>
        </div>
      )}
    </div>
  )
}

export function Annotator() {
  const [joined, setJoined] = useState(!!annotatorToken.get())
  const [task, setTask] = useState<Task | null>(null)
  const [me, setMe] = useState<Me | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const poll = useCallback(async () => {
    try {
      const r = await annotator.task()
      setTask(r.task)
      setMe(r.me)
    } catch (x) {
      if (x instanceof ApiError && x.status === 401) {
        annotatorToken.clear()
        setJoined(false)
      }
    }
  }, [])

  const hasTask = useRef(false)
  hasTask.current = task !== null
  useEffect(() => {
    if (!joined) return
    poll()
    const t = setInterval(() => {
      if (!hasTask.current) poll() // look for new work only while idle
    }, 2500)
    return () => clearInterval(t)
  }, [joined, poll])

  const answer = async (intent: number) => {
    if (!task) return
    setBusy(true)
    setErr('')
    try {
      const r = await annotator.answer(task.task_id, intent)
      setTask(r.task)
      setMe(r.me)
    } catch (x) {
      setErr((x as Error).message)
      await poll()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto flex min-h-full max-w-md flex-col gap-4 p-4">
      <header className="flex items-center gap-2 pt-2">
        <div className="h-6 w-6 rounded-full bg-gradient-to-br from-blue-400 to-indigo-600" />
        <span className="font-semibold">Blindspot</span>
        <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">SANDBOX</span>
      </header>
      {!joined ? (
        <Join onJoined={() => setJoined(true)} />
      ) : (
        <>
          {me && (
            <div className="card grid grid-cols-3 divide-x divide-line">
              <div className="p-3 text-center">
                <div className="label">Answered</div>
                <div className="mono text-xl font-semibold">{me.answered}</div>
              </div>
              <div className="p-3 text-center">
                <div className="label">Accepted</div>
                <div className="mono text-xl font-semibold text-green-300">{me.accepted}</div>
              </div>
              <div className="p-3 text-center">
                <div className="label">Earned</div>
                <div className="mono text-xl font-semibold text-emerald-300">${me.earned}</div>
              </div>
            </div>
          )}
          {me?.banned && <div className="card border-red-500/40 p-4 text-sm text-red-300">Your answers failed too many hidden checks. You are no longer receiving tasks.</div>}
          {task ? (
            <TaskCard task={task} onAnswer={answer} busy={busy} />
          ) : (
            <div className="card p-8 text-center">
              <div className="text-lg font-medium">No tasks right now</div>
              <p className="mt-1 text-sm text-mute">New messages arrive when the buyer agent starts its next round. This page checks automatically.</p>
              {me && me.answered > 0 && <p className="mt-3 text-sm text-slate-300">Accepted work is paid to your PayPal account when the round settles.</p>}
            </div>
          )}
          {err && <p className="text-sm text-red-300">{err}</p>}
          <button
            className="mt-auto text-xs text-mute underline"
            onClick={() => {
              annotatorToken.clear()
              setJoined(false)
            }}
          >
            Sign out
          </button>
        </>
      )}
    </div>
  )
}
