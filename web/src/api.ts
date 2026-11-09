// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron

export type Job = {
  id: string
  name: string
  status: 'DRAFT' | 'FUNDING' | 'ACTIVE' | 'PAYMENT_PROBLEM' | 'CLOSING' | 'CLOSED'
  price_cents: number
  worker_cents: number
  budget_cents: number
  round_size: number
  target_accuracy: number
  min_gain_per_10: number
  strategy: 'blindspot' | 'random'
  paypal_order_id: string | null
  authorization_id: string | null
  hold_status: string | null
  captured_cents: number
  refunded_cents: number
  seed_count: number
  funded_at: number | null
  created_at: number
}

export type CurvePoint = { labels: number; validation: number; sealed: number; spent: string; charged: string; round?: number }

export type Settlement = {
  id: string
  kind: 'round' | 'revocation'
  round_id: string | null
  accepted: number
  rejected: number
  charge: string
  capture_id: string | null
  capture_state: string
  refund: string
  refund_id: string | null
  refund_state: string
  payout: string
  payout_batch_id: string | null
  payout_state: string
  error: string | null
}

export type Round = {
  id: string
  number: number
  status: string
  labels_planned: number
  explanation: string
  confusions: { a: string; b: string; count: number }[]
  val_before: number
  val_after: number | null
  sealed_after: number | null
  open_tasks: number
  settlement: Settlement | null
}

export type AnnotatorRow = { id: string; name: string; kind: string; banned: boolean; gold: string; trust: number; earned: string }

export type JobDetail = {
  job: Job
  curve: CurvePoint[]
  annotators: AnnotatorRow[]
  spent: string
  left: string
  rounds: Round[]
  settlements: Settlement[]
  autopilot: boolean
}

export type JobEvent = { seq: number; at: number; kind: string; data: Record<string, unknown> }

export type Experiment = {
  dataset: string
  split: string
  seeds: number[]
  price: number
  seed_labels: number
  points: { labels: number; dollars: number; random: number; uncertain: number; blindspot: number }[]
}

export type Audit = {
  ok: boolean
  order_id: string
  source: string
  checks: { check: string; ok: boolean; blindspot: string; paypal: string }[]
}

export type Me = { id: string; name: string; kind: string; banned: boolean; open: number; answered: number; accepted: number; earned: string }
export type Task = {
  task_id: string
  tips: { a: string; b: string; tip: string }[]
  text: string
  options: { id: number; label: string }[]
  all_intents: { id: number; label: string }[]
}

const KEY = 'blindspot.operatorKey'
export const operatorKey = {
  get: () => localStorage.getItem(KEY) ?? '',
  set: (k: string) => localStorage.setItem(KEY, k),
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function call<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: { 'content-type': 'application/json', ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* keep status text */
    }
    throw new ApiError(r.status, msg)
  }
  return (await r.json()) as T
}

const op = () => ({ 'X-Operator-Key': operatorKey.get() })

export const console_ = {
  jobs: () => call<Job[]>('GET', '/api/jobs', undefined, op()),
  job: (id: string) => call<JobDetail>('GET', `/api/jobs/${id}`, undefined, op()),
  createJob: (body: Record<string, unknown>) => call<{ job_id: string }>('POST', '/api/jobs', body, op()),
  fundCard: (id: string) => call<{ authorization_id: string; amount: string }>('POST', `/api/jobs/${id}/fund/card`, {}, op()),
  walletOrder: (id: string) => call<{ order_id: string }>('POST', `/api/jobs/${id}/fund/wallet/order`, {}, op()),
  walletAuthorize: (id: string) => call<{ authorization_id: string }>('POST', `/api/jobs/${id}/fund/wallet/authorize`, {}, op()),
  autopilot: (id: string, body: { rounds: number; wait_seconds: number; close_when_done: boolean }) =>
    call<{ autopilot: boolean }>('POST', `/api/jobs/${id}/autopilot`, body, op()),
  stop: (id: string) => call<{ autopilot: boolean }>('DELETE', `/api/jobs/${id}/autopilot`, undefined, op()),
  close: (id: string) => call<{ status: string; released?: string }>('POST', `/api/jobs/${id}/close`, {}, op()),
  reconcile: (id: string) => call<{ settlements: Settlement[] }>('POST', `/api/jobs/${id}/reconcile`, {}, op()),
  audit: (id: string) => call<Audit>('POST', `/api/jobs/${id}/audit`, {}, op()),
  ask: (id: string, question: string) => call<{ answer: string; tools: string[] }>('POST', `/api/jobs/${id}/ask`, { question }, op()),
  annotators: () => call<Me[]>('GET', '/api/annotators', undefined, op()),
  addSimulated: (body: { name: string; paypal_email: string; accuracy: number; late_accuracy?: number | null }) =>
    call<{ annotator_id: string }>('POST', '/api/annotators/simulated', body, op()),
  events: (id: string, after: number) =>
    new EventSource(`/api/jobs/${id}/events?after=${after}&key=${encodeURIComponent(operatorKey.get())}`),
}

export const pub = {
  config: () => call<{ paypal_client_id: string | null; invite_required: boolean; environment: string }>('GET', '/api/config'),
  experiment: () => call<Experiment>('GET', '/api/experiment'),
}

const TOKEN = 'blindspot.annotatorToken'
export const annotatorToken = {
  get: () => localStorage.getItem(TOKEN) ?? '',
  set: (t: string) => localStorage.setItem(TOKEN, t),
  clear: () => localStorage.removeItem(TOKEN),
}
const ann = () => ({ Authorization: `Bearer ${annotatorToken.get()}` })

export const annotator = {
  join: (body: { name: string; paypal_email: string; invite: string }) =>
    call<{ annotator_id: string; token: string }>('POST', '/api/annotators/join', body),
  task: () => call<{ task: Task | null; me: Me }>('GET', '/api/me/task', undefined, ann()),
  answer: (task_id: string, intent: number) => call<{ task: Task | null; me: Me }>('POST', '/api/me/answer', { task_id, intent }, ann()),
}

export const usd = (cents: number) => `$${(cents / 100).toFixed(2)}`
export const pct = (x: number | null | undefined) => (x == null ? '—' : `${(x * 100).toFixed(1)}%`)

/** Model answers may contain light markdown; the console shows plain text. */
export const plain = (t: string) => t.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`([^`]+)`/g, '$1')
