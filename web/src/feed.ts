// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { plain, type JobEvent } from './api'

export type FeedLine = { seq: number; at: number; tone: 'agent' | 'money' | 'good' | 'warn' | 'bad' | 'info'; title: string; body?: string; ids?: string[] }

const s = (v: unknown) => (v == null ? '' : String(v))
const p = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : s(v))

/** Turns recorded job events into plain-English lines for the agent feed. */
export function describe(e: JobEvent): FeedLine | null {
  const d = e.data
  const base = { seq: e.seq, at: e.at }
  switch (e.kind) {
    case 'job_created':
      return { ...base, tone: 'info', title: `Job created: budget $${s(d.budget)}, $${s(d.price)} per accepted label`, body: `${s(d.seed_labels)} existing labelled examples; strategy: ${s(d.strategy)}.` }
    case 'budget_held':
      return { ...base, tone: 'money', title: `PayPal is holding $${s(d.amount)} — nothing charged yet`, ids: [`order ${s(d.order_id)}`, `authorization ${s(d.authorization_id)}`] }
    case 'autopilot_started':
      return { ...base, tone: 'agent', title: 'Buyer agent started', body: Number(d.wait_seconds) > 0 ? `People get ${s(d.wait_seconds)} s to answer each round.` : 'Simulated annotators only (answer instantly).' }
    case 'round_planned':
      return { ...base, tone: 'agent', title: `Round ${s(d.round)}: buying ${s(d.labels)} labels for ${s(d.cost)}`, body: s(d.explanation) }
    case 'agent_stopped':
      return { ...base, tone: 'agent', title: 'Agent decided to stop buying', body: s(d.detail) }
    case 'tasks_expired':
      return { ...base, tone: 'warn', title: `${s(d.count)} tasks expired unanswered — not charged, not paid` }
    case 'round_audited': {
      const anns = (d.annotators ?? {}) as Record<string, { gold: string; ok: boolean }>
      const failed = Object.values(anns).filter((a) => !a.ok).length
      return {
        ...base,
        tone: failed ? 'warn' : 'good',
        title: `Quality check: ${s(d.accepted)} labels accepted, ${s(d.rejected)} rejected`,
        body: failed ? `${failed} annotator(s) failed the hidden test questions and are not paid for this round.` : 'Every annotator passed the hidden test questions.',
      }
    }
    case 'charged':
      return { ...base, tone: 'money', title: `Charged $${s(d.amount)} from the hold for accepted labels only`, ids: [`capture ${s(d.capture_id)}`] }
    case 'charge_uncertain':
      return { ...base, tone: 'warn', title: 'PayPal did not answer the charge — checking before any retry', body: 'The same request id is reused, so a retry can never charge twice.' }
    case 'charge_refused':
      return { ...base, tone: 'bad', title: 'PayPal refused the charge — nobody is paid and those labels are not used', body: `${s(d.name)} ${s(d.issues)}` }
    case 'payout_submitted':
      return { ...base, tone: 'money', title: `Sent $${s(d.amount)} to annotators with PayPal Payouts`, body: 'Each person counts as paid once PayPal reports their item SUCCESS.', ids: [`batch ${s(d.payout_batch_id)}`] }
    case 'payout_already_submitted':
      return { ...base, tone: 'good', title: 'Payout confirmed by PayPal duplicate check — nobody paid twice' }
    case 'payout_refused':
      return { ...base, tone: 'bad', title: 'PayPal refused the payout', body: `${s(d.name)} ${s(d.issues)}` }
    case 'refund_refused':
      return { ...base, tone: 'bad', title: 'PayPal refused the refund', body: `${s(d.name)} ${s(d.issues)}` }
    case 'payout_uncertain':
      return { ...base, tone: 'warn', title: 'Payout response lost — will check before resending the same batch id' }
    case 'charge_pending':
      return { ...base, tone: 'warn', title: `PayPal is still processing the $${s(d.amount)} charge`, body: 'Nobody is paid and no label trains the model until it completes.' }
    case 'charge_needs_review':
    case 'payout_needs_review':
      return { ...base, tone: 'bad', title: 'A payment needs a person to check it in PayPal', body: 'Nothing more is sent automatically for it.' }
    case 'refund_pending':
      return { ...base, tone: 'warn', title: 'Refund accepted by PayPal, still processing' }
    case 'autopilot_waiting':
      return { ...base, tone: 'warn', title: 'Agent paused: waiting for PayPal to confirm a charge' }
    case 'void_refused':
      return { ...base, tone: 'bad', title: `The hold was not released (PayPal shows ${s(d.hold)})` }
    case 'payout_checked':
      return { ...base, tone: d.state === 'DONE' ? 'good' : 'warn', title: `Payout status: ${s(d.state)}`, body: `batch ${s(d.batch_status)}` }
    case 'annotator_revoked':
      return { ...base, tone: 'bad', title: `Annotator removed after failing quality checks: ${s(d.reason)}`, body: `${s(d.labels)} earlier labels removed from training; the customer is refunded for them.` }
    case 'refunded':
      return { ...base, tone: 'money', title: `Refunded $${s(d.amount)} to the customer for revoked labels`, ids: [`refund ${s(d.refund_id)}`] }
    case 'model_measured':
      return d.round == null ? { ...base, tone: 'info', title: `Starting accuracy ${p(d.validation)} on ${s(d.labels)} labels` } : { ...base, tone: 'good', title: `Retrained: accuracy ${p(d.validation)} with ${s(d.labels)} labels`, body: `Customer charged $${s(d.charged)} so far.` }
    case 'budget_released':
      if (d.hold === 'FULLY_USED') return { ...base, tone: 'money', title: 'Budget fully used — nothing left to release', body: `Total spent $${s(d.spent)} after refunds.` }
      return { ...base, tone: 'money', title: `Released $${s(d.released)} back to the customer`, body: `Hold ${s(d.hold)}. Total spent $${s(d.spent)}.` }
    case 'audit_completed': {
      const checks = (d.checks ?? []) as { ok: boolean }[]
      return d.ok
        ? { ...base, tone: 'good', title: `Audit: all ${checks.length} checks match PayPal's records`, body: 'Read through the PayPal Agent Toolkit.' }
        : { ...base, tone: 'bad', title: `Audit: ${checks.filter((c) => !c.ok).length} check(s) disagree with PayPal`, body: 'See the audit table.' }
    }
    case 'money_question':
      return { ...base, tone: 'info', title: `Customer asked: “${s(d.question)}”`, body: plain(s(d.answer)) }
    case 'explanation_fallback':
      return { ...base, tone: 'info', title: 'Explanation written from the template (language model unavailable)' }
    case 'autopilot_finished':
      return { ...base, tone: 'agent', title: 'Buyer agent finished' }
    case 'autopilot_stopped':
      return { ...base, tone: 'warn', title: 'Buyer agent stopped by the operator' }
    case 'autopilot_failed':
      return { ...base, tone: 'bad', title: 'Buyer agent stopped on an error', body: s(d.error) }
    case 'annotator_joined':
      return { ...base, tone: 'info', title: `${s(d.name)} joined as an annotator` }
    default:
      return null
  }
}
