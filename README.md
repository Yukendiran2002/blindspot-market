# Blindspot Market

**An AI agent that pays people to teach it what it gets wrong, settled on PayPal.**

A support team's AI keeps confusing "my card payment is pending" with "my card payment was declined". Labelling
thousands of random messages is slow and expensive, and most of that money goes on messages the model already handles.
Blindspot gives a buyer agent a budget held on PayPal. The agent finds the model's blind spots and buys labels for
exactly those messages from people. It checks their work with hidden test questions, pays them and retrains. It stops
when more labels stop paying off. The customer is charged only for labels that pass the check, and the unused budget is
released.

> PayPal **sandbox** only. No real money moves. Annotators in the scripted demo runs are **simulated** and marked as
> such everywhere. Real people can join on the annotator page.

## What happens in one job

| Step | AI | PayPal (sandbox REST APIs) |
|---|---|---|
| Fund | — | Customer approves a budget **hold** (Orders v2, `intent=AUTHORIZE`, PayPal button or test card) |
| Find blind spots | The model scores every unlabelled message; the agent groups the most uncertain ones into confusion areas ("Pending card payment ↔ Declined card payment") | — |
| Decide | Fixed rules decide whether to buy and how many: target reached? budget left? did the last round pay off per $10? Gemini explains the decision in plain English | — |
| Label | People answer on the mobile annotator page; hidden test questions check them | — |
| Charge | — | **Partial capture** of the hold for accepted labels only (`final_capture=false`; `PayPal-Request-Id` and `custom_id` = settlement id) |
| Pay people | — | **Payouts** batch to each annotator's PayPal email, sent only after the charge is confirmed (`sender_batch_id` = settlement id) |
| Catch cheaters late | The annotator's test-question record falls below the trust bar: they are banned and their earlier labels are removed from training | **Refund** for those labels against the round's capture |
| Retrain & measure | Model retrains (about a second) and is measured on a held-out set | — |
| Stop | Rules stop the agent: target reached, low value, budget used, or the hold is past PayPal's 3-day window | Unused hold **voided** |
| Audit | Gemini answers "Why was I refunded?" with read-only tools | Ledger checked against PayPal's own records through the **PayPal Agent Toolkit** |

## Why the agent is worth paying

Offline benchmark on [Banking77](https://huggingface.co/datasets/PolyAI/banking77): 77 customer-support intents, by
PolyAI, CC-BY-4.0. Each run uses the same model and the same price per label ($0.50). Results are averaged over 5 seeds
and measured on the official 3,080-message test split, which is never used for any decision.
Source: [`experiment/run_experiment.py`](experiment/run_experiment.py). Results:
[`experiment/results.json`](experiment/results.json).

| Spent on labels | Random buying | Uncertainty only | Blindspot |
|---:|---:|---:|---:|
| $50 | 63.6% ± 1.2 | 67.8% ± 2.2 | **69.8% ± 2.9** |
| $100 | 70.3% ± 1.5 | 76.3% ± 1.0 | **77.1% ± 0.9** |
| $250 | 80.1% ± 1.2 | **85.6% ± 0.7** | 85.4% ± 0.3 |
| $400 | 84.1% ± 1.0 | 88.5% ± 0.3 | **88.9% ± 0.4** |
| $600 | 86.8% ± 0.5 | 90.4% ± 0.4 | **90.7% ± 0.3** |

Mean ± standard deviation over the 5 seeds.

- Random buying needs about **$450** to reach the accuracy Blindspot reaches at **$250**: about 45% less spend.
- Most of the gain over random comes from targeting uncertain messages. Grouping them into confusion areas adds only a
  small gain over plain uncertainty sampling, and not at every budget. Its main value is variety within a round, plus
  an explanation a customer can read: "you are paying to fix Pending ↔ Declined".

## Money safety

The agent never moves money itself. The language model writes explanations and answers questions. Deterministic code
moves money, and every step is recorded before it is sent.

- **Charge once.** A capture's `PayPal-Request-Id` and `custom_id` are the settlement id. If a response is lost, the
  order is read back and the capture is found by its `custom_id` before anything is resent. A resend reuses the same id.
- **Pay once, and only after the charge.** Payouts use `sender_batch_id` = settlement id, and PayPal refuses a
  duplicate batch. No payout is sent while a capture is uncertain or refused.
- **Pay only for checked work.** Hidden test questions (the customer's own labelled examples) gate every annotator.
  Labels from a refused charge never train the model.
- **Never exceed the hold.** Planning caps each round by the money left, and settlement checks the limit again before
  charging.
- **"Paid" means received.** A payout counts as done only when every item in it shows SUCCESS. Unclaimed or pending
  items stay visible.
- **Hold window.** PayPal honours an authorization for 3 days
  ([docs](https://developer.paypal.com/payment-methods/auth-honor/)). The agent stops buying on an older hold.
- **Sandbox only.** The REST client and the toolkit reader refuse any host other than `api-m.sandbox.paypal.com`.
- **Restart-safe.** On start, the server reconciles anything left uncertain.

The 29 tests in `tests/` run against a local fake PayPal (`tests/fake_paypal.py`), which copies the behaviour we saw in
the sandbox. They cover:

- lost responses, server errors and captures that are slow to appear
- a capture PayPal no longer matches to our request id
- refused charges, cheaters and budget limits
- annotator tokens, task expiry and the audit

Every money path was also run against the real sandbox.

## AI used

| Where | What |
|---|---|
| Finding blind spots | Sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and logistic regression. The agent ranks messages by how unsure the model is, then uses k-means to group the most uncertain ones |
| Explaining purchases | Gemini (`gemini-3.8-flash`) through the [Nexura Agents SDK](https://pypi.org/project/nexura-agents/), no tools. Falls back to a template |
| Helping annotators | Each round, Gemini writes one tip per confused pair ("Use Pending when the payment is still processing, Declined when it was refused"), shown with the matching tasks |
| Answering money questions | Gemini through Nexura, with two read-only tools: the PayPal Agent Toolkit's `get_order_details` for the job's order, and Blindspot's ledger |
| Checking the books | The PayPal Agent Toolkit (`paypal-agent-toolkit`, framework-neutral core) reads PayPal's records. Code compares them with the ledger |

## Run it

You need:

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node 22+
- a PayPal sandbox REST app (developer.paypal.com → Apps & Credentials → Sandbox)
- optionally, a Gemini API key

```bash
cp .env.example .env            # fill in PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, GEMINI_API_KEY, BLINDSPOT_OPERATOR_KEY
uv sync
(cd web && npm ci && npm run build)
uv run python -c "from blindspot.api import main; main()"
```

The first start downloads Banking77 and embeds it on the CPU. This takes a few minutes and is cached in `data/`. Then
open:

- the console: http://127.0.0.1:8077 (needs the operator key)
- the annotator page: http://127.0.0.1:8077/label (on a phone or in a second window)

To run a demo in the console:

1. Click **New job**.
2. Fund it with the **PayPal** button (sign in with a sandbox Personal account) or with the sandbox test card.
3. Add annotators: create simulated ones through the API (below), or have people join on the annotator page.
4. Click **Run buyer agent**. Set "Time for people" above 0 to give people time to answer.
5. Watch the feed.
6. Click **Run audit**, then ask "Why was I refunded?".

To create a simulated annotator (for scripted runs only):

```bash
curl -X POST localhost:8077/api/annotators/simulated -H "X-Operator-Key: $KEY" -H 'content-type: application/json' \
  -d '{"name":"Spam bot (simulated)","paypal_email":"spam@example.com","accuracy":0.1}'
```

Deploy: `Dockerfile` (one container: API + web, dataset embedded at build time) and `render.yaml`. Set the secrets in
the host's dashboard.

Other commands:

- `uv run pytest`: run the tests
- `uv run python scripts/live_job.py /tmp/live.db 3`: run a whole job against the sandbox from the command line
- `uv run python experiment/run_experiment.py`: rerun the benchmark

## Limits

- Sandbox only. Production use would need PayPal's approval for Payouts and partial captures enabled on the merchant
  account. Blindspot would act as the labelling vendor paying its own contractors.
- One embedded classifier and one public dataset. The customer's "held-out set" is a fixed validation split.
- Simulated annotators answer from a fixed accuracy. Real labelling speed and quality are not measured here.
- SQLite and a single server process.
- Webhooks are not used yet. Status is read back from PayPal on demand.

## Licence and credits

Apache-2.0 (see `LICENSE`). Banking77 by PolyAI (CC-BY-4.0). PayPal Agent Toolkit by PayPal (Apache-2.0). AG Grid
Community (MIT). Built by Liceron for the PayPal AI Hackathon 2026.
