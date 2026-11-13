import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
  CreditCard,
  Cpu,
  Database,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  ShieldCheck,
  Smartphone,
  Sparkles,
  TriangleAlert,
  WalletCards,
} from 'lucide-react'
import {
  CartesianGrid,
  Bar,
  BarChart as RechartsBarChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

const API_URL = 'http://localhost:8000'
const MAX_KEPT = 300

/* ------------------------------------------------------------------ types */
type Channel = 'paysim' | 'sparkov'
type Filter = 'all' | Channel
type Inputs = Record<string, string | number>

type Prediction = {
  id: number
  channel: Channel
  segment: string
  amount: number
  inputs: Inputs
  fraud_probability: number
  is_fraud: boolean
  threshold: number
  model: string
  created_at: string
}

type ChannelSummary = {
  total: number
  flagged: number
  volume: number
  avg_risk: number
  segments: Record<string, { total: number; flagged: number }>
}

type Analytics = {
  total_transactions: number
  channels: Record<Channel, ChannelSummary>
  transactions: Prediction[]
}

type ChannelModelInfo = {
  available: boolean
  error: string | null
  model: string | null
  threshold: number | null
  metrics: { precision: number; recall: number; f1: number; roc_auc: number; pr_auc: number } | null
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number } | null
  rows: { fit: number; val: number; test: number } | null
  top_features: { feature: string; importance: number }[]
}

type ComparisonRow = {
  Dataset: string
  Model: string
  'Val PR-AUC': number
  'Precision@thr': number
  'Recall@thr': number
  'F1@thr': number
  'ROC-AUC': number
  'PR-AUC': number
}

type ModelInfo = {
  channels: Record<Channel, ChannelModelInfo>
  comparison: ComparisonRow[]
}

/* ------------------------------------------------------- channel config */
type FieldDef = {
  key: string
  label: string
  hint: string
  options?: { value: string | number; label: string }[]
  min?: number
  max?: number
  step?: number
}

const PAYSIM_TYPES = ['TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEBIT', 'CASH_IN']
const SPARKOV_CATEGORIES = [
  'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness', 'home',
  'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel',
]
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

const CHANNEL_CONFIG: Record<
  Channel,
  {
    label: string
    dataset: string
    icon: typeof Smartphone
    fields: FieldDef[]
    normal: Inputs
    suspicious: Inputs
  }
> = {
  paysim: {
    label: 'Mobile money',
    dataset: 'PaySim',
    icon: Smartphone,
    fields: [
      { key: 'type', label: 'Transaction type', hint: 'How the money moves.', options: PAYSIM_TYPES.map((t) => ({ value: t, label: t })) },
      { key: 'amount', label: 'Amount', hint: 'The money moved in this transaction.', min: 0, step: 0.01 },
      { key: 'oldbalanceOrg', label: 'Sender balance before', hint: 'The sender’s account balance before the transaction.', min: 0, step: 0.01 },
      { key: 'newbalanceOrig', label: 'Sender balance after', hint: 'The sender’s account balance after the transaction.', min: 0, step: 0.01 },
      { key: 'oldbalanceDest', label: 'Recipient balance before', hint: 'The recipient’s account balance before the transaction.', min: 0, step: 0.01 },
      { key: 'newbalanceDest', label: 'Recipient balance after', hint: 'The recipient’s account balance after the transaction.', min: 0, step: 0.01 },
    ],
    normal: { type: 'PAYMENT', amount: 52.3, oldbalanceOrg: 1200, newbalanceOrig: 1147.7, oldbalanceDest: 0, newbalanceDest: 0 },
    suspicious: { type: 'TRANSFER', amount: 181, oldbalanceOrg: 181, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 0 },
  },
  sparkov: {
    label: 'Card payments',
    dataset: 'Sparkov',
    icon: CreditCard,
    fields: [
      { key: 'category', label: 'Merchant category', hint: 'What kind of merchant was paid.', options: SPARKOV_CATEGORIES.map((c) => ({ value: c, label: c })) },
      { key: 'amt', label: 'Amount', hint: 'The purchase amount.', min: 0, step: 0.01 },
      { key: 'hour', label: 'Hour of day (0–23)', hint: 'Local hour when the purchase was made.', min: 0, max: 23, step: 1 },
      { key: 'day_of_week', label: 'Day of week', hint: 'Day the purchase was made.', options: WEEKDAYS.map((d, i) => ({ value: i, label: d })) },
      { key: 'age', label: 'Cardholder age', hint: 'Age of the cardholder in years.', min: 0, max: 120, step: 1 },
      { key: 'city_pop', label: 'Home city population', hint: 'Population of the cardholder’s home city.', min: 0, step: 1 },
      { key: 'distance_km', label: 'Distance to merchant (km)', hint: 'Distance between the cardholder’s home and the merchant.', min: 0, step: 0.1 },
    ],
    normal: { category: 'grocery_pos', amt: 45.2, hour: 14, day_of_week: 2, age: 41, city_pop: 50000, distance_km: 12 },
    suspicious: { category: 'shopping_net', amt: 912.5, hour: 2, day_of_week: 5, age: 34, city_pop: 1200, distance_km: 88 },
  },
}

const CHANNELS: Channel[] = ['paysim', 'sparkov']

/* -------------------------------------------------------- live simulator */
// Demo traffic only. These are illustrative templates, NOT real data: the models decide
// what is flagged, so a "suspicious" template is not guaranteed to be flagged.
const rnd = (a: number, b: number) => a + Math.random() * (b - a)
const rndInt = (a: number, b: number) => Math.floor(rnd(a, b + 1))
const pick = <T,>(items: T[]): T => items[Math.floor(Math.random() * items.length)]
const round2 = (n: number) => Math.round(n * 100) / 100

function generatePaysim(suspicious: boolean): Inputs {
  if (suspicious) {
    const amount = round2(rnd(500, 900000))
    return { type: pick(['TRANSFER', 'CASH_OUT']), amount, oldbalanceOrg: amount, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 0 }
  }
  const type = pick(['PAYMENT', 'PAYMENT', 'CASH_IN', 'CASH_OUT', 'TRANSFER', 'DEBIT'])
  const amount = round2(rnd(5, 20000))
  const before = round2(amount + rnd(100, 80000))
  const toMerchant = type === 'PAYMENT' || type === 'DEBIT'
  const destBefore = toMerchant ? 0 : round2(rnd(1000, 500000))
  return {
    type,
    amount,
    oldbalanceOrg: before,
    newbalanceOrig: round2(type === 'CASH_IN' ? before + amount : before - amount),
    oldbalanceDest: destBefore,
    newbalanceDest: toMerchant ? 0 : round2(destBefore + amount),
  }
}

function generateSparkov(suspicious: boolean): Inputs {
  if (suspicious) {
    return {
      category: pick(['shopping_net', 'misc_net', 'grocery_pos']),
      amt: round2(rnd(250, 1300)),
      city_pop: rndInt(300, 5000),
      hour: pick([22, 23, 0, 1, 2, 3]),
      day_of_week: rndInt(0, 6),
      age: rndInt(20, 75),
      distance_km: round2(rnd(20, 110)),
    }
  }
  return {
    category: pick(SPARKOV_CATEGORIES.filter((c) => !['shopping_net', 'misc_net'].includes(c))),
    amt: round2(rnd(4, 120)),
    city_pop: rndInt(500, 2000000),
    hour: rndInt(8, 21),
    day_of_week: rndInt(0, 6),
    age: rndInt(20, 75),
    distance_km: round2(rnd(2, 100)),
  }
}

/* --------------------------------------------------------------- helpers */
function formatMoney(value: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)
}

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? '–' : `${(value * 100).toFixed(digits)}%`
const num3 = (value: number | null | undefined) => (value === null || value === undefined ? '–' : value.toFixed(3))
const cleanFeature = (name: string) => name.replace(/^(cat|num)__/, '')

function mergePrediction(current: Analytics, p: Prediction): Analytics {
  if (current.transactions.some((t) => t.id === p.id)) return current // same row arrives via POST reply and SSE
  const before = current.channels[p.channel]
  const total = before.total + 1
  const segment = before.segments[p.segment] ?? { total: 0, flagged: 0 }
  return {
    total_transactions: current.total_transactions + 1,
    transactions: [...current.transactions, p].slice(-MAX_KEPT),
    channels: {
      ...current.channels,
      [p.channel]: {
        total,
        flagged: before.flagged + (p.is_fraud ? 1 : 0),
        volume: before.volume + p.amount,
        avg_risk: (before.avg_risk * before.total + p.fraud_probability) / total,
        segments: {
          ...before.segments,
          [p.segment]: { total: segment.total + 1, flagged: segment.flagged + (p.is_fraud ? 1 : 0) },
        },
      },
    },
  }
}

/* ------------------------------------------------------------------- App */
function App() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [channel, setChannel] = useState<Channel>('paysim')
  const [forms, setForms] = useState<Record<Channel, Inputs>>({
    paysim: { ...CHANNEL_CONFIG.paysim.suspicious },
    sparkov: { ...CHANNEL_CONFIG.sparkov.suspicious },
  })
  const [lastResult, setLastResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [simulating, setSimulating] = useState(false)
  const [live, setLive] = useState(false)
  const [error, setError] = useState('')

  const loadAll = useCallback(async () => {
    try {
      const [analyticsResponse, modelResponse] = await Promise.all([
        fetch(`${API_URL}/analytics?limit=${MAX_KEPT}`),
        fetch(`${API_URL}/model-info`),
      ])
      if (!analyticsResponse.ok) throw new Error('Could not load analytics')
      setAnalytics((await analyticsResponse.json()) as Analytics)
      if (modelResponse.ok) setModelInfo((await modelResponse.json()) as ModelInfo)
      setError('')
    } catch {
      setError('Unable to connect to the FraudShield API. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  const sendTransaction = useCallback(async (ch: Channel, inputs: Inputs) => {
    const response = await fetch(`${API_URL}/transaction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: ch, ...inputs }),
    })
    if (!response.ok) throw new Error('Transaction could not be processed')
    const prediction = (await response.json()) as Prediction
    setAnalytics((current) => (current ? mergePrediction(current, prediction) : current))
    return prediction
  }, [])

  useEffect(() => {
    let events: EventSource | null = null
    let cancelled = false

    void loadAll().then(() => {
      if (cancelled) return
      events = new EventSource(`${API_URL}/events`)
      events.onopen = () => setLive(true)
      events.onmessage = (event) => {
        const prediction = JSON.parse(event.data) as Prediction
        setAnalytics((current) => (current ? mergePrediction(current, prediction) : current))
      }
      events.onerror = () => setLive(false) // the browser reconnects on its own
    })

    return () => {
      cancelled = true
      events?.close()
    }
  }, [loadAll])

  useEffect(() => {
    if (!simulating) return
    let stopped = false
    const tick = async () => {
      const ch: Channel = Math.random() < 0.5 ? 'paysim' : 'sparkov'
      const suspicious = Math.random() < 0.15
      try {
        await sendTransaction(ch, ch === 'paysim' ? generatePaysim(suspicious) : generateSparkov(suspicious))
      } catch {
        if (!stopped) {
          setError('Simulation stopped: the backend did not accept a transaction.')
          setSimulating(false)
        }
      }
    }
    const timer = setInterval(() => void tick(), 2000)
    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [simulating, sendTransaction])

  /* ---- derived values for the selected channel filter ---- */
  const activeChannels = filter === 'all' ? CHANNELS : [filter]
  const summaries = analytics ? activeChannels.map((key) => ({ key, ...analytics.channels[key] })) : []
  const totalCount = summaries.reduce((sum, s) => sum + s.total, 0)
  const flaggedCount = summaries.reduce((sum, s) => sum + s.flagged, 0)
  const totalVolume = summaries.reduce((sum, s) => sum + s.volume, 0)
  const averageRisk = totalCount ? summaries.reduce((sum, s) => sum + s.avg_risk * s.total, 0) / totalCount : 0

  const visible = (analytics?.transactions ?? []).filter((t) => filter === 'all' || t.channel === filter)
  const chartData = visible.slice(-120).map((t, index) => ({ ...t, chartIndex: index + 1, risk: t.fraud_probability * 100 }))

  const segmentRows = summaries
    .flatMap((s) => Object.entries(s.segments).map(([segment, stat]) => ({ channel: s.key, segment, ...stat })))
    .sort((a, b) => b.total - a.total)
  const flaggedBySegment = segmentRows
    .filter((row) => row.flagged > 0)
    .sort((a, b) => b.flagged - a.flagged)
    .slice(0, 8)
    .map((row) => ({ type: row.segment, frauds: row.flagged }))

  const isEmpty = !loading && totalCount === 0
  const config = CHANNEL_CONFIG[channel]

  function updateField(field: FieldDef, value: string) {
    const parsed = field.options && typeof field.options[0].value === 'string' ? value : Number(value)
    setForms((current) => ({ ...current, [channel]: { ...current[channel], [field.key]: parsed } }))
  }

  function loadExample(kind: 'normal' | 'suspicious') {
    setForms((current) => ({ ...current, [channel]: { ...CHANNEL_CONFIG[channel][kind] } }))
  }

  async function submitTransaction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      setLastResult(await sendTransaction(channel, forms[channel]))
    } catch {
      setError('The transaction could not be processed. Please check the backend.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={22} /></div>
          <div>
            <strong>FraudShield </strong>
            <span style={{ marginLeft: '5px' }}>Real-time transaction intelligence</span>
          </div>
        </div>
        <div className={`live-indicator ${live ? '' : 'offline'}`}><span /> {live ? 'Live monitoring' : 'Reconnecting'} <RefreshCw size={14} /></div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow"><Sparkles size={15} /> TRANSACTION COMMAND CENTER</p>
          <h1>Know every transaction<br /><em>before it becomes a risk.</em></h1>
          <p className="hero-copy">Two specialist models watch two channels, mobile money and card payments, and surface suspicious activity as it happens.</p>
        </div>
        <div className="hero-badge"><ShieldCheck size={28} /><span>MODEL STATUS<strong>{modelInfo ? (CHANNELS.every((k) => modelInfo.channels[k].available) ? 'Both models active' : 'Degraded: a model is offline') : 'Checking…'}</strong></span></div>
      </section>

      {error && <div className="error-banner"><TriangleAlert size={18} /> {error}</div>}

      <div className="channel-tabs" role="tablist" aria-label="Channel filter">
        {(['all', ...CHANNELS] as Filter[]).map((key) => {
          const Icon = key === 'all' ? WalletCards : CHANNEL_CONFIG[key].icon
          return (
            <button key={key} role="tab" aria-selected={filter === key} className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>
              <Icon size={15} /> {key === 'all' ? 'All channels' : `${CHANNEL_CONFIG[key].label} · ${CHANNEL_CONFIG[key].dataset}`}
            </button>
          )
        })}
      </div>

      {loading ? (
        <div className="loading-state"><LoaderCircle className="spin" size={28} /> Loading transaction intelligence...</div>
      ) : isEmpty ? (
        <section className="empty-state">
          <div className="empty-icon"><Database size={42} /></div>
          <h2>No transactions yet</h2>
          <p>Nothing to analyze for this view yet.<br />Submit a transaction below, or start the live simulation.</p>
          <a href="#new-transaction" className="button primary">Add first transaction <ArrowDownRight size={17} /></a>
        </section>
      ) : (
        <>
          <section className="metrics-grid">
            <Metric icon={<Database />} label="Total transactions" value={String(totalCount)} note="All recorded activity" />
            <Metric icon={<CircleDollarSign />} label="Transaction volume" value={formatMoney(totalVolume)} note="Across all activity" />
            <Metric icon={<AlertTriangle />} label="Flagged transactions" value={String(flaggedCount)} note={flaggedCount ? 'Requires attention' : 'No threats detected'} tone={flaggedCount ? 'danger' : 'safe'} />
            <Metric icon={<BarChart3 />} label="Average risk score" value={pct(averageRisk)} note="Model fraud probability" tone={averageRisk > 0.5 ? 'danger' : 'safe'} />
          </section>
          <section className="content-grid">
            <div className="panel chart-panel">
              <div className="panel-heading"><div><p className="eyebrow">ACTIVITY OVERVIEW</p><h2>Risk score by transaction</h2></div><span className="legend"><i className="legend-dot" /> Clear <i className="legend-dot fraud-legend-dot" /> Flagged</span></div>
              <div className="chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#e9efeb" vertical={false} />
                    <XAxis dataKey="chartIndex" hide />
                    <YAxis domain={[0, 100]} tickFormatter={(value: number) => `${value}%`} width={40} tick={{ fill: '#a0aca9', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                    <Tooltip shared={false} formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Risk']} labelFormatter={(label, payload) => {
                      const row = payload?.[0]?.payload as (Prediction & { chartIndex: number }) | undefined
                      return row ? `#${row.id} · ${CHANNEL_CONFIG[row.channel].label} · ${row.segment} · ${formatMoney(row.amount)}` : `Transaction ${label}`
                    }} />
                    <Line
                      type="monotone"
                      dataKey="risk"
                      stroke="#2f8f5b"
                      strokeWidth={2}
                      dot={(props) => {
                        const { cx, cy, payload } = props
                        return <circle cx={cx} cy={cy} r={2} fill={payload.is_fraud ? '#d95c5c' : '#ffffff'} stroke={payload.is_fraud ? '#d95c5c' : '#2f8f5b'} strokeWidth={2} />
                      }}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-foot"><span>Oldest</span><span>Latest</span></div>
            </div>
            <div className="panel breakdown-panel">
              <div className="panel-heading"><div><p className="eyebrow">DISTRIBUTION</p><h2>Transaction segments</h2></div><WalletCards size={21} /></div>
              <div className="type-list">{segmentRows.slice(0, 8).map((row) => (
                <div className="type-row" key={`${row.channel}-${row.segment}`}>
                  <div><span className="type-icon">{row.segment.slice(0, 1).toUpperCase()}</span><strong>{row.segment}</strong><span className={`channel-badge ${row.channel}`}>{CHANNEL_CONFIG[row.channel].dataset}</span></div>
                  <span>{row.total} <small>{row.flagged ? `· ${row.flagged} flagged` : row.total === 1 ? 'transaction' : 'transactions'}</small></span>
                </div>
              ))}</div>
            </div>
          </section>
          <section className="panel fraud-chart-panel">
            <div className="panel-heading"><div><p className="eyebrow">RISK DISTRIBUTION</p><h2>Flagged transactions by segment</h2></div><span className="legend"><i className="legend-dot fraud-legend-dot" /> Flagged</span></div>
            <div className="fraud-chart">
              {flaggedBySegment.length === 0 ? (
                <p className="chart-empty">No flagged transactions in this view.</p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsBarChart data={flaggedBySegment} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}>
                    <CartesianGrid stroke="#e9efeb" vertical={false} />
                    <XAxis dataKey="type" interval={0} tick={{ fill: '#758286', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                    <YAxis allowDecimals={false} width={28} tick={{ fill: '#a0aca9', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                    <Tooltip formatter={(value) => [value, 'Flagged']} />
                    <Bar dataKey="frauds" fill="#d9574d" radius={[5, 5, 0, 0]} />
                  </RechartsBarChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>
          <section className="panel recent-panel">
            <div className="panel-heading"><div><p className="eyebrow">LIVE FEED</p><h2>Recent transactions</h2></div><span className={`updated ${live ? '' : 'offline'}`}><span /> {live ? 'Live updates' : 'Reconnecting'}</span></div>
            <div className="table-wrap"><table><thead><tr><th>Channel</th><th>Transaction</th><th>Amount</th><th>Risk score</th><th>Decision</th></tr></thead><tbody>{visible.slice().reverse().slice(0, 8).map((t) => (
              <tr key={t.id}>
                <td><span className={`channel-badge ${t.channel}`}>{CHANNEL_CONFIG[t.channel].dataset}</span></td>
                <td><strong>{t.segment}</strong><small>Scored by {t.model}</small></td>
                <td>{formatMoney(t.amount)}</td>
                <td><div className="risk-cell" title={`Flag threshold ${pct(t.threshold)}`}><div className="risk-bar"><i style={{ width: `${t.fraud_probability * 100}%` }} /></div>{(t.fraud_probability * 100).toFixed(2)}%</div></td>
                <td><span className={`status ${t.is_fraud ? 'fraud' : 'clear'}`}>{t.is_fraud ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}{t.is_fraud ? 'Flagged' : 'Clear'}</span></td>
              </tr>
            ))}</tbody></table></div>
          </section>
        </>
      )}

      {modelInfo && <ModelSection info={modelInfo} channels={activeChannels} />}

      <section id="new-transaction" className="transaction-section">
        <div className="section-intro">
          <p className="eyebrow">MANUAL REVIEW</p>
          <h2>Analyze a new transaction</h2>
          <p>Choose a channel, enter the details, and the matching model scores it and adds it to the live feed.</p>
        </div>
        <div className="form-toolbar">
          <div className="segmented" role="tablist" aria-label="Channel">
            {CHANNELS.map((key) => {
              const Icon = CHANNEL_CONFIG[key].icon
              return <button key={key} type="button" role="tab" aria-selected={channel === key} className={channel === key ? 'active' : ''} onClick={() => { setChannel(key); setLastResult(null) }}><Icon size={15} /> {CHANNEL_CONFIG[key].label}</button>
            })}
          </div>
          <div className="preset-row">
            <button type="button" className="chip" onClick={() => loadExample('normal')}>Load normal example</button>
            <button type="button" className="chip" onClick={() => loadExample('suspicious')}>Load suspicious example</button>
            <button type="button" className={`chip sim ${simulating ? 'on' : ''}`} onClick={() => setSimulating((s) => !s)}>
              {simulating ? <Pause size={13} /> : <Play size={13} />}{simulating ? 'Stop simulation' : 'Simulate live traffic'}
            </button>
          </div>
        </div>
        <form className="transaction-form" onSubmit={submitTransaction}>
          {config.fields.map((field) => (
            <label key={`${channel}-${field.key}`} title={field.hint}>{field.label}
              {field.options ? (
                <select value={String(forms[channel][field.key])} onChange={(event) => updateField(field, event.target.value)}>
                  {field.options.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}
                </select>
              ) : (
                <input type="number" min={field.min} max={field.max} step={field.step} value={forms[channel][field.key]} onChange={(event) => updateField(field, event.target.value)} required />
              )}
            </label>
          ))}
          <button className="button primary submit-button" disabled={submitting}>{submitting ? <LoaderCircle className="spin" size={17} /> : <ShieldCheck size={17} />}{submitting ? 'Analyzing...' : 'Analyze transaction'}</button>
        </form>
        <p className="sim-note">Simulated traffic uses illustrative templates, not real data. Decisions always come from the trained models.</p>
        {lastResult && <div className={`result-banner ${lastResult.is_fraud ? 'result-fraud' : 'result-clear'}`}><div>{lastResult.is_fraud ? <AlertTriangle /> : <CheckCircle2 />}<div><strong>{lastResult.is_fraud ? 'Potential fraud detected' : 'Transaction appears clear'}</strong><span>{CHANNEL_CONFIG[lastResult.channel].label} model · risk score {(lastResult.fraud_probability * 100).toFixed(3)}% · flag threshold {pct(lastResult.threshold)} · saved to live feed</span></div></div><span className="result-amount">{formatMoney(lastResult.amount)}</span></div>}
      </section>
      <footer><span>FraudShield Intelligence</span><span>Protected by real-time ML analysis</span></footer>
    </main>
  )
}

/* --------------------------------------------------- model transparency */
function ModelSection({ info, channels }: { info: ModelInfo; channels: Channel[] }) {
  return (
    <section className="model-section">
      <div className="section-intro">
        <p className="eyebrow"><Cpu size={13} /> MODEL TRANSPARENCY</p>
        <h2>How each model performs</h2>
        <p>Scores come from a held-out test set the models never saw during training. Models were chosen on a separate validation split.</p>
      </div>
      <div className={`model-grid ${channels.length === 1 ? 'single' : ''}`}>
        {channels.map((key) => <ModelCard key={key} channel={key} info={info.channels[key]} comparison={info.comparison.filter((row) => row.Dataset === key)} />)}
      </div>
    </section>
  )
}

function ModelCard({ channel, info, comparison }: { channel: Channel; info: ChannelModelInfo; comparison: ComparisonRow[] }) {
  const config = CHANNEL_CONFIG[channel]
  const Icon = config.icon
  const maxImportance = Math.max(...info.top_features.map((f) => f.importance), 1e-9)
  const cm = info.confusion_matrix

  if (!info.available) {
    return (
      <div className="panel model-card">
        <div className="panel-heading"><div><p className="eyebrow">{config.dataset.toUpperCase()}</p><h2>{config.label}</h2></div><Icon size={21} /></div>
        <div className="error-banner"><TriangleAlert size={18} /> This model is offline{info.error ? `: ${info.error}` : '.'}</div>
      </div>
    )
  }

  return (
    <div className="panel model-card">
      <div className="panel-heading"><div><p className="eyebrow">{config.dataset.toUpperCase()} · {info.model}</p><h2>{config.label}</h2></div><Icon size={21} /></div>
      <div className="kpi-row">
        <div><span>Precision</span><strong>{pct(info.metrics?.precision)}</strong></div>
        <div><span>Recall</span><strong>{pct(info.metrics?.recall)}</strong></div>
        <div><span>F1</span><strong>{pct(info.metrics?.f1)}</strong></div>
        <div><span>PR-AUC</span><strong>{num3(info.metrics?.pr_auc)}</strong></div>
      </div>
      {cm && (
        <p className="model-note">
          On {info.rows ? info.rows.test.toLocaleString() : 'the'} test transactions it caught <strong>{cm.tp.toLocaleString()}</strong> of {(cm.tp + cm.fn).toLocaleString()} frauds
          and raised <strong>{cm.fp.toLocaleString()}</strong> false alarms. Flag threshold: {pct(info.threshold)}.
        </p>
      )}

      <h3>What drives its decisions</h3>
      <div className="importance-list">
        {info.top_features.map((f) => (
          <div className="importance-row" key={f.feature}>
            <span>{cleanFeature(f.feature)}</span>
            <div className="imp-bar"><i style={{ width: `${(f.importance / maxImportance) * 100}%` }} /></div>
          </div>
        ))}
      </div>

      <h3>Models compared (test set)</h3>
      <div className="table-wrap">
        <table className="compare-table">
          <thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th></tr></thead>
          <tbody>
            {comparison.map((row) => (
              <tr key={row.Model} className={row.Model === info.model ? 'selected' : ''}>
                <td>{row.Model}{row.Model === info.model && <small> · deployed</small>}</td>
                <td>{pct(row['Precision@thr'])}</td>
                <td>{pct(row['Recall@thr'])}</td>
                <td>{pct(row['F1@thr'])}</td>
                <td>{num3(row['PR-AUC'])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Metric({ icon, label, value, note, tone = 'default' }: { icon: React.ReactNode; label: string; value: string; note: string; tone?: string }) {
  return <div className="metric-card"><div className={`metric-icon ${tone}`}>{icon}</div><div><span>{label}</span><strong>{value}</strong><small className={tone}>{tone === 'danger' ? <ArrowUpRight size={13} /> : tone === 'safe' ? <ArrowDownRight size={13} /> : null}{note}</small></div></div>
}

export default App