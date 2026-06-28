import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import StatusBadge from '../components/StatusBadge.jsx'
import { buildDistributionSeries, buildLossSeries, buildResourceSeries, pickPrimaryEvaluation, summarizeMetrics } from '../lib/metrics.js'
import { fetchRunDetails } from '../services/api.js'

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—'
}

function ChartPlaceholder({ title, message }) {
  return (
    <section className="chart-card">
      <h3>{title}</h3>
      <p className="panel-subtext">{message}</p>
    </section>
  )
}

export default function RunDetailsPage() {
  const { runId } = useParams()
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadDetails = async () => {
      try {
        setLoading(true)
        const details = await fetchRunDetails(runId)
        setPayload(details)
        setError('')
      } catch (loadError) {
        setError(loadError.response?.data?.detail ?? loadError.message ?? 'Unable to load run details.')
      } finally {
        setLoading(false)
      }
    }

    loadDetails()
  }, [runId])

  if (loading) {
    return (
      <section className="panel">
        <h2>Loading run details…</h2>
      </section>
    )
  }

  if (error) {
    return <div className="error-banner">{error}</div>
  }

  const run = payload?.run ?? {}
  const runMetrics = payload?.run_metrics ?? null
  const { splitName, payload: evaluationPayload } = pickPrimaryEvaluation(runMetrics)
  const evaluationMetrics = summarizeMetrics(evaluationPayload?.metrics)
  const lossSeries = buildLossSeries(runMetrics?.training_loss)
  const resourceSeries = buildResourceSeries(runMetrics?.resource_usage)
  const distributionSeries = buildDistributionSeries(evaluationPayload)

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Run Details</div>
        <div className="hero-copy">
          <h2>{run.nickname || `Run ${run.id}`}</h2>
          <p>
            Database metadata and decoupled JSON artifacts are merged here so the frontend can render
            charts without any static backend PNG generation.
          </p>
        </div>
        <div className="hero-actions">
          <StatusBadge status={run.status} />
          <Link className="button secondary" to="/">
            Back to dashboard
          </Link>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Run metadata</h2>
            <p className="panel-subtext">Backed by SQLite plus the filesystem artifact bundle.</p>
          </div>
        </div>

        <div className="detail-grid">
          <article className="metric-card">
            <h3>Dataset</h3>
            <strong>{run.dataset_name || '—'}</strong>
          </article>
          <article className="metric-card">
            <h3>Started</h3>
            <strong style={{ fontSize: '1.1rem' }}>{run.start_time || '—'}</strong>
          </article>
          <article className="metric-card">
            <h3>Accuracy</h3>
            <strong>{formatMetric(run.accuracy)}</strong>
          </article>
          <article className="metric-card">
            <h3>F1</h3>
            <strong>{formatMetric(run.f1_score)}</strong>
          </article>
        </div>
      </section>

      {runMetrics ? null : (
        <div className="error-banner">This run does not have a `run_metrics.json` artifact yet.</div>
      )}

      {evaluationMetrics.length ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Evaluation snapshot</h2>
              <p className="panel-subtext">Primary split: {splitName || 'unavailable'}</p>
            </div>
          </div>
          <div className="metric-grid">
            {evaluationMetrics.map((entry) => (
              <article className="metric-card" key={entry.key}>
                <h3>{entry.label}</h3>
                <strong>{entry.value === null ? '—' : entry.value.toFixed(4)}</strong>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <div className="charts-grid">
        {lossSeries.length ? (
          <section className="chart-card">
            <h3>Training loss curve</h3>
            <p className="panel-subtext">Rendered from `run_metrics.json.training_loss`.</p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(69, 50, 35, 0.12)" />
                  <XAxis dataKey="step" stroke="#685e54" />
                  <YAxis stroke="#685e54" />
                  <Tooltip />
                  <Line type="monotone" dataKey="loss" stroke="#be5c2b" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        ) : (
          <ChartPlaceholder title="Training loss curve" message="This artifact does not include training loss samples." />
        )}

        {resourceSeries.length ? (
          <section className="chart-card">
            <h3>RAM / VRAM usage</h3>
            <p className="panel-subtext">Time-series built from `run_metrics.json.resource_usage`.</p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={resourceSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(69, 50, 35, 0.12)" />
                  <XAxis dataKey="time" stroke="#685e54" label={{ value: 'Seconds', position: 'insideBottomRight', offset: -6 }} />
                  <YAxis stroke="#685e54" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="ramUsageGb" name="RAM (GB)" stroke="#2f7b74" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="gpuMemUsedGb" name="VRAM (GB)" stroke="#be5c2b" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        ) : (
          <ChartPlaceholder title="RAM / VRAM usage" message="Resource telemetry is unavailable for this run." />
        )}

        {distributionSeries.length ? (
          <section className="chart-card">
            <h3>Anomaly score distributions</h3>
            <p className="panel-subtext">Histogram buckets derived from `all_probs` and `all_labels`.</p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(69, 50, 35, 0.12)" />
                  <XAxis dataKey="range" stroke="#685e54" interval={1} angle={-30} textAnchor="end" height={72} />
                  <YAxis stroke="#685e54" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="normal" name="Normal" fill="#2f7b74" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="anomaly" name="Anomaly" fill="#be5c2b" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        ) : (
          <ChartPlaceholder title="Anomaly score distributions" message="Probability outputs were not present in the stored evaluation payload." />
        )}
      </div>
    </div>
  )
}