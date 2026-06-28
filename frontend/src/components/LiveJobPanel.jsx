import { Link } from 'react-router-dom'

import StatusBadge from './StatusBadge.jsx'

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)}%` : '0%'
}

function MetricList({ validationMetrics }) {
  if (!validationMetrics || !Object.keys(validationMetrics).length) {
    return null
  }

  return (
    <div className="metric-grid">
      {Object.entries(validationMetrics).map(([key, value]) => (
        <article className="metric-card" key={key}>
          <h3>{key.replaceAll('_', ' ')}</h3>
          <strong>{Number.isFinite(Number(value)) ? Number(value).toFixed(4) : String(value)}</strong>
        </article>
      ))}
    </div>
  )
}

export default function LiveJobPanel({ job, loading, error }) {
  if (loading && !job) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Preparing live telemetry</h2>
            <p className="panel-subtext">Connecting to the backend job monitor.</p>
          </div>
        </div>
      </section>
    )
  }

  if (error && !job) {
    return <div className="error-banner">{error}</div>
  }

  const logs = job?.latest_logs ?? []

  return (
    <div className="page-grid">
      {error ? <div className="error-banner">{error}</div> : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <div className="eyebrow">Live Progress</div>
            <h2>Job {job.job_id}</h2>
            <p className="panel-subtext">{job.job_type} task streaming from FastAPI background execution.</p>
          </div>
          <StatusBadge status={job.status} />
        </div>

        <div className="progress-block">
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${job.progress ?? 0}%` }} />
          </div>
          <div className="progress-meta">
            <strong>{formatPercent(job.progress)}</strong>
            <span>{job.done ? 'Polling stopped: terminal state reached.' : 'Polling every 3 seconds.'}</span>
          </div>
        </div>

        <div className="detail-grid" style={{ marginTop: '1rem' }}>
          <article className="metric-card">
            <h3>Run ID</h3>
            <strong>{job.run_id ?? 'Pending'}</strong>
          </article>
          <article className="metric-card">
            <h3>Execution Dir</h3>
            <strong className="mono" style={{ fontSize: '1rem' }}>{job.execution_dir ?? 'Pending'}</strong>
          </article>
        </div>

        <MetricList validationMetrics={job.validation_metrics} />

        {job.run_id ? (
          <div className="panel-actions" style={{ marginTop: '1rem' }}>
            <Link className="button primary" to={`/runs/${job.run_id}`}>
              Open completed run
            </Link>
          </div>
        ) : null}

        <div className="log-stream" aria-live="polite">
          {logs.length ? (
            logs.map((log, index) => (
              <div className="log-line" key={`${index}-${log.slice(0, 24)}`}>
                {log}
              </div>
            ))
          ) : (
            <div className="log-line">No log messages received yet.</div>
          )}
        </div>
      </section>
    </div>
  )
}