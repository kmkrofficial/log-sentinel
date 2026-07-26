import { useContext } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { JobContext } from '../context/JobContext.jsx'
import useJobStatus from '../hooks/useJobStatus.js'
import StatusBadge from './StatusBadge.jsx'

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)}%` : '0%'
}

const TERMINAL_STATUSES = new Set(['completed', 'failed'])

const START_PATHS = {
  setup: '/run?step=setup',
  training: '/train',
  inference: '/inference',
  prep: '/data-prep',
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

function getJobCategory(job, jobId, activeSetupJob, activeTrainingJob, activeInferenceJob, activePrepJob) {
  const jobType = String(job?.job_type ?? '').toLowerCase()

  if (jobType.includes('setup')) {
    return 'setup'
  }
  if (jobType.includes('train')) {
    return 'training'
  }
  if (jobType.includes('infer')) {
    return 'inference'
  }
  if (jobType.includes('prep')) {
    return 'prep'
  }
  if (jobId === activeSetupJob) {
    return 'setup'
  }
  if (jobId === activeTrainingJob) {
    return 'training'
  }
  if (jobId === activeInferenceJob) {
    return 'inference'
  }
  if (jobId === activePrepJob) {
    return 'prep'
  }

  return null
}

export default function LiveJobPanel({ jobId, jobType = null, onClear }) {
  const navigate = useNavigate()
  const {
    activeSetupJob,
    setActiveSetupJob,
    activeTrainingJob,
    setActiveTrainingJob,
    activeInferenceJob,
    setActiveInferenceJob,
    activePrepJob,
    setActivePrepJob,
  } = useContext(JobContext)
  const { job, loading, error } = useJobStatus(jobId)

  const jobCategory = jobType ?? getJobCategory(
    job,
    jobId,
    activeSetupJob,
    activeTrainingJob,
    activeInferenceJob,
    activePrepJob,
  )
  const normalizedStatus = String(job?.status ?? '').toLowerCase()
  const isTerminal = Boolean(job?.done) || TERMINAL_STATUSES.has(normalizedStatus)

  const clearJobAndStartNew = () => {
    if (onClear) {
      onClear()
      return
    }

    if (jobCategory === 'setup' && activeSetupJob === jobId) {
      setActiveSetupJob(null)
    } else if (jobCategory === 'training' && activeTrainingJob === jobId) {
      setActiveTrainingJob(null)
    } else if (jobCategory === 'inference' && activeInferenceJob === jobId) {
      setActiveInferenceJob(null)
    } else if (jobCategory === 'prep' && activePrepJob === jobId) {
      setActivePrepJob(null)
    }

    navigate(START_PATHS[jobCategory] ?? '/')
  }

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
    return (
      <div className="page-grid">
        <div className="error-banner">{error}</div>
        <div className="panel-actions">
          <button className="button primary" type="button" onClick={clearJobAndStartNew}>
            Clear Job and Start New
          </button>
        </div>
      </div>
    )
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

        <div className="panel-actions" style={{ marginTop: '1rem' }}>
          {job.run_id ? (
            <Link className="button secondary" to={`/runs/${job.run_id}`}>
              Open completed run
            </Link>
          ) : null}
          {isTerminal ? (
            <button className="button primary" type="button" onClick={clearJobAndStartNew}>
              Clear Job and Start New
            </button>
          ) : null}
        </div>

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