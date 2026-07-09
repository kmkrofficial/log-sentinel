import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import RunTable from '../components/RunTable.jsx'
import { fetchRunHistory } from '../services/api.js'

function countByStatus(runs, status) {
  return runs.filter((run) => String(run.status ?? '').toLowerCase().includes(status)).length
}

function averageMetric(runs, key) {
  const values = runs.map((run) => Number(run[key])).filter((value) => Number.isFinite(value))
  if (!values.length) {
    return '—'
  }
  return (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3)
}

export default function DashboardPage() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadRuns = async () => {
    try {
      setLoading(true)
      const payload = await fetchRunHistory()
      setRuns(payload.runs ?? [])
      setError('')
    } catch (loadError) {
      setError(loadError.response?.data?.detail ?? loadError.message ?? 'Unable to load runs.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadRuns()
  }, [])

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Monorepo Dashboard</div>
        <div className="hero-copy">
          <h2>Track every experiment from launch to forensic replay.</h2>
          <p>
            The React control plane sits on top of the FastAPI backend and renders historical runs,
            live jobs, and the decoupled `run_metrics.json` artifacts.
          </p>
        </div>
        <div className="hero-actions">
          <Link className="button primary" to="/train">
            Start a training run
          </Link>
          <button className="button secondary" type="button" onClick={loadRuns} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh runs'}
          </button>
        </div>
        <div className="kpi-grid">
          <article className="kpi-card">
            <h3>Total runs</h3>
            <strong>{runs.length}</strong>
          </article>
          <article className="kpi-card">
            <h3>Completed</h3>
            <strong>{countByStatus(runs, 'complete')}</strong>
          </article>
          <article className="kpi-card">
            <h3>Failed</h3>
            <strong>{countByStatus(runs, 'fail')}</strong>
          </article>
          <article className="kpi-card">
            <h3>Avg accuracy</h3>
            <strong>{averageMetric(runs, 'accuracy')}</strong>
          </article>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Historical run ledger</h2>
            <p className="panel-subtext">Every training and inference execution stored by the backend database.</p>
          </div>
        </div>
        <RunTable runs={runs} />
      </section>
    </div>
  )
}