import { useContext, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import CurrentActivityPanel from '../components/CurrentActivityPanel.jsx'
import RunTable from '../components/RunTable.jsx'
import { JobContext } from '../context/JobContext.jsx'
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

export default function ResultsPage() {
  const {
    activeSetupJob,
    activePrepJob,
    activeTrainingJob,
    activeInferenceJob,
  } = useContext(JobContext)
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
      <section className="results-header">
        <div>
          <div className="eyebrow">Results</div>
          <h2>Watch active work and compare every completed run.</h2>
          <p>Live application activity stays separate from the experiment ledger, so a running job never disappears while you inspect prior evidence.</p>
        </div>
        <div className="panel-actions">
          <Link className="button primary" to="/run">Run model workflow</Link>
          <button className="button secondary" type="button" onClick={loadRuns} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh results'}
          </button>
        </div>
      </section>

      <CurrentActivityPanel
        jobs={[
          { title: 'Configuration setup', jobId: activeSetupJob, stepPath: '/run?step=setup' },
          { title: 'Data preparation', jobId: activePrepJob, stepPath: '/run?step=preparation' },
          { title: 'Model training', jobId: activeTrainingJob, stepPath: '/run?step=training' },
          { title: 'Model evaluation', jobId: activeInferenceJob, stepPath: '/run?step=training&mode=inference' },
        ]}
      />

      <section className="results-kpi-grid">
        <article><span>Total runs</span><strong>{runs.length}</strong></article>
        <article><span>Completed</span><strong>{countByStatus(runs, 'complete')}</strong></article>
        <article><span>Failed</span><strong>{countByStatus(runs, 'fail')}</strong></article>
        <article><span>Average accuracy</span><strong>{averageMetric(runs, 'accuracy')}</strong></article>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="workflow-surface">
        <div className="surface-heading">
          <div>
            <h3>Historical run ledger</h3>
            <p>Training and inference summaries stored by the backend database.</p>
          </div>
        </div>
        <RunTable runs={runs} />
      </section>
    </div>
  )
}