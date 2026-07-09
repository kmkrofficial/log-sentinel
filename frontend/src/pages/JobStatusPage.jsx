import { Link, useParams } from 'react-router-dom'

import LiveJobPanel from '../components/LiveJobPanel.jsx'
import useJobStatus from '../hooks/useJobStatus.js'

export default function JobStatusPage() {
  const { jobId } = useParams()
  const { job, loading, error } = useJobStatus(jobId)

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Live Progress</div>
        <div className="hero-copy">
          <h2>Polling the backend control plane every three seconds.</h2>
          <p>
            This view streams job state, progress, and logs from `GET /api/status/{'{job_id}'}` until the
            backend reports a completed or failed terminal state.
          </p>
        </div>
        <div className="hero-actions">
          <Link className="button secondary" to="/">
            Back to dashboard
          </Link>
        </div>
      </section>

      <LiveJobPanel job={job} loading={loading} error={error} />
    </div>
  )
}