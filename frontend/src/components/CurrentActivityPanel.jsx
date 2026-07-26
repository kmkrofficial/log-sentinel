import { Link } from 'react-router-dom'

import useJobStatus from '../hooks/useJobStatus.js'
import StatusBadge from './StatusBadge.jsx'

function ActivityCard({ title, jobId, stepPath }) {
  const { job, loading, error } = useJobStatus(jobId)

  return (
    <article className="activity-card">
      <div className="activity-card-heading">
        <span>{title}</span>
        <StatusBadge status={job?.status ?? (loading ? 'PENDING' : 'FAILED')} />
      </div>
      <strong>{loading ? 'Connecting...' : job?.status ?? 'Unavailable'}</strong>
      <p>{error ?? job?.latest_logs?.at(-1) ?? 'No activity message received yet.'}</p>
      <div className="panel-actions">
        <Link className="button secondary" to={stepPath}>Open workflow</Link>
        <Link className="button ghost" to={`/jobs/${jobId}`}>Job details</Link>
      </div>
    </article>
  )
}

export default function CurrentActivityPanel({ jobs }) {
  const activeJobs = jobs.filter((job) => job.jobId)

  if (!activeJobs.length) {
    return (
      <section className="workflow-surface empty-state">
        <h3>No active application jobs</h3>
        <p>Provisioning, preparation, training, and evaluation jobs will appear here while they run.</p>
      </section>
    )
  }

  return (
    <section className="workflow-surface">
      <div className="surface-heading">
        <div>
          <h3>Current activity</h3>
          <p>Every active application job remains visible here while you inspect historical results.</p>
        </div>
      </div>
      <div className="activity-grid">
        {activeJobs.map((job) => <ActivityCard key={job.jobId} {...job} />)}
      </div>
    </section>
  )
}