import { Link } from 'react-router-dom'

import StatusBadge from './StatusBadge.jsx'

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—'
}

function formatDuration(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}s` : '—'
}

export default function RunTable({ runs }) {
  if (!runs.length) {
    return (
      <div className="empty-state">
        <h3>No runs yet</h3>
        <p>Kick off a training job to populate the run ledger and unlock charts.</p>
      </div>
    )
  }

  return (
    <div className="table-wrap">
      <table className="run-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Dataset</th>
            <th>Started</th>
            <th>Status</th>
            <th>Accuracy</th>
            <th>F1</th>
            <th>Duration</th>
            <th>Inspect</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <div className="run-title">
                  <strong>{run.nickname || `Run ${run.id}`}</strong>
                  <small className="muted">ID {run.id}</small>
                </div>
              </td>
              <td>{run.dataset_name || '—'}</td>
              <td>{run.start_time || '—'}</td>
              <td>
                <StatusBadge status={run.status} />
              </td>
              <td>{formatMetric(run.accuracy)}</td>
              <td>{formatMetric(run.f1_score)}</td>
              <td>{formatDuration(run.total_run_time_sec)}</td>
              <td>
                <Link className="button secondary" to={`/runs/${run.id}`}>
                  Open
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}