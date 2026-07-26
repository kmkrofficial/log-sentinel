const STATUS_LABELS = {
  passed: 'Passed',
  warning: 'Warning',
  failed: 'Failed',
}

export default function PrecheckPanel({ report, checking, onRun, disabled }) {
  const summary = report?.ready ? 'Ready to execute' : report ? 'Action required' : 'Not run'
  const summaryStatus = report?.ready ? 'passed' : report ? 'failed' : 'warning'

  return (
    <section className="precheck-phase" aria-live="polite">
      <div className="precheck-header">
        <div>
          <div className="eyebrow">Pre-check phase</div>
          <h3>Execution requirements</h3>
        </div>
        <span className={`precheck-summary is-${summaryStatus}`}>{summary}</span>
      </div>

      <p className="panel-subtext">Validate the environment and selected inputs before starting the workload.</p>

      <div className="panel-actions">
        <button className="button secondary" type="button" onClick={onRun} disabled={disabled || checking}>
          {checking ? 'Running pre-checks...' : 'Run pre-checks'}
        </button>
      </div>

      {report ? (
        <ul className="precheck-list">
          {report.checks.map((check) => (
            <li className={`precheck-item is-${check.status}`} key={check.key}>
              <div className="precheck-item-heading">
                <strong>{check.label}</strong>
                <span>{STATUS_LABELS[check.status] ?? check.status}</span>
              </div>
              <p>{check.detail}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  )
}