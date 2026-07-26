import { useEffect, useState } from 'react'

import LiveJobPanel from './LiveJobPanel.jsx'

function formatBytes(value) {
  if (!Number.isFinite(Number(value))) {
    return 'Unknown size'
  }
  const gib = Number(value) / 1024 ** 3
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(Number(value) / 1024 ** 2).toFixed(1)} MiB`
}

function AssetStatus({ ready, blockedText }) {
  const status = blockedText ? 'blocked' : ready ? 'ready' : 'missing'
  const label = blockedText ? 'Action required' : ready ? 'Ready' : 'Missing'
  return <span className={`asset-status is-${status}`}>{label}</span>
}

export default function ConfigurationSetupStep({
  status,
  loading,
  error,
  activeJobId,
  onClearJob,
  onRefresh,
  onProvision,
}) {
  const [selectedDatasets, setSelectedDatasets] = useState([])
  const [selectedModels, setSelectedModels] = useState([])
  const [force, setForce] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (!status) {
      return
    }
    setSelectedDatasets((selected) => selected.filter((datasetId) => status.datasets.some((dataset) => dataset.id === datasetId)))
    setSelectedModels((selected) => selected.filter((modelKey) => status.models.some((model) => model.key === modelKey)))
  }, [status])

  if (activeJobId) {
    return <LiveJobPanel jobId={activeJobId} jobType="setup" onClear={onClearJob} />
  }

  const toggleSelection = (setter, selected, value) => {
    setter(selected.includes(value) ? selected.filter((item) => item !== value) : [...selected, value])
  }

  const provision = async (datasets, models) => {
    if (!datasets.length && !models.length) {
      return
    }
    if (force && !window.confirm('Replace the selected managed assets? Existing prepared CSV outputs for selected datasets will be removed and must be regenerated.')) {
      return
    }
    try {
      setSubmitting(true)
      await onProvision({ datasets, models, force })
    } finally {
      setSubmitting(false)
    }
  }

  const missingDatasets = status?.datasets.filter((dataset) => !dataset.raw.ready).map((dataset) => dataset.id) ?? []
  const missingModels = status?.models.filter((model) => !model.ready).map((model) => model.key) ?? []
  const runtimeChecks = status?.runtime_checks ?? []

  return (
    <div className="workflow-step-content">
      <section className="workflow-intro">
        <div>
          <div className="eyebrow">Step 1</div>
          <h2>Configuration and setup</h2>
          <p>Verify the environment, provision verified archives, and make the shared model cache ready before data preparation begins.</p>
        </div>
        <div className="panel-actions">
          <button className="button secondary" type="button" onClick={onRefresh} disabled={loading || submitting}>
            {loading ? 'Scanning...' : 'Refresh inventory'}
          </button>
          <button
            className="button primary"
            type="button"
            onClick={() => provision(missingDatasets, missingModels)}
            disabled={submitting || (!missingDatasets.length && !missingModels.length)}
          >
            Provision all missing
          </button>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="setup-summary-band">
        <article>
          <span>HF_TOKEN</span>
          <strong>{status?.hf_token_configured ? 'Configured' : 'Not configured'}</strong>
          <small>The browser never receives this value.</small>
        </article>
        <article>
          <span>Setup storage</span>
          <strong>{formatBytes(status?.storage?.free_bytes)} free</strong>
          <small>{status?.downloads_path ?? 'Checking download directory...'}</small>
        </article>
        <article>
          <span>Backend package checks</span>
          <strong>{runtimeChecks.filter((check) => check.status === 'failed').length ? 'Needs attention' : 'Ready'}</strong>
          <small>{runtimeChecks.length} environment checks reported</small>
        </article>
      </section>

      <section className="workflow-surface">
        <div className="surface-heading">
          <div>
            <h3>Dataset archives</h3>
            <p>Each archive is downloaded with its Zenodo checksum and extracted into a managed raw-data folder.</p>
          </div>
        </div>
        <div className="asset-table">
          {status?.datasets.map((dataset) => (
            <label className="asset-row" key={dataset.id}>
              <input
                type="checkbox"
                checked={selectedDatasets.includes(dataset.id)}
                onChange={() => toggleSelection(setSelectedDatasets, selectedDatasets, dataset.id)}
                disabled={submitting}
              />
              <span className="asset-main">
                <strong>{dataset.display_name}</strong>
                <small>{dataset.archive.filename} · {formatBytes(dataset.archive.expected_size)} · {dataset.preparation_strategy}</small>
              </span>
              <span className="asset-meta">
                <AssetStatus ready={dataset.raw.ready} />
                <small>{dataset.raw.ready ? `${dataset.raw.layout} raw assets found` : 'Archive extraction required'}</small>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section className="workflow-surface">
        <div className="surface-heading">
          <div>
            <h3>Model assets</h3>
            <p>The Llama backbone uses the backend server&apos;s `HF_TOKEN` when a download is required.</p>
          </div>
        </div>
        <div className="asset-table">
          {status?.models.map((model) => (
            <label className="asset-row" key={model.key}>
              <input
                type="checkbox"
                checked={selectedModels.includes(model.key)}
                onChange={() => toggleSelection(setSelectedModels, selectedModels, model.key)}
                disabled={submitting}
              />
              <span className="asset-main">
                <strong>{model.model_id}</strong>
                <small>{model.path}</small>
              </span>
              <span className="asset-meta">
                <AssetStatus ready={model.ready} blockedText={model.requires_hf_token && !status?.hf_token_configured && !model.ready} />
                <small>{model.requires_hf_token ? 'Gated Hugging Face model' : 'Public Hugging Face model'}</small>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section className="workflow-surface setup-runtime-checks">
        <div className="surface-heading">
          <div>
            <h3>Runtime readiness</h3>
            <p>System-level blockers remain visible here; privileged package installation is intentionally not performed by the browser.</p>
          </div>
        </div>
        <ul className="precheck-list">
          {runtimeChecks.map((check) => (
            <li className={`precheck-item is-${check.status}`} key={check.key}>
              <div className="precheck-item-heading">
                <strong>{check.label}</strong>
                <span>{check.status}</span>
              </div>
              <p>{check.detail}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="workflow-actions">
        <label className="checkbox-option" htmlFor="forceProvision">
          <input id="forceProvision" type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} disabled={submitting} />
          <span>Replace selected managed raw assets or model snapshots and invalidate affected prepared splits</span>
        </label>
        <button
          className="button primary"
          type="button"
          disabled={submitting || (!selectedDatasets.length && !selectedModels.length)}
          onClick={() => provision(selectedDatasets, selectedModels)}
        >
          {submitting ? 'Submitting setup...' : 'Provision selected assets'}
        </button>
      </section>
    </div>
  )
}