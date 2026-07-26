import { useEffect, useState } from 'react'

import LiveJobPanel from './LiveJobPanel.jsx'

const THUNDERBIRD_DEFAULTS = {
  start_line: '160000000',
  end_line: '170000000',
  window_size: '100',
  step_size: '100',
  oversampling_factor: '10',
}

function AssetStatus({ status }) {
  return <span className={`asset-status is-${status}`}>{status}</span>
}

export default function DatasetPreparationStep({
  setupStatus,
  preparationStatus,
  loading,
  error,
  activeJobId,
  onClearJob,
  onRefresh,
  onStart,
}) {
  const [datasetId, setDatasetId] = useState('')
  const [thunderbirdOptions, setThunderbirdOptions] = useState(THUNDERBIRD_DEFAULTS)
  const [submitting, setSubmitting] = useState(false)

  const statusByDataset = new Map((preparationStatus?.datasets ?? []).map((item) => [item.dataset_name, item]))
  const rawDatasets = setupStatus?.datasets.filter((dataset) => dataset.raw.ready) ?? []
  const selectedStatus = statusByDataset.get(datasetId)

  useEffect(() => {
    if (!datasetId || !rawDatasets.some((dataset) => dataset.id === datasetId)) {
      setDatasetId(rawDatasets[0]?.id ?? '')
    }
  }, [datasetId, rawDatasets])

  if (activeJobId) {
    return <LiveJobPanel jobId={activeJobId} jobType="prep" onClear={onClearJob} />
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!datasetId || !selectedStatus?.eligible) {
      return
    }
    try {
      setSubmitting(true)
      await onStart({
        dataset_name: datasetId,
        options: datasetId === 'Thunderbird'
          ? Object.fromEntries(Object.entries(thunderbirdOptions).map(([key, value]) => [key, Number(value)]))
          : {},
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="workflow-step-content">
      <section className="workflow-intro">
        <div>
          <div className="eyebrow">Step 2</div>
          <h2>Data preparation</h2>
          <p>Convert verified raw archives into the `Content` and `Label` split files required by model training.</p>
        </div>
        <div className="panel-actions">
          <button className="button secondary" type="button" onClick={onRefresh} disabled={loading || submitting}>
            {loading ? 'Scanning...' : 'Refresh preparation state'}
          </button>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="workflow-surface">
        <div className="surface-heading">
          <div>
            <h3>Preparation eligibility</h3>
            <p>Only extracted datasets with a compatible supervised labeling strategy can continue to preparation.</p>
          </div>
        </div>
        <div className="dataset-strategy-grid">
          {(setupStatus?.datasets ?? []).map((dataset) => {
            const prep = statusByDataset.get(dataset.id)
            const prepared = Object.values(dataset.prepared).every((output) => output.ready)
            const state = !dataset.raw.ready ? 'missing' : prep?.eligible ? prepared ? 'ready' : 'available' : 'blocked'
            return (
              <label className={`dataset-strategy-card is-${state}`} key={dataset.id}>
                <input
                  type="radio"
                  name="preparedDataset"
                  value={dataset.id}
                  checked={datasetId === dataset.id}
                  onChange={() => setDatasetId(dataset.id)}
                  disabled={!dataset.raw.ready || submitting}
                />
                <span>
                  <strong>{dataset.display_name}</strong>
                  <small>{dataset.preparation_strategy}</small>
                </span>
                <AssetStatus status={state} />
                <p>{!dataset.raw.ready ? 'Complete configuration setup first.' : prep?.blocker ?? (prepared ? 'Prepared split files are available.' : 'Ready to prepare.')}</p>
              </label>
            )
          })}
        </div>
      </section>

      {datasetId ? (
        <form className="workflow-surface preparation-form" onSubmit={handleSubmit}>
          <div className="surface-heading">
            <div>
              <h3>Prepare {datasetId}</h3>
              <p>{selectedStatus?.blocker ?? 'The selected strategy will create train.csv, validation.csv, and test.csv atomically.'}</p>
            </div>
          </div>

          {datasetId === 'Thunderbird' ? (
            <div className="field-grid">
              {Object.entries(thunderbirdOptions).map(([key, value]) => (
                <div className="field" key={key}>
                  <label htmlFor={`thunderbird-${key}`}>{key.replaceAll('_', ' ')}</label>
                  <input
                    id={`thunderbird-${key}`}
                    type="number"
                    min="1"
                    value={value}
                    onChange={(event) => setThunderbirdOptions({ ...thunderbirdOptions, [key]: event.target.value })}
                    disabled={submitting}
                  />
                </div>
              ))}
            </div>
          ) : null}

          <div className="panel-actions">
            <button className="button primary" type="submit" disabled={submitting || !selectedStatus?.eligible}>
              {submitting ? 'Submitting preparation...' : 'Run preparation'}
            </button>
          </div>
        </form>
      ) : null}
    </div>
  )
}