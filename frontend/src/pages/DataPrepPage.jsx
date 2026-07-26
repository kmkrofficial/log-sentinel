import { useContext, useEffect, useState } from 'react'

import LiveJobPanel from '../components/LiveJobPanel.jsx'
import { JobContext } from '../context/JobContext.jsx'
import { fetchDatasets, startDataPrep } from '../services/api.js'

export default function DataPrepPage() {
  const { activePrepJob, setActivePrepJob } = useContext(JobContext)
  const [datasets, setDatasets] = useState([])
  const [datasetName, setDatasetName] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (activePrepJob) {
      setLoadingDatasets(false)
      return undefined
    }

    const loadDatasets = async () => {
      try {
        setLoadingDatasets(true)
        const payload = await fetchDatasets()
        const nextDatasets = payload.datasets ?? []
        setDatasets(nextDatasets)
        setDatasetName((currentDatasetName) => currentDatasetName || nextDatasets[0] || '')
        setError('')
      } catch (loadError) {
        setError(loadError.response?.data?.detail ?? loadError.message ?? 'Unable to load datasets.')
      } finally {
        setLoadingDatasets(false)
      }
    }

    loadDatasets()
    return undefined
  }, [activePrepJob])

  const handleRunPrep = async (event) => {
    event.preventDefault()

    if (!datasetName) {
      setError('Select a dataset before starting the preprocessing pipeline.')
      return
    }

    try {
      setSubmitting(true)
      setError('')

      const payload = await startDataPrep({ dataset_name: datasetName })
      setActivePrepJob(payload.job_id)
    } catch (submitError) {
      setError(submitError.response?.data?.detail ?? submitError.message ?? 'Unable to start data preparation job.')
    } finally {
      setSubmitting(false)
    }
  }

  if (activePrepJob) {
    return <LiveJobPanel jobId={activePrepJob} jobType="prep" onClear={() => setActivePrepJob(null)} />
  }

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Prepare Data</div>
        <div className="hero-copy">
          <h2>Run dataset preparation without losing the active pipeline view.</h2>
          <p>Select a dataset to launch its preprocessing job through the backend control plane.</p>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Preprocessing pipeline</h2>
            <p className="panel-subtext">This maps directly to `POST /api/data-prep`.</p>
          </div>
        </div>

        <form className="page-grid" onSubmit={handleRunPrep}>
          <div className="field">
            <label htmlFor="prepDatasetName">Dataset</label>
            <select
              id="prepDatasetName"
              value={datasetName}
              onChange={(event) => setDatasetName(event.target.value)}
              disabled={loadingDatasets || submitting}
            >
              {datasets.map((dataset) => (
                <option key={dataset} value={dataset}>
                  {dataset}
                </option>
              ))}
            </select>
            <div className="field-hint">Loaded from `GET /api/datasets`.</div>
          </div>

          <div className="panel-actions">
            <button className="button primary" type="submit" disabled={submitting || loadingDatasets || !datasetName}>
              {submitting ? 'Submitting...' : 'Run Preprocessing Pipeline'}
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}