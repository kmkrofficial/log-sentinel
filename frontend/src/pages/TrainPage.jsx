import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchDatasets, submitTrainingJob } from '../services/api.js'

const DEFAULT_HYPERPARAMETERS = '{\n  "micro_batch_size": 32,\n  "n_epochs_phase_adapters": 5\n}'

export default function TrainPage() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState([])
  const [datasetName, setDatasetName] = useState('')
  const [hyperparametersText, setHyperparametersText] = useState(DEFAULT_HYPERPARAMETERS)
  const [isTestRun, setIsTestRun] = useState(false)
  const [testRunPercentage, setTestRunPercentage] = useState('0.30')
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        setLoadingDatasets(true)
        const payload = await fetchDatasets()
        const nextDatasets = payload.datasets ?? []
        setDatasets(nextDatasets)
        setDatasetName(nextDatasets[0] ?? '')
      } catch (loadError) {
        setError(loadError.response?.data?.detail ?? loadError.message ?? 'Unable to load datasets.')
      } finally {
        setLoadingDatasets(false)
      }
    }

    loadDatasets()
  }, [])

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!datasetName) {
      setError('Select a dataset before launching a training run.')
      return
    }

    let hyperparameters = null

    try {
      const trimmed = hyperparametersText.trim()
      hyperparameters = trimmed ? JSON.parse(trimmed) : null
    } catch {
      setError('Hyperparameters must be valid JSON.')
      return
    }

    try {
      setSubmitting(true)
      setError('')

      const payload = await submitTrainingJob({
        dataset_name: datasetName,
        hyperparameters,
        is_test_run: isTestRun,
        test_run_percentage: Number(testRunPercentage),
      })

      navigate(`/jobs/${payload.job_id}`)
    } catch (submitError) {
      setError(submitError.response?.data?.detail ?? submitError.message ?? 'Unable to start training job.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Configuration</div>
        <div className="hero-copy">
          <h2>Launch a new backend training run without leaving the browser.</h2>
          <p>
            Choose a dataset, override hyperparameters as JSON, and hand the workload off to the FastAPI
            background executor. The UI will pivot straight into live job telemetry after submission.
          </p>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="split-layout">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Start training</h2>
              <p className="panel-subtext">This maps directly to `POST /api/train`.</p>
            </div>
          </div>

          <form className="page-grid" onSubmit={handleSubmit}>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="datasetName">Dataset</label>
                <select
                  id="datasetName"
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

              <div className="field">
                <label htmlFor="testRunPercentage">Quick test percentage</label>
                <input
                  id="testRunPercentage"
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={testRunPercentage}
                  onChange={(event) => setTestRunPercentage(event.target.value)}
                  disabled={!isTestRun || submitting}
                />
                <div className="field-hint">Used only when quick test mode is enabled.</div>
              </div>
            </div>

            <fieldset className="checkbox-field">
              <legend>Execution mode</legend>
              <label className="checkbox-option" htmlFor="isTestRun">
                <input
                  id="isTestRun"
                  type="checkbox"
                  checked={isTestRun}
                  onChange={(event) => setIsTestRun(event.target.checked)}
                  disabled={submitting}
                />
                <span>Run a reduced quick-test job first</span>
              </label>
            </fieldset>

            <div className="field">
              <label htmlFor="hyperparameters">Hyperparameters JSON</label>
              <textarea
                id="hyperparameters"
                value={hyperparametersText}
                onChange={(event) => setHyperparametersText(event.target.value)}
                disabled={submitting}
              />
              <div className="field-hint">Only include overrides. The backend merges these into dataset defaults.</div>
            </div>

            <div className="panel-actions">
              <button className="button primary" type="submit" disabled={submitting || loadingDatasets || !datasetName}>
                {submitting ? 'Submitting…' : 'Launch training job'}
              </button>
            </div>
          </form>
        </section>

        <aside className="helper-stack">
          <section className="sidebar-card">
            <h3>What happens next</h3>
            <p>
              After submission the backend creates a job ID, stores it in the in-memory job manager, and
              starts the training controller in a background task.
            </p>
          </section>

          <section className="sidebar-card">
            <h3>Available datasets</h3>
            <div className="list-inline">
              {datasets.map((dataset) => (
                <span className="chip" key={dataset}>
                  {dataset}
                </span>
              ))}
            </div>
          </section>

          <section className="sidebar-card">
            <h3>Payload shape</h3>
            <p className="mono">{`{ dataset_name, hyperparameters, is_test_run, test_run_percentage }`}</p>
          </section>
        </aside>
      </div>
    </div>
  )
}