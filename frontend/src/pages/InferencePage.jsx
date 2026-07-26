import { useContext, useEffect, useState } from 'react'

import LiveJobPanel from '../components/LiveJobPanel.jsx'
import PrecheckPanel from '../components/PrecheckPanel.jsx'
import { JobContext } from '../context/JobContext.jsx'
import { fetchDatasets, runPrecheck, startInference } from '../services/api.js'

export default function InferencePage() {
  const { activeInferenceJob, setActiveInferenceJob } = useContext(JobContext)
  const [datasets, setDatasets] = useState([])
  const [datasetName, setDatasetName] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [checkingPrecheck, setCheckingPrecheck] = useState(false)
  const [precheckReport, setPrecheckReport] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (activeInferenceJob) {
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
  }, [activeInferenceJob])

  const resetPrecheck = () => {
    setPrecheckReport(null)
  }

  const handleRunPrecheck = async () => {
    if (!datasetName || !modelPath.trim()) {
      setError('Select a dataset and provide a trained model path before running pre-checks.')
      return
    }

    try {
      setCheckingPrecheck(true)
      setError('')
      const report = await runPrecheck({
        phase: 'inference',
        dataset_name: datasetName,
        model_run_path: modelPath.trim(),
      })
      setPrecheckReport(report)
    } catch (precheckError) {
      setError(precheckError.response?.data?.detail ?? precheckError.message ?? 'Unable to run inference pre-checks.')
    } finally {
      setCheckingPrecheck(false)
    }
  }

  const handleRunInference = async (event) => {
    event.preventDefault()

    if (!datasetName || !modelPath.trim()) {
      setError('Select a dataset and provide a trained model path before starting inference.')
      return
    }

    if (!precheckReport?.ready) {
      setError('Run pre-checks and resolve every failed requirement before starting inference.')
      return
    }

    try {
      setSubmitting(true)
      setError('')

      const payload = await startInference({
        dataset_name: datasetName,
        model_run_path: modelPath.trim(),
      })

      setActiveInferenceJob(payload.job_id)
    } catch (submitError) {
      const detail = submitError.response?.data?.detail
      if (detail?.precheck) {
        setPrecheckReport(detail.precheck)
      }
      setError(typeof detail === 'string' ? detail : detail?.message ?? submitError.message ?? 'Unable to start inference job.')
    } finally {
      setSubmitting(false)
    }
  }

  if (activeInferenceJob) {
    return (
      <LiveJobPanel
        jobId={activeInferenceJob}
        jobType="inference"
        onClear={() => setActiveInferenceJob(null)}
      />
    )
  }

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="eyebrow">Inference</div>
        <div className="hero-copy">
          <h2>Evaluate a saved model against a prepared log dataset.</h2>
          <p>Submit a model run directory and keep its live evaluation status visible across navigation.</p>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Run inference</h2>
            <p className="panel-subtext">This maps directly to `POST /api/inference`.</p>
          </div>
        </div>

        <form className="page-grid" onSubmit={handleRunInference}>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="inferenceDatasetName">Dataset</label>
              <select
                id="inferenceDatasetName"
                value={datasetName}
                onChange={(event) => {
                  setDatasetName(event.target.value)
                  resetPrecheck()
                }}
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
              <label htmlFor="modelPath">Model run path</label>
              <input
                id="modelPath"
                type="text"
                value={modelPath}
                onChange={(event) => {
                  setModelPath(event.target.value)
                  resetPrecheck()
                }}
                placeholder="/path/to/trained-run or output_model"
                disabled={submitting}
              />
              <div className="field-hint">Use a completed run directory or its `output_model` directory.</div>
            </div>
          </div>

          <PrecheckPanel
            report={precheckReport}
            checking={checkingPrecheck}
            onRun={handleRunPrecheck}
            disabled={submitting || loadingDatasets || !datasetName || !modelPath.trim()}
          />

          <div className="panel-actions">
            <button className="button primary" type="submit" disabled={submitting || loadingDatasets || !precheckReport?.ready}>
              {submitting ? 'Submitting...' : 'Start Inference'}
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}