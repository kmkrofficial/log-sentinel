import { useContext, useEffect, useState } from 'react'

import { JobContext } from '../context/JobContext.jsx'
import { fetchModels, runPrecheck, startInference, submitTrainingJob } from '../services/api.js'
import LiveJobPanel from './LiveJobPanel.jsx'
import PrecheckPanel from './PrecheckPanel.jsx'

const DEFAULT_HYPERPARAMETERS = '{\n  "micro_batch_size": 32,\n  "n_epochs_phase_adapters": 5\n}'

function getPreparedDatasets(setupStatus) {
  return (setupStatus?.datasets ?? [])
    .filter((dataset) => dataset.prepared.train.ready && dataset.prepared.test.ready)
    .map((dataset) => dataset.id)
}

export default function ModelExecutionStep({ setupStatus, mode, onModeChange, onRefresh }) {
  const {
    activeTrainingJob,
    setActiveTrainingJob,
    activeInferenceJob,
    setActiveInferenceJob,
  } = useContext(JobContext)
  const preparedDatasets = getPreparedDatasets(setupStatus)
  const [datasetName, setDatasetName] = useState('')
  const [hyperparametersText, setHyperparametersText] = useState(DEFAULT_HYPERPARAMETERS)
  const [isTestRun, setIsTestRun] = useState(false)
  const [testRunPercentage, setTestRunPercentage] = useState('0.30')
  const [modelPath, setModelPath] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [checkingPrecheck, setCheckingPrecheck] = useState(false)
  const [precheckReport, setPrecheckReport] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!preparedDatasets.includes(datasetName)) {
      setDatasetName(preparedDatasets[0] ?? '')
    }
  }, [datasetName, preparedDatasets])

  useEffect(() => {
    if (mode !== 'inference') {
      return undefined
    }

    const loadModels = async () => {
      try {
        setLoadingModels(true)
        const payload = await fetchModels()
        const nextModels = payload.models ?? []
        setAvailableModels(nextModels)
        setModelPath((currentModelPath) => currentModelPath || nextModels[0] || '')
      } catch (loadError) {
        setError(loadError.response?.data?.detail ?? loadError.message ?? 'Unable to load completed model paths.')
      } finally {
        setLoadingModels(false)
      }
    }

    loadModels()
    return undefined
  }, [mode])

  const resetPrecheck = () => {
    setPrecheckReport(null)
  }

  const handleRunPrecheck = async () => {
    if (!datasetName || (mode === 'inference' && !modelPath.trim())) {
      setError(mode === 'inference' ? 'Choose a prepared dataset and model path before pre-checking.' : 'Choose a prepared dataset before pre-checking.')
      return
    }

    try {
      setCheckingPrecheck(true)
      setError('')
      const report = await runPrecheck({
        phase: mode === 'inference' ? 'inference' : 'training',
        dataset_name: datasetName,
        model_run_path: mode === 'inference' ? modelPath.trim() : null,
        is_test_run: isTestRun,
        test_run_percentage: Number(testRunPercentage),
      })
      setPrecheckReport(report)
    } catch (precheckError) {
      setError(precheckError.response?.data?.detail ?? precheckError.message ?? 'Unable to run execution pre-checks.')
    } finally {
      setCheckingPrecheck(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!precheckReport?.ready) {
      setError('Run pre-checks and resolve every failed requirement before execution.')
      return
    }

    try {
      setSubmitting(true)
      setError('')
      if (mode === 'inference') {
        const payload = await startInference({
          dataset_name: datasetName,
          model_run_path: modelPath.trim(),
          is_test_run: isTestRun,
          test_run_percentage: Number(testRunPercentage),
        })
        setActiveInferenceJob(payload.job_id)
        return
      }

      const hyperparameters = hyperparametersText.trim() ? JSON.parse(hyperparametersText) : null
      const payload = await submitTrainingJob({
        dataset_name: datasetName,
        hyperparameters,
        is_test_run: isTestRun,
        test_run_percentage: Number(testRunPercentage),
      })
      setActiveTrainingJob(payload.job_id)
    } catch (submitError) {
      const detail = submitError.response?.data?.detail
      if (detail?.precheck) {
        setPrecheckReport(detail.precheck)
      }
      setError(typeof detail === 'string' ? detail : detail?.message ?? submitError.message ?? 'Unable to start model execution.')
    } finally {
      setSubmitting(false)
    }
  }

  if (mode === 'training' && activeTrainingJob) {
    return <LiveJobPanel jobId={activeTrainingJob} jobType="training" onClear={() => { setActiveTrainingJob(null); onRefresh() }} />
  }
  if (mode === 'inference' && activeInferenceJob) {
    return <LiveJobPanel jobId={activeInferenceJob} jobType="inference" onClear={() => { setActiveInferenceJob(null); onRefresh() }} />
  }

  return (
    <div className="workflow-step-content">
      <section className="workflow-intro">
        <div>
          <div className="eyebrow">Step 3</div>
          <h2>Model training</h2>
          <p>Run the final environment pre-check, then train a new model or evaluate a completed one against prepared splits.</p>
        </div>
      </section>

      <div className="mode-toggle" role="group" aria-label="Model execution mode">
        <button className={mode === 'training' ? 'is-active' : ''} type="button" onClick={() => { onModeChange('training'); resetPrecheck() }}>
          Train model
        </button>
        <button className={mode === 'inference' ? 'is-active' : ''} type="button" onClick={() => { onModeChange('inference'); resetPrecheck() }}>
          Evaluate model
        </button>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      {!preparedDatasets.length ? (
        <section className="workflow-surface empty-state">
          <h3>No prepared datasets are ready</h3>
          <p>Return to Step 2 and create train.csv and test.csv for an eligible dataset before starting model execution.</p>
        </section>
      ) : (
        <form className="workflow-surface execution-form" onSubmit={handleSubmit}>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="executionDataset">Prepared dataset</label>
              <select
                id="executionDataset"
                value={datasetName}
                onChange={(event) => {
                  setDatasetName(event.target.value)
                  resetPrecheck()
                }}
                disabled={submitting}
              >
                {preparedDatasets.map((dataset) => <option key={dataset} value={dataset}>{dataset}</option>)}
              </select>
              <div className="field-hint">Only datasets with train.csv and test.csv are available here.</div>
            </div>

            <div className="field">
              <label htmlFor="executionTestFraction">Quick test percentage</label>
              <input
                id="executionTestFraction"
                type="number"
                min="0.01"
                max="1"
                step="0.01"
                value={testRunPercentage}
                onChange={(event) => {
                  setTestRunPercentage(event.target.value)
                  resetPrecheck()
                }}
                disabled={!isTestRun || submitting}
              />
              <div className="field-hint">Used only when quick test mode is enabled.</div>
            </div>
          </div>

          {mode === 'inference' ? (
            <div className="field">
              <label htmlFor="executionModelPath">Completed model path</label>
              <input
                id="executionModelPath"
                list="completed-models"
                type="text"
                value={modelPath}
                onChange={(event) => {
                  setModelPath(event.target.value)
                  resetPrecheck()
                }}
                disabled={submitting || loadingModels}
                placeholder="/path/to/run or output_model"
              />
              <datalist id="completed-models">
                {availableModels.map((model) => <option key={model} value={model} />)}
              </datalist>
              <div className="field-hint">The pre-check verifies the projector, classifier, and LoRA adapter artifacts.</div>
            </div>
          ) : (
            <div className="field">
              <label htmlFor="executionHyperparameters">Hyperparameter overrides JSON</label>
              <textarea
                id="executionHyperparameters"
                value={hyperparametersText}
                onChange={(event) => {
                  setHyperparametersText(event.target.value)
                  resetPrecheck()
                }}
                disabled={submitting}
              />
              <div className="field-hint">Only include overrides. Dataset defaults are merged on the backend.</div>
            </div>
          )}

          <label className="checkbox-option" htmlFor="executionQuickRun">
            <input
              id="executionQuickRun"
              type="checkbox"
              checked={isTestRun}
              onChange={(event) => {
                setIsTestRun(event.target.checked)
                resetPrecheck()
              }}
              disabled={submitting}
            />
            <span>Run a reduced quick-test workflow</span>
          </label>

          <PrecheckPanel
            report={precheckReport}
            checking={checkingPrecheck}
            onRun={handleRunPrecheck}
            disabled={submitting || !datasetName || (mode === 'inference' && !modelPath.trim())}
          />

          <div className="panel-actions">
            <button className="button primary" type="submit" disabled={submitting || !precheckReport?.ready}>
              {submitting ? 'Submitting...' : mode === 'inference' ? 'Start evaluation' : 'Start training'}
            </button>
          </div>
        </form>
      )}
    </div>
  )
}