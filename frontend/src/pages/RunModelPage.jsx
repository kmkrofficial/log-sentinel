import { useContext, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import ConfigurationSetupStep from '../components/ConfigurationSetupStep.jsx'
import DatasetPreparationStep from '../components/DatasetPreparationStep.jsx'
import ModelExecutionStep from '../components/ModelExecutionStep.jsx'
import WorkflowStepper from '../components/WorkflowStepper.jsx'
import { JobContext } from '../context/JobContext.jsx'
import {
  fetchDataPrepStatus,
  fetchSetupStatus,
  startDataPrep,
  startSetupProvision,
} from '../services/api.js'

const STEP_IDS = new Set(['setup', 'preparation', 'training'])

function toErrorMessage(error, fallback) {
  const detail = error.response?.data?.detail
  return typeof detail === 'string' ? detail : detail?.message ?? error.message ?? fallback
}

export default function RunModelPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const {
    activeSetupJob,
    setActiveSetupJob,
    activePrepJob,
    setActivePrepJob,
    activeTrainingJob,
    activeInferenceJob,
  } = useContext(JobContext)
  const [setupStatus, setSetupStatus] = useState(null)
  const [preparationStatus, setPreparationStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const requestedStep = searchParams.get('step')
  const mode = searchParams.get('mode') === 'inference' ? 'inference' : 'training'

  const refresh = async () => {
    try {
      setLoading(true)
      const [nextSetupStatus, nextPreparationStatus] = await Promise.all([fetchSetupStatus(), fetchDataPrepStatus()])
      setSetupStatus(nextSetupStatus)
      setPreparationStatus(nextPreparationStatus)
      setError('')
    } catch (loadError) {
      setError(toErrorMessage(loadError, 'Unable to load configuration and preparation status.'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const runtimeReady = !(setupStatus?.runtime_checks ?? []).some((check) => check.status === 'failed')
  const configurationReady = Boolean(
    setupStatus
      && runtimeReady
      && setupStatus.datasets.every((dataset) => dataset.raw.ready)
      && setupStatus.models.every((model) => model.ready),
  )
  const preparedDatasetCount = (setupStatus?.datasets ?? []).filter(
    (dataset) => dataset.prepared.train.ready && dataset.prepared.test.ready,
  ).length
  const preparationReady = preparedDatasetCount > 0
  const requestedStepId = STEP_IDS.has(requestedStep) ? requestedStep : 'setup'
  const activeStep = !configurationReady && requestedStepId !== 'setup'
    ? 'setup'
    : !preparationReady && requestedStepId === 'training'
      ? 'preparation'
      : requestedStepId

  const steps = [
    {
      id: 'setup',
      title: 'Configuration setup',
      description: 'Verify and provision assets',
      locked: false,
      state: activeSetupJob ? 'running' : configurationReady ? 'complete' : error ? 'blocked' : 'ready',
      stateLabel: activeSetupJob ? 'Running' : configurationReady ? 'Complete' : error ? 'Blocked' : 'Required',
    },
    {
      id: 'preparation',
      title: 'Data preparation',
      description: 'Build supervised splits',
      locked: !configurationReady,
      state: !configurationReady ? 'locked' : activePrepJob ? 'running' : preparationReady ? 'complete' : 'ready',
      stateLabel: !configurationReady ? 'Locked' : activePrepJob ? 'Running' : preparationReady ? 'Complete' : 'Required',
    },
    {
      id: 'training',
      title: 'Model training',
      description: 'Pre-check and execute',
      locked: !configurationReady || !preparationReady,
      state: !configurationReady || !preparationReady ? 'locked' : activeTrainingJob || activeInferenceJob ? 'running' : 'ready',
      stateLabel: !configurationReady || !preparationReady ? 'Locked' : activeTrainingJob || activeInferenceJob ? 'Running' : 'Ready',
    },
  ]

  const setStep = (step, nextMode = mode) => {
    setSearchParams({ step, ...(step === 'training' ? { mode: nextMode } : {}) })
  }

  const handleProvision = async (payload) => {
    try {
      setError('')
      const response = await startSetupProvision(payload)
      setActiveSetupJob(response.job_id)
    } catch (provisionError) {
      setError(toErrorMessage(provisionError, 'Unable to start setup provisioning.'))
    }
  }

  const handleStartPreparation = async (payload) => {
    try {
      setError('')
      const response = await startDataPrep(payload)
      setActivePrepJob(response.job_id)
    } catch (preparationError) {
      setError(toErrorMessage(preparationError, 'Unable to start data preparation.'))
    }
  }

  return (
    <div className="run-model-page">
      <section className="run-model-header">
        <div className="eyebrow">Run model</div>
        <h2>Provision, prepare, and train from one controlled workflow.</h2>
        <p>Each step exposes its real readiness state. Long-running setup, preparation, training, and evaluation jobs remain available while you navigate.</p>
      </section>

      <WorkflowStepper steps={steps} activeStep={activeStep} onSelect={setStep} />

      {activeStep === 'setup' ? (
        <ConfigurationSetupStep
          status={setupStatus}
          loading={loading}
          error={error}
          activeJobId={activeSetupJob}
          onClearJob={() => { setActiveSetupJob(null); refresh() }}
          onRefresh={refresh}
          onProvision={handleProvision}
        />
      ) : null}

      {activeStep === 'preparation' ? (
        <DatasetPreparationStep
          setupStatus={setupStatus}
          preparationStatus={preparationStatus}
          loading={loading}
          error={error}
          activeJobId={activePrepJob}
          onClearJob={() => { setActivePrepJob(null); refresh() }}
          onRefresh={refresh}
          onStart={handleStartPreparation}
        />
      ) : null}

      {activeStep === 'training' ? (
        <ModelExecutionStep
          setupStatus={setupStatus}
          mode={mode}
          onModeChange={(nextMode) => setStep('training', nextMode)}
          onRefresh={refresh}
        />
      ) : null}
    </div>
  )
}