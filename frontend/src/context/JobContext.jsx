import { createContext, useContext, useEffect, useState } from 'react'

import { fetchActiveJobs } from '../services/api.js'

export const JobContext = createContext(null)

const STORAGE_KEYS = {
  setup: 'logsentinel.activeSetupJob',
  training: 'logsentinel.activeTrainingJob',
  inference: 'logsentinel.activeInferenceJob',
  prep: 'logsentinel.activePrepJob',
}

const ACTIVE_JOB_REFRESH_MS = 5000

function getStoredJobId(storageKey) {
  if (typeof window === 'undefined') {
    return null
  }

  const value = window.localStorage.getItem(storageKey)
  return typeof value === 'string' && value.trim() ? value : null
}

function usePersistedJobId(storageKey) {
  const [jobId, setJobId] = useState(() => getStoredJobId(storageKey))

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    if (jobId === null) {
      window.localStorage.removeItem(storageKey)
      return
    }

    window.localStorage.setItem(storageKey, jobId)
  }, [jobId, storageKey])

  return [jobId, setJobId]
}

export function JobProvider({ children }) {
  const [activeSetupJob, setActiveSetupJob] = usePersistedJobId(STORAGE_KEYS.setup)
  const [activeTrainingJob, setActiveTrainingJob] = usePersistedJobId(STORAGE_KEYS.training)
  const [activeInferenceJob, setActiveInferenceJob] = usePersistedJobId(STORAGE_KEYS.inference)
  const [activePrepJob, setActivePrepJob] = usePersistedJobId(STORAGE_KEYS.prep)

  useEffect(() => {
    let cancelled = false

    const recoverActiveJobs = async () => {
      try {
        const payload = await fetchActiveJobs()
        if (cancelled) {
          return
        }

        const jobsByType = new Map(payload.jobs.map((job) => [job.job_type, job.job_id]))
        const setupJobId = jobsByType.get('setup')
        const trainingJobId = jobsByType.get('training')
        const inferenceJobId = jobsByType.get('inference')
        const prepJobId = jobsByType.get('prep')

        if (setupJobId) {
          setActiveSetupJob((currentJobId) => currentJobId ?? setupJobId)
        }
        if (trainingJobId) {
          setActiveTrainingJob((currentJobId) => currentJobId ?? trainingJobId)
        }
        if (inferenceJobId) {
          setActiveInferenceJob((currentJobId) => currentJobId ?? inferenceJobId)
        }
        if (prepJobId) {
          setActivePrepJob((currentJobId) => currentJobId ?? prepJobId)
        }
      } catch {
        // The existing persisted job IDs remain usable when discovery is unavailable.
      }
    }

    recoverActiveJobs()
    const intervalId = window.setInterval(recoverActiveJobs, ACTIVE_JOB_REFRESH_MS)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [setActiveInferenceJob, setActivePrepJob, setActiveSetupJob, setActiveTrainingJob])

  return (
    <JobContext.Provider
      value={{
        activeSetupJob,
        setActiveSetupJob,
        activeTrainingJob,
        setActiveTrainingJob,
        activeInferenceJob,
        setActiveInferenceJob,
        activePrepJob,
        setActivePrepJob,
      }}
    >
      {children}
    </JobContext.Provider>
  )
}

export function useJobContext() {
  const context = useContext(JobContext)

  if (context === null) {
    throw new Error('useJobContext must be used within a JobProvider.')
  }

  return context
}