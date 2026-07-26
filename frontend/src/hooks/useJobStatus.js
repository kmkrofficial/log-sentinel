import { useEffect, useState } from 'react'

import { fetchJobStatus } from '../services/api.js'

const TERMINAL_STATUSES = new Set(['completed', 'failed'])

function normalizeStatus(status) {
  return String(status ?? 'pending').toLowerCase()
}

export default function useJobStatus(jobId) {
  const [job, setJob] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      setLoading(false)
      setError('')
      return undefined
    }

    let cancelled = false
    let intervalId = null

    const poll = async () => {
      try {
        const payload = await fetchJobStatus(jobId)
        if (cancelled) {
          return
        }

        const nextJob = {
          ...payload,
          normalizedStatus: normalizeStatus(payload.status),
        }

        setJob(nextJob)
        setError('')
        setLoading(false)

        if (TERMINAL_STATUSES.has(nextJob.normalizedStatus) && intervalId) {
          window.clearInterval(intervalId)
          intervalId = null
        }
      } catch (pollError) {
        if (cancelled) {
          return
        }

        const responseStatus = pollError.response?.status
        const message = responseStatus === 404
          ? 'This job is no longer available. It may have been interrupted when the backend restarted.'
          : (pollError.response?.data?.detail ?? pollError.message ?? 'Unable to load job status.')

        setError(message)
        setLoading(false)

        if (responseStatus === 404 && intervalId) {
          window.clearInterval(intervalId)
          intervalId = null
        }
      }
    }

    poll()
    intervalId = window.setInterval(poll, 3000)

    return () => {
      cancelled = true
      if (intervalId) {
        window.clearInterval(intervalId)
      }
    }
  }, [jobId])

  return { job, loading, error }
}