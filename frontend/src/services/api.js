import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

const unwrap = (response) => response.data

export async function fetchDatasets() {
  return unwrap(await apiClient.get('/datasets'))
}

export async function fetchRunHistory() {
  return unwrap(await apiClient.get('/runs'))
}

export async function fetchRunDetails(runId) {
  return unwrap(await apiClient.get(`/runs/${runId}`))
}

export async function submitTrainingJob(payload) {
  return unwrap(await apiClient.post('/train', payload))
}

export async function submitInferenceJob(payload) {
  return unwrap(await apiClient.post('/inference', payload))
}

export async function fetchJobStatus(jobId) {
  return unwrap(await apiClient.get(`/status/${jobId}`))
}