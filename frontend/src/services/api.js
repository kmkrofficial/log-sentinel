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

export async function fetchSetupStatus() {
  return unwrap(await apiClient.get('/setup/status'))
}

export async function startSetupProvision(payload) {
  return unwrap(await apiClient.post('/setup/provision', payload))
}

export async function fetchRunHistory() {
  return unwrap(await apiClient.get('/runs'))
}

export async function fetchRunDetails(runId) {
  return unwrap(await apiClient.get(`/runs/${runId}`))
}

export async function fetchModels() {
  return unwrap(await apiClient.get('/models'))
}

export async function submitTrainingJob(payload) {
  return unwrap(await apiClient.post('/train', payload))
}

export async function startInference(payload) {
  return unwrap(await apiClient.post('/inference', payload))
}

export const submitInferenceJob = startInference

export async function startDataPrep(payload) {
  return unwrap(await apiClient.post('/data-prep', payload))
}

export async function fetchDataPrepStatus() {
  return unwrap(await apiClient.get('/data-prep/status'))
}

export async function runPrecheck(payload) {
  return unwrap(await apiClient.post('/pre-check', payload))
}

export async function fetchJobStatus(jobId) {
  return unwrap(await apiClient.get(`/status/${jobId}`))
}

export async function fetchActiveJobs() {
  return unwrap(await apiClient.get('/jobs/active'))
}