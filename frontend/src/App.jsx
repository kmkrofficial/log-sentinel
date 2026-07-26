import { Navigate, Route, Routes } from 'react-router-dom'

import AppShell from './components/AppShell.jsx'
import JobStatusPage from './pages/JobStatusPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import RunDetailsPage from './pages/RunDetailsPage.jsx'
import RunModelPage from './pages/RunModelPage.jsx'

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/run" replace />} />
        <Route path="/run" element={<RunModelPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/train" element={<Navigate to="/run?step=training" replace />} />
        <Route path="/inference" element={<Navigate to="/run?step=training&mode=inference" replace />} />
        <Route path="/data-prep" element={<Navigate to="/run?step=preparation" replace />} />
        <Route path="/jobs/:jobId" element={<JobStatusPage />} />
        <Route path="/runs/:runId" element={<RunDetailsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/run" replace />} />
    </Routes>
  )
}

export default App
