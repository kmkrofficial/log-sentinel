import { Navigate, Route, Routes } from 'react-router-dom'

import AppShell from './components/AppShell.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import JobStatusPage from './pages/JobStatusPage.jsx'
import RunDetailsPage from './pages/RunDetailsPage.jsx'
import TrainPage from './pages/TrainPage.jsx'

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/train" element={<TrainPage />} />
        <Route path="/jobs/:jobId" element={<JobStatusPage />} />
        <Route path="/runs/:runId" element={<RunDetailsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
