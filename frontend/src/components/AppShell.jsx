import { useContext } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { JobContext } from '../context/JobContext.jsx'

export default function AppShell() {
  const { activeSetupJob, activePrepJob, activeTrainingJob, activeInferenceJob } = useContext(JobContext)
  const hasActiveWorkflow = Boolean(activeSetupJob || activePrepJob || activeTrainingJob || activeInferenceJob)

  return (
    <div className="app-shell">
      <div className="shell-backdrop" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">LS</div>
          <div className="brand-copy">
            <h1>LogSentinel Control Plane</h1>
            <p>FastAPI-backed training, evaluation, and forensic metrics.</p>
          </div>
        </div>

        <nav className="nav-links" aria-label="Primary">
          <NavLink to="/run" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <span className="nav-link-content">
              Run Model
              {hasActiveWorkflow ? <span className="nav-job-indicator" role="status" aria-label="A model workflow job is active" /> : null}
            </span>
          </NavLink>
          <NavLink to="/results" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            Results
          </NavLink>
        </nav>
      </header>

      <main className="shell-content">
        <Outlet />
      </main>
    </div>
  )
}