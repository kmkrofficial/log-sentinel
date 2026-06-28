import { NavLink, Outlet } from 'react-router-dom'

export default function AppShell() {
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
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            Dashboard
          </NavLink>
          <NavLink to="/train" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            Start Run
          </NavLink>
        </nav>
      </header>

      <main className="shell-content">
        <Outlet />
      </main>
    </div>
  )
}