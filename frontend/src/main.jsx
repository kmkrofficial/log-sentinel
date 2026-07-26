import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import { JobProvider } from './context/JobContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <JobProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </JobProvider>
  </StrictMode>,
)
