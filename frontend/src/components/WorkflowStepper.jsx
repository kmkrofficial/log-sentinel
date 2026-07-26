const STEP_NUMBERS = {
  setup: '01',
  preparation: '02',
  training: '03',
}

export default function WorkflowStepper({ steps, activeStep, onSelect }) {
  return (
    <ol className="workflow-stepper" aria-label="Run model workflow">
      {steps.map((step) => {
        const isActive = step.id === activeStep
        const isLocked = step.locked && !isActive
        return (
          <li className={`workflow-step is-${step.state}${isActive ? ' is-active' : ''}`} key={step.id}>
            <button
              type="button"
              disabled={isLocked}
              aria-current={isActive ? 'step' : undefined}
              onClick={() => onSelect(step.id)}
            >
              <span className="workflow-step-number">{STEP_NUMBERS[step.id]}</span>
              <span className="workflow-step-copy">
                <strong>{step.title}</strong>
                <small>{step.description}</small>
              </span>
              <span className="workflow-step-state">{step.stateLabel}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}