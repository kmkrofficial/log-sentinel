function toTone(status) {
  const normalized = String(status ?? 'pending').toLowerCase()

  if (normalized.includes('fail')) {
    return 'is-failed'
  }
  if (normalized.includes('complete')) {
    return 'is-completed'
  }
  if (normalized.includes('pending')) {
    return 'is-pending'
  }
  return 'is-running'
}

function toLabel(status) {
  const normalized = String(status ?? 'pending').replaceAll('_', ' ')
  return normalized
}

export default function StatusBadge({ status }) {
  return <span className={`status-badge ${toTone(status)}`}>{toLabel(status)}</span>
}