const API_BASE_URL = (
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
).replace(/\/+$/, '')

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new Error('The backend is offline or unreachable. Check that it is running and try again.')
  }

  let payload
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    if (response.status === 413) {
      throw new Error('This photo is larger than 10 MB. Choose a smaller image.')
    }
    if (response.status >= 500) {
      throw new Error(payload?.detail || 'The server could not process this photo. Please try again.')
    }
    throw new Error(payload?.detail || `The request failed (${response.status}).`)
  }
  return payload
}

export function getHealth() {
  return request('/api/health')
}

export function getMetrics() {
  return request('/api/metrics')
}

export function getClasses() {
  return request('/api/classes')
}

export function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request('/api/predict', {
    method: 'POST',
    body: formData,
  })
}

export function getAssetUrl(path) {
  return new URL(path, `${API_BASE_URL}/`).toString()
}
