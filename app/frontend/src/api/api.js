import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000/api' })

// session_id is omitted on the first request of a visit; the backend creates one
// and returns it, and every later request reuses it so history groups by visit
export const analyzeImage = (file, modelName, sessionId) => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_name', modelName)
  if (sessionId) fd.append('session_id', sessionId)
  return api.post('/analyze', fd)
}

// sends zip + model for batch analysis
export const analyzeBatch = (file, modelName, sessionId) => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_name', modelName)
  if (sessionId) fd.append('session_id', sessionId)
  return api.post('/batch', fd)
}

// omit undefined params so the backend doesn't filter on empty strings
export const getHistory = (search = '', category = 'all') =>
  api.get('/history', {
    params: {
      search: search || undefined,
      category: category !== 'all' ? category : undefined,
    },
  })
