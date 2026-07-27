import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000/api' })

// sends image + selected model to the backend for single-image inference
export const analyzeImage = (file, modelName) => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_name', modelName)
  return api.post('/analyze', fd)
}

// sends zip + model for batch analysis
export const analyzeBatch = (file, modelName) => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_name', modelName)
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
