import { useEffect, useState } from 'react'
import { getHistory } from '../api/api'

const CATS = [
  { value: 'all', label: 'All verdicts' },
  { value: 'AI_GENERATED', label: 'AI Generated' },
  { value: 'AUTHENTIC', label: 'Authentic' },
]

export default function HistoryPage() {
  const [records, setRecords] = useState([])
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // default params allow callers to pass fresh values before state updates propagate
  const loadHistory = async (s = search, c = category) => {
    setLoading(true)
    setError(null)
    try { const { data } = await getHistory(s, c); setRecords(data) }
    catch (e) { setError(e.response?.data?.detail || 'Failed to load history. Is the backend running?') }
    finally { setLoading(false) }
  }

  useEffect(() => { loadHistory() }, [])

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="page-title">AI Image Detection</h1>

      <div className="flex items-center gap-3 mb-4">
        <span className="text-xl font-black text-espresso whitespace-nowrap">History</span>
        <input
          type="text"
          placeholder="Search by filename"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && loadHistory()}
          className="flex-1 max-w-xs bg-latte border border-cappuccino/60 rounded-md px-4 py-2
                     text-sm text-espresso placeholder-cappuccino
                     focus:outline-none focus:ring-2 focus:ring-caramel/40 focus:border-caramel"
        />
        <div className="ml-auto">
          <select
            value={category}
            onChange={(e) => { setCategory(e.target.value); loadHistory(search, e.target.value) }}
            className="bg-latte border border-cappuccino/60 rounded-md px-4 py-2 text-sm
                       text-espresso focus:outline-none focus:ring-2 focus:ring-caramel/40
                       focus:border-caramel font-medium"
          >
            {CATS.map(({ value, label }) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <p className="text-sm text-roast text-center py-2">{error}</p>}

      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-latte border-b border-cappuccino/40">
              {['Record ID', 'Timestamp', 'File Name', 'Model', 'Verdict'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-xs font-bold text-roast uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-cappuccino/20">
            {loading ? (
              <tr><td colSpan={5} className="px-4 py-12 text-center text-roast text-sm">Loading...</td></tr>
            ) : records.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center">
                  <p className="text-roast text-sm">No records found</p>
                  <p className="text-roast text-xs mt-1">Analyze an image to see history here</p>
                </td>
              </tr>
            ) : records.map((r, i) => (
              <tr key={i} className="hover:bg-latte/60 transition-colors">
                <td className="px-4 py-3 font-mono text-xs text-cappuccino">{r.record_id}</td>
                <td className="px-4 py-3 text-xs text-roast whitespace-nowrap">{new Date(r.timestamp).toLocaleString()}</td>
                <td className="px-4 py-3 text-sm text-espresso truncate max-w-xs">{r.file_name}</td>
                <td className="px-4 py-3 text-xs text-cappuccino">{r.model_name}</td>
                <td className="px-4 py-3">
                  {r.verdict === 'AI_GENERATED'
                    ? <span className="text-xs font-bold text-espresso">AI Generated</span>
                    : <span className="text-xs font-bold text-roast">Authentic</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-roast text-center mt-3">
        Only image metadata is stored (not the actual image) to protect user privacy.
      </p>
    </div>
  )
}
