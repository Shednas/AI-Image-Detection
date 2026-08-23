import { useRef, useState } from 'react'
import { analyzeBatch } from '../api/api'
import { useSession } from '../session'
import { MODELS, DEFAULT_MODEL } from '../models'

// visual feedback during async operations
function Spinner() {
  return <span className="inline-block w-4 h-4 border-2 border-cream border-t-transparent rounded-full spin mr-2 align-middle" />
}

export default function BatchPage() {
  const [zip, setZip] = useState(null)
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()
  const { sessionId, setSessionId } = useSession()

  const handleFile = (f) => { if (!f) return; setZip(f); setResult(null); setError(null) }
  // zip is fully read into memory before inference; large batches may be slow
  const handleAnalyze = async () => {
    if (!zip) return
    setLoading(true); setError(null)
    try {
      const { data } = await analyzeBatch(zip, model, sessionId)
      setResult(data)
      if (data.session_id) setSessionId(data.session_id)
    }
    catch (e) { setError(e.response?.data?.detail || 'Could not reach the server. Check that the backend is running and that /api/health reports ok.') }
    finally { setLoading(false) }
  }
  const reset = () => { setZip(null); setResult(null); setError(null) }

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="page-title">AI Image Detection</h1>

      <div className="space-y-4">
        <div className={`upload-outer ${dragging ? 'ring-2 ring-caramel' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
          onClick={() => fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".zip" className="hidden"
            onChange={(e) => handleFile(e.target.files[0])} />
          <div className="upload-inner">
            {zip
              ? <p className="text-base font-bold text-roast">{zip.name}</p>
              : <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-10 h-10 text-cappuccino mx-auto mb-3">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776" />
                  </svg>
                  <p className="text-base font-semibold text-roast">Upload Batch</p>
                  <p className="text-xs text-roast mt-1">Drag a .zip folder or click to select</p>
                  <p className="text-xs text-roast mt-1">Up to 100 images per zip, 10MB per image, 200MB uncompressed in total.</p>
                </>
            }
          </div>
        </div>

        <div className="card p-5">
          <p className="section-label">Select a Model</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {MODELS.map(({ key, label, sub }) => (
              <button key={key} onClick={() => setModel(key)}
                className={`model-btn ${model === key ? 'model-btn-on' : ''}`}>
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sm">{label}</span>
                  {model === key && <span className="text-xs opacity-60 font-mono">ON</span>}
                </div>
                <span className={`text-xs mt-0.5 block ${model === key ? 'text-cream/60' : 'text-roast'}`}>{sub}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-center">
          <button onClick={handleAnalyze} disabled={!zip || loading} className="btn-primary">
            {loading ? <><Spinner />Analyzing...</> : 'Analyze'}
          </button>
        </div>

        {error && <p className="text-sm text-roast text-center py-2">{error}</p>}

        {result?.warning && (
          <div className="rounded-lg border border-cinnamon bg-cinnamon/10 px-4 py-3">
            <p className="text-xs font-bold text-cinnamon">{result.warning}</p>
          </div>
        )}

        {result && (
          <div className="space-y-4 fade-in">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: 'Files in zip', value: result.total },
                { label: 'Analysed', value: result.valid },
                { label: 'AI, of analysed', value: `${result.ai_count} (${result.ai_pct}%)` },
                { label: 'Real, of analysed', value: `${result.real_count} (${result.real_pct}%)` },
              ].map(({ label, value }) => (
                <div key={label} className="card p-4 text-center">
                  <p className="text-xl font-black tabular-nums text-espresso">{value}</p>
                  <p className="text-xs text-roast mt-0.5 font-semibold uppercase tracking-wide">{label}</p>
                </div>
              ))}
            </div>

            <div className="card overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-latte border-b border-cappuccino/40">
                    {['File', 'Verdict', 'P(AI)', 'Model', 'Latency'].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-bold text-espresso uppercase tracking-wide">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-cappuccino/20">
                  {result.rows.map((row, i) => (
                    <tr key={i} className="hover:bg-latte/50 transition-colors">
                      <td className="px-4 py-2.5 font-mono text-xs text-espresso max-w-xs truncate">{row.file_name}</td>
                      <td className="px-4 py-2.5">
                        {row.error
                          ? <span className="text-xs text-roast italic">Error</span>
                          : row.verdict === 'AI_GENERATED'
                            ? <span className="text-xs font-bold text-cinnamon">AI Generated</span>
                            : <span className="text-xs font-bold text-espresso">Authentic</span>
                        }
                      </td>
                      <td className="px-4 py-2.5 text-xs text-roast tabular-nums">
                        {row.p_ai != null ? `${(row.p_ai * 100).toFixed(1)}%` : '-'}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-roast">{row.model_name ?? '-'}</td>
                      <td className="px-4 py-2.5 text-xs text-roast tabular-nums">{row.latency_ms != null ? `${row.latency_ms} ms` : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex justify-center">
              <button onClick={reset} className="btn-secondary">Upload new batch</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
