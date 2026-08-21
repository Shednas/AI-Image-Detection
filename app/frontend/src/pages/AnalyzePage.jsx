import { useRef, useState } from 'react'
import { analyzeImage } from '../api/api'
import { useSession } from '../session'
import { MODELS, DEFAULT_MODEL } from '../models'

const ZONE_STYLES = {
  very_likely_real: { text: 'text-teal-700', bg: 'bg-teal-50 border-teal-200' },
  likely_real: { text: 'text-teal-600', bg: 'bg-teal-50/60 border-teal-100' },
  uncertain: { text: 'text-caramel', bg: 'bg-latte border-cappuccino/60' },
  likely_ai: { text: 'text-orange-700', bg: 'bg-orange-50 border-orange-200' },
  very_likely_ai: { text: 'text-red-800', bg: 'bg-red-50 border-red-200' },
}

// Labels describe where the value sits, nothing more. They previously read as
// evidence of AI origin, which fired on renders, screenshots and compressed
// photographs regardless of the verdict. No model except STM reads these.
const CONTRAST_ZONES = [
  { max: 18, label: 'Very low', text: 'text-roast' },
  { max: 32, label: 'Low', text: 'text-roast' },
  { max: 72, label: 'Typical for photographs', text: 'text-roast' },
  { max: 90, label: 'High', text: 'text-roast' },
  { max: 999, label: 'Very high', text: 'text-roast' },
]

const NOISE_ZONES = [
  { max: 4, label: 'Very low', text: 'text-roast' },
  { max: 8, label: 'Low', text: 'text-roast' },
  { max: 25, label: 'Typical for photographs', text: 'text-roast' },
  { max: 999, label: 'High', text: 'text-roast' },
]

// last zone acts as the fallback for values above all thresholds
function getZone(v, zones) { return zones.find(z => v <= z.max) ?? zones[zones.length - 1] }

// visual feedback during async operations
function Spinner() {
  return <span className="inline-block w-4 h-4 border-2 border-cream border-t-transparent rounded-full spin mr-2 align-middle" />
}

// inline SVG avoids external icon library dependency
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-10 h-10 text-cappuccino mx-auto mb-3">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
    </svg>
  )
}

// slider-style metric display with zone-based color coding
function Gauge({ label, value, displayMax, zones, desc }) {
  const z = getZone(value, zones)
  const pct = Math.min((value / displayMax) * 100, 99)
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-bold text-roast uppercase tracking-widest">{label}</span>
        <span className={`text-2xl font-black tabular-nums ${z.text}`}>{value}</span>
      </div>
      <div className="relative h-2 rounded-full bg-cappuccino/30">
        <div className="absolute left-0 top-0 h-2 rounded-full bg-cappuccino/70 transition-all duration-700"
          style={{ width: `${pct}%` }} />
        <div className="absolute -top-2 w-6 h-6 rounded-full bg-cream border-2 border-espresso shadow-sm transition-all duration-700"
          style={{ left: `calc(${pct}% - 12px)` }} />
      </div>
      <div className="flex justify-between text-xs text-roast font-medium mt-1">
        <span>0</span><span>{displayMax}</span>
      </div>
      <p className={`text-xs font-bold ${z.text}`}>{z.label}</p>
      <p className="text-xs text-roast leading-relaxed">{desc}</p>
    </div>
  )
}

// renders channel distributions as SVG bars to spot AI smoothing
function RGBHistogram({ data }) {
  if (!data) return null
  const max = Math.max(...data.r, ...data.g, ...data.b, 1)
  const W = 260, H = 80, n = data.r.length, bw = W / n
  const smooth = data.smoothness
  const note =
    smooth < 0.8 ? { t: 'Low variation', c: 'text-roast' } :
    smooth < 1.5 ? { t: 'Moderate variation', c: 'text-roast' } :
                   { t: 'High variation', c: 'text-roast' }
  return (
    <div className="space-y-3">
      <svg width={W} height={H} className="block rounded-lg w-full" style={{ maxWidth: W }}>
        {[['r','#c0392b'],['g','#27ae60'],['b','#2980b9']].map(([ch, color]) =>
          data[ch].map((val, i) => (
            <rect key={`${ch}${i}`} x={i*bw} y={H-(val/max)*H}
              width={Math.max(bw-0.4,0.4)} height={(val/max)*H} fill={color} opacity={0.6} />
          ))
        )}
      </svg>
      <div className="flex items-center justify-between text-xs">
        <div className="flex gap-3 text-roast">
          <span><span className="inline-block w-2 h-2 rounded-sm bg-red-700 mr-1 align-middle" />R</span>
          <span><span className="inline-block w-2 h-2 rounded-sm bg-green-700 mr-1 align-middle" />G</span>
          <span><span className="inline-block w-2 h-2 rounded-sm bg-blue-700 mr-1 align-middle" />B</span>
        </div>
        <span className={`font-bold text-xs ${note.c}`}>{note.t}</span>
      </div>
      <div className="bg-cream rounded-lg p-3 border border-cappuccino/40 text-xs text-roast">
        <p>Counts of pixels per brightness level for each channel. The variation figure is the standard deviation of those counts divided by their mean, so it is higher when the distribution is uneven. Clipping at 0 or 255 indicates clipped highlights or shadows.</p>
      </div>
    </div>
  )
}

// horizontal bar chart showing relative feature group contributions
function FeatureChart({ data }) {
  if (!data) return null
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = Math.max(...entries.map(([, v]) => v), 1)
  const shades = ['bg-espresso', 'bg-roast', 'bg-caramel', 'bg-cappuccino', 'bg-cappuccino/50']
  return (
    <div className="space-y-3">
      {entries.map(([name, value], i) => (
        <div key={name} className="flex items-center gap-3">
          <span className="text-xs text-roast w-44 flex-shrink-0">{name}</span>
          <div className="flex-1 bg-cappuccino/30 rounded-full h-3">
            <div className={`${shades[i] ?? 'bg-roast'} h-3 rounded-full transition-all duration-700`}
              style={{ width: `${(value / max) * 100}%` }} />
          </div>
          <span className="text-xs text-roast w-8 text-right tabular-nums font-bold">{value}%</span>
        </div>
      ))}
      <p className="text-xs text-roast">Darker bars had stronger influence on the verdict.</p>
    </div>
  )
}

// wraps base64 visualization images from backend
function VizImage({ b64, alt }) {
  return (
    <div className="flex justify-center bg-espresso/5 rounded-lg p-2">
      <img src={`data:image/png;base64,${b64}`} alt={alt}
        className="max-h-52 rounded-lg object-contain" />
    </div>
  )
}

// every visualisation is implemented, so a null payload means generation threw
// for this image. Saying "coming soon" hid real failures behind a roadmap message.
function VizFailed() {
  return (
    <div className="rounded-lg bg-cappuccino/20 px-3 py-4 text-center">
      <p className="text-xs font-semibold text-roast">Could not be generated for this image</p>
      <p className="text-xs text-cappuccino mt-0.5">The verdict above is unaffected. Details are in the backend log.</p>
    </div>
  )
}

export default function AnalyzePage() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [showDetails, setShowDetails] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()
  const { sessionId, setSessionId } = useSession()

  // reset result state so stale data doesn't persist alongside the new file
  const handleFile = (f) => {
    if (!f) return
    setImage(f); setPreview(URL.createObjectURL(f))
    setResult(null); setError(null); setShowDetails(false)
  }
  const handleAnalyze = async () => {
    if (!image) return
    setLoading(true); setError(null)
    try {
      const { data } = await analyzeImage(image, model, sessionId)
      setResult(data)
      if (data.session_id) setSessionId(data.session_id)
    }
    catch (e) { setError(e.response?.data?.detail || 'Could not reach the server. Check that the backend is running and that /api/health reports ok.') }
    finally { setLoading(false) }
  }
  const reset = () => { setImage(null); setPreview(null); setResult(null); setError(null); setShowDetails(false) }

  const viz = result?.visualizations
  const isModelAI = result?.verdict === 'AI_GENERATED'
  const isSTM = model === 'stm'

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="page-title">AI Image Detection</h1>

      <div className="space-y-4">

        <div className={`upload-outer ${dragging ? 'ring-2 ring-caramel' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
          onClick={() => fileRef.current.click()}>
          <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp"
            className="hidden" onChange={(e) => handleFile(e.target.files[0])} />
          <div className="upload-inner">
            {preview
              ? <img src={preview} alt="Preview" className="max-h-48 rounded-lg object-contain" />
              : <><UploadIcon /><p className="text-base font-semibold text-roast">Upload an Image</p>
                  <p className="text-xs text-roast mt-1">PNG, JPEG or WEBP, up to 10MB. Trained on photographs, so results on screenshots, renders and scans are unreliable.</p></>
            }
          </div>
          {image && <p className="text-center text-xs text-roast mt-2">{image.name}</p>}
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
          <button onClick={handleAnalyze} disabled={!image || loading} className="btn-primary">
            {loading ? <><Spinner />Analyzing...</> : 'Analyze'}
          </button>
        </div>

        {error && <p className="text-sm text-roast text-center py-2">{error}</p>}

        {result?.warning && (
          <div className="rounded-lg border border-orange-200 bg-orange-50 px-4 py-3">
            <p className="text-xs font-bold text-orange-800">{result.warning}</p>
          </div>
        )}

        {result && (() => {
          const { zone, ai_pct, confidence_pct, model_name, latency_ms, model_explanation, verdict_explanation } = result
          const zoneStyle = ZONE_STYLES[zone?.key] ?? ZONE_STYLES.uncertain
          const pct = Math.min(Math.max(ai_pct, 1), 99)
          const contrast = viz?.generic_metrics?.contrast
          const noise = viz?.generic_metrics?.noise
          // low contrast/noise on an AUTHENTIC verdict suggests compressed or indoor images, not a conflict
          const conflict = !isModelAI && ((contrast != null && contrast < 32) || (noise != null && noise < 8))

          return (
            <div className="space-y-4 fade-in">

              <div className={`rounded-lg border p-5 ${zoneStyle.bg}`}>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div>
                    <p className={`text-2xl font-black ${zoneStyle.text}`}>{zone?.label}</p>
                    <p className="text-sm text-roast mt-1">
                      <span className="font-bold">{confidence_pct}%</span> confidence
                      <span className="text-roast mx-2">/</span>
                      P(AI) = {ai_pct}%
                    </p>
                  </div>
                  <p className="text-xs text-cappuccino text-right">{model_name} / {latency_ms} ms</p>
                </div>
                <div className="mt-4 relative">
                  <div className="h-4 rounded-full"
                    style={{ background: 'linear-gradient(to right,#0d9488,#84cc16,#eab308,#f97316,#dc2626)' }} />
                  <div className="absolute top-0 w-5 h-5 rounded-full bg-cream border-2 border-espresso shadow-md -translate-y-0.5 transition-all duration-700"
                    style={{ left: `calc(${pct}% - 10px)` }} />
                </div>
                <div className="flex justify-between text-xs font-bold mt-2">
                  <span className="text-roast">Real (0)</span>
                  <span className="text-caramel">Uncertain (0.5)</span>
                  <span className="text-espresso">AI (1)</span>
                </div>
              </div>

              <div className="text-sm text-roast leading-relaxed space-y-1">
                <p>{model_explanation}</p>
                <p className="font-semibold text-espresso">{verdict_explanation}</p>
              </div>

              <div className="flex flex-wrap gap-3">
                <button onClick={() => setShowDetails(!showDetails)} className="btn-secondary">
                  {showDetails ? 'Hide analysis details' : 'Show analysis details'}
                </button>
                <button onClick={reset} className="btn-secondary">
                  Analyze another image
                </button>
              </div>

              {showDetails && (
                <div className="space-y-4 fade-in">

                  {viz?.generic_metrics && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="card p-5 space-y-5">
                        <div>
                          <p className="section-label">{isSTM ? 'Image properties, partly used by STM' : 'Image properties'}</p>
                          <p className="text-xs text-roast leading-relaxed">
                            {isSTM
                              ? 'Computed from the image. STM uses noise residuals and colour statistics among its 1,822 inputs, so for this model they are genuine evidence. Contrast is not one of its inputs.'
                              : 'Computed from the image and shown for context. This model does not read them, so they neither support nor contradict the verdict above.'}
                          </p>
                        </div>
                        <Gauge label="Contrast" value={contrast} displayMax={100} zones={CONTRAST_ZONES}
                          desc="Standard deviation of pixel brightness, scaled by 100. Photographs commonly fall between 30 and 70. Uniform subjects, low light and compression all push it down." />
                        <div className="border-t border-cappuccino/40" />
                        <Gauge label="Sensor Noise" value={noise} displayMax={35} zones={NOISE_ZONES}
                          desc="Standard deviation of the high-frequency residual after subtracting a blurred copy, scaled by 100. Photographs commonly fall between 8 and 25. Compression and resizing reduce it." />
                      </div>

                      {viz?.rgb_distribution && (
                        <div className="card p-5">
                          <p className="section-label">RGB Distribution</p>
                          <p className="text-xs text-roast leading-relaxed mb-3">
                            {isSTM
                              ? 'Colour statistics are among the features STM classifies on.'
                              : 'Computed from the image, not read by this model.'}
                          </p>
                          <RGBHistogram data={viz.rgb_distribution} />
                        </div>
                      )}
                    </div>
                  )}

                  {conflict && (
                    <div className="conflict-note">
                      <p className="font-bold text-espresso mb-1">Low contrast or noise does not mean AI</p>
                      <p>Both are commonly low in JPEG-compressed images, indoor or overcast photographs, uniform subjects such as walls, sky and screens, and images processed with phone HDR.</p>
                      <p className="mt-1.5">These two numbers are computed from the image and are shown for context. Except for STM, which uses noise and colour statistics among its inputs, no model on this page reads them, so they neither support nor contradict the verdict.</p>
                    </div>
                  )}

                  {viz?.heatmap && (
                    <div className="card p-5">
                      <p className="section-label">What this model analysed: Grad-CAM Heatmap</p>
                      {viz.heatmap.data ? <VizImage b64={viz.heatmap.data} alt="Grad-CAM heatmap" /> : <VizFailed />}
                      <p className="text-xs text-roast mt-3 leading-relaxed">{viz.heatmap.description}</p>
                    </div>
                  )}

                  {viz?.spectrogram && (
                    <div className="card p-5">
                      <p className="section-label">What this model analysed: Frequency Spectrogram</p>
                      {viz.spectrogram.data ? <VizImage b64={viz.spectrogram.data} alt="Frequency spectrogram" /> : <VizFailed />}
                      <p className="text-xs text-roast mt-3 leading-relaxed">{viz.spectrogram.description}</p>
                    </div>
                  )}

                  {viz?.feature_importance && (
                    <div className="card p-5">
                      <p className="section-label">What this model analysed: Feature Contributions</p>
                      {viz.feature_importance.data ? <FeatureChart data={viz.feature_importance.data} /> : <VizFailed />}
                      <p className="text-xs text-roast mt-3 leading-relaxed">{viz.feature_importance.description}</p>
                    </div>
                  )}

                </div>
              )}
            </div>
          )
        })()}
      </div>
    </div>
  )
}
