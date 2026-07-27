import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import AnalyzePage from './pages/AnalyzePage'
import BatchPage from './pages/BatchPage'
import FaqPage from './pages/FaqPage'
import HistoryPage from './pages/HistoryPage'

const NAV = [
  { to: '/', label: 'Analyze', end: true },
  { to: '/batch', label: 'Batch' },
  { to: '/history', label: 'History' },
  { to: '/faq', label: 'FAQ' },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden">

        <aside className="w-52 bg-neutral-700 text-white flex flex-col flex-shrink-0">
          <div className="px-6 pt-7 pb-6">
            <p className="text-xl font-black tracking-tight">Dashboard</p>
          </div>
          <nav className="flex flex-col px-3 gap-0.5">
            {NAV.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `px-4 py-2.5 rounded-lg text-base font-semibold transition-all duration-150 ${
                    isActive
                      ? 'bg-neutral-600 text-white'
                      : 'text-neutral-300 hover:bg-neutral-600 hover:text-white'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="flex flex-col flex-1 overflow-hidden bg-neutral-200">
          <main className="flex-1 overflow-y-auto p-8 md:p-10">
            <Routes>
              <Route path="/" element={<AnalyzePage />} />
              <Route path="/batch" element={<BatchPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/faq" element={<FaqPage />} />
            </Routes>
          </main>
        </div>

      </div>
    </BrowserRouter>
  )
}
