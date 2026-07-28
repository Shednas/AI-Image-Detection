import { createContext, useContext, useState } from 'react'

const SessionContext = createContext(null)
const STORAGE_KEY = 'ai-detection-session-id'

// one id for the whole browser session, held above the router so Analyze and
// Batch share it. the backend groups history rows by this, so a per-page id
// would put every request in its own session again.
// mirrored into sessionStorage so a refresh mid-demo does not fragment the
// grouping; it still clears when the tab closes, which is the boundary we want
export function SessionProvider({ children }) {
  const [sessionId, setSessionIdState] = useState(() => sessionStorage.getItem(STORAGE_KEY))

  const setSessionId = (id) => {
    setSessionIdState(id)
    if (id) sessionStorage.setItem(STORAGE_KEY, id)
    else sessionStorage.removeItem(STORAGE_KEY)
  }

  return (
    <SessionContext.Provider value={{ sessionId, setSessionId }}>
      {children}
    </SessionContext.Provider>
  )
}

export function useSession() {
  return useContext(SessionContext)
}
