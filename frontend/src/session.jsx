import { createContext, useContext, useState } from 'react'

const SessionContext = createContext(null)

// one id for the whole browser session, held above the router so Analyze and
// Batch share it. the backend groups history rows by this, so a per-page id
// would put every request in its own session again
export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(null)
  return (
    <SessionContext.Provider value={{ sessionId, setSessionId }}>
      {children}
    </SessionContext.Provider>
  )
}

export function useSession() {
  return useContext(SessionContext)
}
