const ROUTER = process.env.AUA_ROUTER_URL || 'http://localhost:8000'

export async function createSession(title = ''): Promise<any> {
  const r = await fetch(`${ROUTER}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!r.ok) throw new Error('Failed to create session')
  return r.json()
}

export async function listSessions(): Promise<any[]> {
  const r = await fetch(`${ROUTER}/sessions`)
  if (!r.ok) return []
  const d = await r.json()
  return d.sessions || []
}

export async function getMessages(sessionId: string): Promise<any[]> {
  const r = await fetch(`${ROUTER}/sessions/${sessionId}/messages`)
  if (!r.ok) return []
  const d = await r.json()
  return d.messages || []
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${ROUTER}/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function sendMessage(sessionId: string, content: string): Promise<any> {
  const r = await fetch(`${ROUTER}/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!r.ok) throw new Error('Failed to send message')
  return r.json()
}

export async function streamMessage(
  sessionId: string,
  content: string,
  onEvent: (e: any) => void
): Promise<void> {
  // Fall back to buffered if stream endpoint not available
  const result = await sendMessage(sessionId, content)
  onEvent({ type: 'done', ...result })
}

export async function getStatus(): Promise<any> {
  try {
    const r = await fetch(`${ROUTER}/status`)
    return r.ok ? r.json() : null
  } catch { return null }
}

export async function getConfig(): Promise<any> {
  try {
    const r = await fetch(`${ROUTER}/config`)
    return r.ok ? r.json() : null
  } catch { return null }
}
