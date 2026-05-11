'use client'

import { useState, useEffect, useRef } from 'react'
import { signOut, useSession } from 'next-auth/react'
import { MessageSquare, Plus, Trash2, Settings, ChevronRight, LogOut, Activity } from 'lucide-react'
import { DebuggerPanel } from '@/components/debugger/DebuggerPanel'
import { ControlsDrawer } from '@/components/controls/ControlsDrawer'
import { createSession, listSessions, getMessages, sendMessage, deleteSession } from '@/lib/api'
import type { Session, Message, RouteDebug } from '@/types'

export function ChatLayout() {
  const { data: authSession } = useSession()
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSession, setActiveSession] = useState<Session | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showDebugger, setShowDebugger] = useState(true)
  const [showControls, setShowControls] = useState(false)
  const [lastDebug, setLastDebug] = useState<RouteDebug | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { loadSessions() }, [])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function loadSessions() {
    const s = await listSessions()
    setSessions(s.filter((x: any) => !x.deleted))
  }

  async function newSession() {
    const s = await createSession()
    setSessions(prev => [s, ...prev])
    setActiveSession(s)
    setMessages([])
    setLastDebug(null)
  }

  async function selectSession(s: Session) {
    setActiveSession(s)
    const msgs = await getMessages(s.id)
    setMessages(msgs)
    const last = [...msgs].reverse().find(m => m.role === 'assistant')
    if (last?.domain) {
      setLastDebug({
        domain: last.domain,
        routing_mode: last.routing_mode || 'single',
        domain_distribution: {},
        u_score: last.u_score || 0,
        confidence: 0,
        latency_ms: last.latency_ms || 0,
        contradictions_detected: 0,
      })
    }
  }

  async function removeSession(s: Session, e: React.MouseEvent) {
    e.stopPropagation()
    await deleteSession(s.id)
    setSessions(prev => prev.filter(x => x.id !== s.id))
    if (activeSession?.id === s.id) { setActiveSession(null); setMessages([]) }
  }

  async function send() {
    if (!input.trim() || loading || !activeSession) return
    const content = input.trim()
    setInput('')
    setLoading(true)
    const tempId = 'tmp-' + Date.now()
    setMessages(prev => [...prev, { id: tempId, session_id: activeSession.id, role: 'user', content, created_at: Date.now() / 1000 }])
    try {
      const r = await sendMessage(activeSession.id, content)
      setMessages(prev => [...prev, {
        id: r.message_id || 'asst-' + Date.now(),
        session_id: activeSession.id,
        role: 'assistant',
        content: r.response,
        created_at: Date.now() / 1000,
        domain: r.domain,
        routing_mode: r.routing_mode,
        u_score: r.u_score,
        latency_ms: r.latency_ms,
      }])
      setLastDebug({
        domain: r.domain,
        routing_mode: r.routing_mode,
        domain_distribution: r.domain_distribution || {},
        u_score: r.u_score,
        confidence: r.confidence || 0,
        latency_ms: r.latency_ms,
        contradictions_detected: r.contradictions_detected || 0,
        specialist_responses: r.specialist_responses,
      })
      loadSessions()
    } catch (err: any) {
      setMessages(prev => [...prev, { id: 'err-' + Date.now(), session_id: activeSession.id, role: 'assistant', content: `Error: ${err?.message || 'Unknown'}`, created_at: Date.now() / 1000 }])
    } finally { setLoading(false) }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* Sidebar */}
      <aside className="w-60 border-r border-[#e4e1da] bg-[#fafaf8] flex flex-col flex-shrink-0">
        <div className="px-4 py-4 border-b border-[#e4e1da] flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-[#4338ca] flex items-center justify-center">
            <span className="text-white text-xs font-bold">A</span>
          </div>
          <span className="font-semibold text-sm">AUA Framework</span>
        </div>
        <div className="px-3 py-3">
          <button onClick={newSession} className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm bg-[#4338ca] text-white hover:bg-[#3730a3] transition-colors">
            <Plus size={14} /> New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 space-y-0.5">
          {sessions.map(s => (
            <button key={s.id} onClick={() => selectSession(s)}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left text-xs group transition-colors
                ${activeSession?.id === s.id ? 'bg-[#eef2ff] text-[#4338ca]' : 'text-[#374151] hover:bg-[#f0ede6]'}`}>
              <span className="flex items-center gap-1.5 min-w-0">
                <MessageSquare size={11} className="flex-shrink-0" />
                <span className="truncate">{s.title}</span>
              </span>
              <button onClick={e => removeSession(s, e)} className="opacity-0 group-hover:opacity-100 hover:text-red-500">
                <Trash2 size={11} />
              </button>
            </button>
          ))}
        </div>
        <div className="border-t border-[#e4e1da] px-3 py-3 space-y-0.5">
          <button onClick={() => setShowControls(true)} className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-[#374151] hover:bg-[#f0ede6]">
            <Settings size={13} /> AUA Controls
          </button>
          <button onClick={() => setShowDebugger(d => !d)} className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors ${showDebugger ? 'bg-[#eef2ff] text-[#4338ca]' : 'text-[#374151] hover:bg-[#f0ede6]'}`}>
            <Activity size={13} /> Framework Debugger
          </button>
          <button onClick={() => signOut()} className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-[#6b7280] hover:bg-[#f0ede6]">
            <LogOut size={13} /> Sign out ({authSession?.user?.name || 'user'})
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0">
        <div className="border-b border-[#e4e1da] px-6 py-3 flex items-center justify-between">
          <span className="text-sm font-medium truncate">{activeSession?.title || 'Select or create a chat'}</span>
          {lastDebug && <span className="text-xs text-[#6b7280] font-mono">{lastDebug.domain} · U={lastDebug.u_score.toFixed(3)} · {lastDebug.latency_ms.toFixed(0)}ms</span>}
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {!activeSession && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#eef2ff] flex items-center justify-center mb-4">
                <MessageSquare size={26} className="text-[#4338ca]" />
              </div>
              <h2 className="text-base font-semibold mb-1">AUA Framework Chat</h2>
              <p className="text-sm text-[#6b7280] max-w-xs mb-4">Multi-specialist AI routing with utility scoring and arbitration.</p>
              <button onClick={newSession} className="px-4 py-2 bg-[#4338ca] text-white text-sm rounded-lg hover:bg-[#3730a3]">Start a new chat</button>
            </div>
          )}
          {messages.map(m => (
            <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${m.role === 'user' ? 'bg-[#4338ca] text-white rounded-br-sm' : 'bg-[#f0ede6] text-[#18181b] rounded-bl-sm'}`}>
                <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                {m.role === 'assistant' && m.domain && (
                  <p className="mt-1 text-[10px] opacity-60 font-mono">{m.domain} · {m.routing_mode} · U={m.u_score?.toFixed(3)} · {m.latency_ms?.toFixed(0)}ms</p>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-[#f0ede6] rounded-2xl rounded-bl-sm px-4 py-3 flex gap-1">
                {[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full bg-[#4338ca] animate-bounce" style={{animationDelay:`${i*0.15}s`}} />)}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <div className="border-t border-[#e4e1da] px-6 py-4">
          <div className="flex gap-3 items-end">
            <textarea value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              placeholder={activeSession ? 'Ask anything… (Enter to send)' : 'Create a chat first'}
              disabled={!activeSession || loading} rows={1}
              className="flex-1 resize-none px-4 py-3 rounded-xl border border-[#e4e1da] bg-white text-sm focus:outline-none focus:ring-2 focus:ring-[#4338ca] disabled:opacity-50 placeholder:text-[#9ca3af]"
              style={{ maxHeight: '120px', overflowY: 'auto' }} />
            <button onClick={send} disabled={!input.trim() || loading || !activeSession}
              className="px-4 py-3 bg-[#4338ca] text-white rounded-xl text-sm hover:bg-[#3730a3] disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
      </main>

      {showDebugger && <DebuggerPanel debug={lastDebug} loading={loading} />}
      {showControls && <ControlsDrawer onClose={() => setShowControls(false)} />}
    </div>
  )
}
