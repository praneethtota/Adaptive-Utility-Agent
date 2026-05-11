'use client'

import { useState, useEffect, useRef } from 'react'
import { signOut, useSession } from 'next-auth/react'
import { Plus, Trash2, Settings, Send, Activity, MessageSquare, LogOut } from 'lucide-react'
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
      setMessages(prev => [...prev, {
        id: 'err-' + Date.now(),
        session_id: activeSession.id,
        role: 'assistant',
        content: `Error: ${err?.message || 'Unknown'}`,
        created_at: Date.now() / 1000,
      }])
    } finally { setLoading(false) }
  }

  return (
    /* Outer wrapper — warm page background shows as "gutter" between panels */
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)', gap: '0', padding: '0' }}>

      {/* ── SIDEBAR PANEL — warm nav, same as HTML page nav ── */}
      <aside style={{
        width: '220px',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        background: '#fafaf8',
        borderRight: '1px solid var(--line)',
        boxShadow: '2px 0 8px rgba(24,24,27,.05)',
        zIndex: 2,
      }}>
        {/* Wordmark */}
        <div style={{
          padding: '1rem 1.1rem .9rem',
          borderBottom: '2px solid var(--line)',
          display: 'flex', alignItems: 'center', gap: '.6rem',
        }}>
          <div style={{
            width: '26px', height: '26px', borderRadius: '6px',
            background: 'var(--accent)', flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ color: '#fff', fontSize: '.68rem', fontWeight: 700, fontFamily: '"JetBrains Mono",monospace' }}>A</span>
          </div>
          <span style={{ fontFamily: '"DM Serif Display",Georgia,serif', fontSize: '1rem', color: 'var(--ink)', lineHeight: 1.2 }}>
            AUA Framework
          </span>
        </div>

        {/* New chat */}
        <div style={{ padding: '.7rem .9rem' }}>
          <button onClick={newSession} style={{
            width: '100%', display: 'flex', alignItems: 'center', gap: '.45rem',
            padding: '.35rem .8rem', borderRadius: '999px',
            background: 'var(--tag-bg)', color: 'var(--accent)',
            border: '1px solid #c7d2fe',
            fontSize: '.72rem', fontWeight: 600,
            fontFamily: '"JetBrains Mono",monospace',
            letterSpacing: '.05em', textTransform: 'uppercase', cursor: 'pointer',
            transition: 'background .15s',
          }}
            onMouseEnter={e => (e.currentTarget.style.background = '#dde4ff')}
            onMouseLeave={e => (e.currentTarget.style.background = 'var(--tag-bg)')}
          >
            <Plus size={11} /> New Chat
          </button>
        </div>

        {/* Session list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 .65rem' }}>
          {sessions.length === 0 && (
            <p style={{ fontSize: '.76rem', color: 'var(--muted)', padding: '.4rem .4rem', fontStyle: 'italic' }}>No chats yet</p>
          )}
          {sessions.map(s => (
            <button key={s.id} onClick={() => selectSession(s)} style={{
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '.38rem .55rem', borderRadius: '6px', border: 'none',
              background: activeSession?.id === s.id ? 'var(--tag-bg)' : 'transparent',
              color: activeSession?.id === s.id ? 'var(--accent)' : '#374151',
              fontSize: '.8rem', textAlign: 'left', cursor: 'pointer', marginBottom: '1px',
              transition: 'background .1s',
            }}
              onMouseEnter={e => { if (activeSession?.id !== s.id) e.currentTarget.style.background = 'var(--soft)' }}
              onMouseLeave={e => { if (activeSession?.id !== s.id) e.currentTarget.style.background = 'transparent' }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem', minWidth: 0 }}>
                <MessageSquare size={10} style={{ flexShrink: 0, opacity: .5 }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title}</span>
              </span>
              <span onClick={e => removeSession(s, e)} style={{
                opacity: 0, flexShrink: 0, padding: '2px', color: 'var(--accent3)', cursor: 'pointer',
              }}
                onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '0')}
              >
                <Trash2 size={10} />
              </span>
            </button>
          ))}
        </div>

        {/* Bottom actions */}
        <div style={{ borderTop: '1px solid var(--line)', padding: '.65rem .8rem .85rem' }}>
          {[
            { icon: <Settings size={11} />, label: 'AUA Controls', onClick: () => setShowControls(true), active: false },
            { icon: <Activity size={11} />, label: 'Framework Debugger', onClick: () => setShowDebugger(d => !d), active: showDebugger },
            { icon: <LogOut size={11} />, label: `Sign out${authSession?.user?.name ? ` (${authSession.user.name})` : ''}`, onClick: () => signOut(), active: false, muted: true },
          ].map(({ icon, label, onClick, active, muted }) => (
            <button key={label} onClick={onClick} style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: '.45rem',
              padding: '.32rem .55rem', borderRadius: '6px', border: 'none',
              background: active ? 'var(--tag-bg)' : 'transparent',
              color: active ? 'var(--accent)' : muted ? 'var(--muted)' : '#374151',
              fontSize: '.8rem', fontWeight: active ? 600 : 400,
              textAlign: 'left', cursor: 'pointer', marginBottom: '1px', transition: 'background .1s',
            }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--soft)' }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}
            >
              {icon} {label}
            </button>
          ))}
        </div>
      </aside>

      {/* ── CHAT PANEL — pure white, primary content ── */}
      <main style={{
        flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0,
        background: '#ffffff',
        borderRight: showDebugger ? '1px solid var(--line)' : 'none',
        boxShadow: showDebugger ? '2px 0 8px rgba(24,24,27,.04)' : 'none',
        zIndex: 1,
      }}>
        {/* Top bar */}
        <div style={{
          padding: '.7rem 1.5rem',
          borderBottom: '2px solid var(--line)',
          background: '#ffffff',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
          flexShrink: 0,
        }}>
          <span style={{
            fontFamily: '"DM Serif Display",Georgia,serif',
            fontSize: '1rem', color: activeSession ? 'var(--ink)' : 'var(--muted)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {activeSession?.title || 'Select or create a chat'}
          </span>
          {lastDebug && (
            <span style={{
              fontFamily: '"JetBrains Mono",monospace', fontSize: '.68rem', color: 'var(--muted)',
              flexShrink: 0, background: 'var(--soft)', padding: '.2rem .6rem',
              borderRadius: '999px', border: '1px solid var(--line)',
            }}>
              {lastDebug.domain} · U={lastDebug.u_score.toFixed(3)} · {lastDebug.latency_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
          {!activeSession && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '3rem 1rem' }}>
              <span style={{
                display: 'inline-block',
                fontFamily: '"JetBrains Mono",monospace', fontSize: '.66rem', fontWeight: 600,
                letterSpacing: '.1em', textTransform: 'uppercase',
                background: 'var(--tag-bg)', color: 'var(--accent)',
                padding: '.2rem .7rem', borderRadius: '999px', marginBottom: '1rem',
              }}>AUA v1.0</span>
              <h2 style={{ fontFamily: '"DM Serif Display",Georgia,serif', fontSize: '1.75rem', lineHeight: 1.2, margin: '0 0 .65rem', color: 'var(--ink)' }}>
                Framework Chat
              </h2>
              <p style={{ fontSize: '.9rem', color: 'var(--muted)', maxWidth: '320px', marginBottom: '1.5rem', lineHeight: 1.6 }}>
                Multi-specialist routing with utility scoring and arbitration.
              </p>
              <button onClick={newSession} style={{
                padding: '.5rem 1.3rem', background: 'var(--accent)', color: '#fff',
                borderRadius: '8px', border: 'none', fontSize: '.85rem', fontWeight: 500, cursor: 'pointer',
              }}
                onMouseEnter={e => (e.currentTarget.style.background = '#3730a3')}
                onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}
              >
                Start a new chat
              </button>
            </div>
          )}

          {messages.map(m => (
            <div key={m.id} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
              {m.role === 'user' ? (
                <div style={{
                  maxWidth: '68%', background: 'var(--accent)', color: '#fff',
                  borderRadius: '8px 8px 2px 8px', padding: '.6rem .95rem',
                  fontSize: '.88rem', lineHeight: 1.65,
                }}>
                  <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{m.content}</p>
                </div>
              ) : (
                <div style={{
                  maxWidth: '78%', background: '#fff',
                  border: '1px solid var(--line)', borderLeft: '3px solid var(--accent)',
                  borderRadius: '0 8px 8px 0', padding: '.8rem 1rem',
                  fontSize: '.88rem', lineHeight: 1.7, color: 'var(--ink)',
                }}>
                  <p className="prose-aua" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{m.content}</p>
                  {m.domain && (
                    <p style={{
                      marginTop: '.5rem', marginBottom: 0,
                      fontFamily: '"JetBrains Mono",monospace', fontSize: '.66rem', color: 'var(--muted)',
                      paddingTop: '.45rem', borderTop: '1px solid var(--line)',
                    }}>
                      {m.domain} · {m.routing_mode} · U={m.u_score?.toFixed(3)} · {m.latency_ms?.toFixed(0)}ms
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
              <div style={{
                background: '#fff', border: '1px solid var(--line)', borderLeft: '3px solid var(--accent)',
                borderRadius: '0 8px 8px 0', padding: '.65rem .9rem',
                display: 'flex', gap: '5px', alignItems: 'center',
              }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: 'var(--accent)', opacity: .7,
                    animation: 'aua-bounce 1.1s ease-in-out infinite',
                    animationDelay: `${i * 0.18}s`,
                  }} />
                ))}
                <style>{`@keyframes aua-bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}`}</style>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          borderTop: '2px solid var(--line)', padding: '.9rem 1.5rem 1.1rem',
          background: '#fafaf8', flexShrink: 0,
        }}>
          <div style={{ display: 'flex', gap: '.65rem', alignItems: 'flex-end' }}>
            <textarea value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              placeholder={activeSession ? 'Ask anything… (Enter to send)' : 'Create a chat first'}
              disabled={!activeSession || loading}
              rows={1}
              style={{
                flex: 1, resize: 'none', padding: '.6rem .9rem',
                borderRadius: '8px', border: '1px solid var(--line)',
                background: '#fff', color: 'var(--ink)',
                fontSize: '.88rem', lineHeight: 1.5,
                fontFamily: '"DM Sans",system-ui,sans-serif',
                outline: 'none', maxHeight: '130px', overflowY: 'auto',
                transition: 'border-color .15s, box-shadow .15s',
              }}
              onFocus={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.boxShadow = '0 0 0 2px #c7d2fe' }}
              onBlur={e => { e.currentTarget.style.borderColor = 'var(--line)'; e.currentTarget.style.boxShadow = 'none' }}
            />
            <button onClick={send} disabled={!input.trim() || loading || !activeSession} style={{
              padding: '.6rem .85rem', background: 'var(--accent)', color: '#fff',
              border: 'none', borderRadius: '8px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
              opacity: (!input.trim() || loading || !activeSession) ? .4 : 1,
              transition: 'background .15s, opacity .15s',
            }}
              onMouseEnter={e => { if (!(e.currentTarget as HTMLButtonElement).disabled) e.currentTarget.style.background = '#3730a3' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'var(--accent)' }}
            >
              <Send size={15} />
            </button>
          </div>
        </div>
      </main>

      {showDebugger && <DebuggerPanel debug={lastDebug} loading={loading} />}
      {showControls && <ControlsDrawer onClose={() => setShowControls(false)} />}
    </div>
  )
}
