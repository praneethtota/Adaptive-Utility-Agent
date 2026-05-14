'use client'

import type { RouteDebug } from '@/types'

interface Props {
  debug: RouteDebug | null
  loading: boolean
}

/* Debugger palette — mint/teal (default) */
const MINT = {
  bg:         '#ecfdf5',
  bgHeader:   '#d1fae5',
  bgCode:     '#bbf7d0',
  border:     '#6ee7b7',
  borderTop:  '#34d399',
  accent:     '#0f766e',
  accentSoft: '#ccfbf1',
  ink:        '#134e4a',
  muted:      '#4d7c72',
  row:        '#a7f3d0',
}

/* Debugger palette — indigo/purple for VCG mode */
const INDIGO = {
  bg:         '#eef2ff',
  bgHeader:   '#e0e7ff',
  bgCode:     '#c7d2fe',
  border:     '#a5b4fc',
  borderTop:  '#6366f1',
  accent:     '#4338ca',
  accentSoft: '#e0e7ff',
  ink:        '#1e1b4b',
  muted:      '#4338ca',
  row:        '#c7d2fe',
}

export function DebuggerPanel({ debug, loading }: Props) {
  const isVCG = debug?.routing_mode === 'vcg'
  const D = isVCG ? INDIGO : MINT

  return (
    <aside style={{
      width: '262px', flexShrink: 0,
      display: 'flex', flexDirection: 'column', overflowY: 'auto',
      background: D.bg,
      borderLeft: `1px solid ${D.border}`,
      boxShadow: isVCG
        ? '-2px 0 10px rgba(67,56,202,.12)'
        : '-2px 0 10px rgba(15,118,110,.08)',
      zIndex: 1,
      transition: 'background .3s, border-color .3s, box-shadow .3s',
    }}>
      {/* Header */}
      <div style={{
        padding: '.85rem 1.1rem .75rem',
        borderBottom: `2px solid ${D.borderTop}`,
        background: D.bgHeader,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        transition: 'background .3s, border-color .3s',
      }}>
        <span style={{
          fontFamily: '"DM Serif Display",Georgia,serif',
          fontSize: '1rem', color: D.ink, lineHeight: 1.2,
        }}>
          Framework Debugger{isVCG ? ' — VCG' : ''}
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono",monospace', fontSize: '.62rem', fontWeight: 700,
          letterSpacing: '.08em', textTransform: 'uppercase',
          background: D.accentSoft, color: D.accent,
          padding: '.15rem .5rem', borderRadius: '999px',
          border: `1px solid ${D.border}`,
        }}>
          {isVCG ? 'vcg' : 'live'}
        </span>
      </div>

      {loading && (
        <div style={{
          padding: '.75rem 1.1rem', fontSize: '.78rem',
          color: D.accent, fontFamily: '"JetBrains Mono",monospace',
          borderBottom: `1px solid ${D.border}`,
          background: D.bgHeader,
        }}>
          Routing query…
        </div>
      )}

      {!debug && !loading && (
        <div style={{ padding: '2rem 1.1rem', textAlign: 'center' }}>
          <p style={{ fontSize: '.8rem', color: D.muted, fontStyle: 'italic', margin: 0 }}>
            Send a message to see routing debug info
          </p>
        </div>
      )}

      {debug && (
        <div style={{ fontSize: '.8rem' }}>

          <Section label="Route Summary" colors={D}>
            <Row label="Domain"         value={debug.domain}                          accent colors={D} />
            <Row label="Mode"           value={debug.routing_mode}                    colors={D} />
            <Row label="U Score"        value={debug.u_score.toFixed(4)}              colors={D} />
            <Row label="Confidence"     value={debug.confidence.toFixed(3)}           colors={D} />
            <Row label="Latency"        value={`${debug.latency_ms.toFixed(0)}ms`}    colors={D} />
            <Row label="Contradictions" value={String(debug.contradictions_detected)} colors={D} />
          </Section>

          {/* VCG welfare scores — only shown in VCG mode */}
          {isVCG && debug.welfare_scores && Object.keys(debug.welfare_scores).length > 0 && (
            <Section label="VCG Welfare Scores" colors={D}>
              <p style={{ margin: '0 0 .6rem', color: D.muted, fontSize: '.72rem', lineHeight: 1.5 }}>
                W_i = P(domain) × confidence × prior_U
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.55rem' }}>
                {Object.entries(debug.welfare_scores)
                  .sort(([, a], [, b]) => b - a)
                  .map(([name, w]) => {
                    const maxW = Math.max(...Object.values(debug.welfare_scores!))
                    const isWinner = w === maxW
                    return (
                      <div key={name} style={{
                        background: isWinner ? D.bgCode : 'transparent',
                        border: isWinner ? `1px solid ${D.border}` : '1px solid transparent',
                        borderRadius: '6px', padding: isWinner ? '.4rem .6rem' : '.1rem 0',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ color: D.ink, fontWeight: isWinner ? 700 : 400, fontSize: '.76rem' }}>
                            {name} {isWinner ? '✓' : ''}
                          </span>
                          <span style={{ fontFamily: '"JetBrains Mono",monospace', color: isWinner ? D.accent : D.muted, fontWeight: isWinner ? 700 : 400, fontSize: '.72rem' }}>
                            {w.toFixed(4)}
                          </span>
                        </div>
                        <div style={{ height: '6px', background: D.bg, borderRadius: '999px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${(w / (maxW || 1)) * 100}%`,
                            background: isWinner ? D.accent : D.muted,
                            borderRadius: '999px', transition: 'width .3s', opacity: isWinner ? 1 : 0.45,
                          }} />
                        </div>
                      </div>
                    )
                  })}
              </div>
            </Section>
          )}

          {Object.keys(debug.domain_distribution).length > 0 && (
            <Section label="Classifier Output" colors={D}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.55rem' }}>
                {Object.entries(debug.domain_distribution)
                  .sort(([, a], [, b]) => b - a)
                  .map(([domain, prob]) => (
                    <div key={domain}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ color: D.ink, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%', fontSize: '.76rem' }}>{domain}</span>
                        <span style={{ fontFamily: '"JetBrains Mono",monospace', color: D.muted, fontSize: '.72rem' }}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '7px', background: D.bgCode, borderRadius: '999px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${prob * 100}%`, background: D.accent, borderRadius: '999px', transition: 'width .3s' }} />
                      </div>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          <Section label="Utility Breakdown" colors={D}>
            <div style={{
              background: D.bgCode, border: `1px solid ${D.border}`,
              borderRadius: '6px', padding: '.55rem .8rem',
              fontFamily: '"JetBrains Mono",monospace', fontSize: '.72rem',
              lineHeight: 1.7, color: D.ink,
            }}>
              U = w_e·E + w_c·C + w_k·K<br />
              <span style={{ color: D.accent, fontWeight: 700 }}>= {debug.u_score.toFixed(4)}</span>
              {isVCG && (
                <><br /><span style={{ color: D.muted, fontSize: '.68rem' }}>Winner selected by welfare maximization</span></>
              )}
            </div>
          </Section>

          {debug.specialist_responses && Object.keys(debug.specialist_responses).length > 0 && (
            <Section label="Specialist Calls" colors={D}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.7rem' }}>
                {Object.entries(debug.specialist_responses).map(([name, resp]) => (
                  <div key={name}>
                    <p style={{ margin: '0 0 3px', fontWeight: 600, color: D.ink, fontSize: '.76rem' }}>{name}</p>
                    <p style={{
                      margin: 0, color: D.muted, background: D.bgCode,
                      border: `1px solid ${D.border}`, borderRadius: '6px',
                      padding: '.45rem .65rem', lineHeight: 1.6, fontSize: '.75rem',
                      overflow: 'hidden', display: '-webkit-box',
                      WebkitLineClamp: 4, WebkitBoxOrient: 'vertical' as const,
                    }}>
                      {String(resp).slice(0, 200)}{String(resp).length > 200 ? '…' : ''}
                    </p>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </div>
      )}
    </aside>
  )
}

function Section({ label, children, colors: C }: { label: string; children: React.ReactNode; colors: typeof MINT }) {
  return (
    <div style={{ borderBottom: `1px solid ${C.border}` }}>
      <div style={{ padding: '.5rem 1.1rem .35rem', borderBottom: `1px solid ${C.border}`, background: C.bgHeader }}>
        <span style={{
          fontFamily: '"JetBrains Mono",monospace', fontSize: '.62rem',
          fontWeight: 700, letterSpacing: '.1em', textTransform: 'uppercase', color: C.muted,
        }}>
          {label}
        </span>
      </div>
      <div style={{ padding: '.7rem 1.1rem' }}>{children}</div>
    </div>
  )
}

function Row({ label, value, accent, colors: C }: { label: string; value: string; accent?: boolean; colors: typeof MINT }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '3px 0', borderBottom: `1px solid ${C.row}`,
    }}>
      <span style={{ color: C.muted, fontSize: '.76rem' }}>{label}</span>
      <span style={{
        fontFamily: '"JetBrains Mono",monospace', fontSize: '.73rem',
        fontWeight: accent ? 700 : 400, color: accent ? C.accent : C.ink,
      }}>{value}</span>
    </div>
  )
}
