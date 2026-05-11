'use client'

import type { RouteDebug } from '@/types'

interface Props {
  debug: RouteDebug | null
  loading: boolean
}

/* Debugger palette — mint/teal (--soft2 / --accent2) */
const D = {
  bg:         '#ecfdf5',   // --soft2
  bgHeader:   '#d1fae5',   // slightly deeper mint for section headers
  bgCode:     '#bbf7d0',   // code block / formula bg
  border:     '#6ee7b7',   // mint border
  borderTop:  '#34d399',   // header top border (2px, accent)
  accent:     '#0f766e',   // --accent2 teal
  accentSoft: '#ccfbf1',   // tag bg
  ink:        '#134e4a',   // dark teal ink
  muted:      '#4d7c72',   // muted teal
  row:        '#a7f3d0',   // row separator
}

export function DebuggerPanel({ debug, loading }: Props) {
  return (
    <aside style={{
      width: '262px', flexShrink: 0,
      display: 'flex', flexDirection: 'column', overflowY: 'auto',
      background: D.bg,
      borderLeft: `1px solid ${D.border}`,
      boxShadow: '-2px 0 10px rgba(15,118,110,.08)',
      zIndex: 1,
    }}>
      {/* Header */}
      <div style={{
        padding: '.85rem 1.1rem .75rem',
        borderBottom: `2px solid ${D.borderTop}`,
        background: D.bgHeader,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{
          fontFamily: '"DM Serif Display",Georgia,serif',
          fontSize: '1rem', color: D.ink, lineHeight: 1.2,
        }}>
          Framework Debugger
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono",monospace', fontSize: '.62rem', fontWeight: 700,
          letterSpacing: '.08em', textTransform: 'uppercase',
          background: D.accentSoft, color: D.accent,
          padding: '.15rem .5rem', borderRadius: '999px',
          border: `1px solid ${D.border}`,
        }}>
          live
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
            <Row label="Domain"         value={debug.domain}                       accent colors={D} />
            <Row label="Mode"           value={debug.routing_mode}                 colors={D} />
            <Row label="U Score"        value={debug.u_score.toFixed(4)}           colors={D} />
            <Row label="Confidence"     value={debug.confidence.toFixed(3)}        colors={D} />
            <Row label="Latency"        value={`${debug.latency_ms.toFixed(0)}ms`} colors={D} />
            <Row label="Contradictions" value={String(debug.contradictions_detected)} colors={D} />
          </Section>

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

function Section({ label, children, colors: C }: { label: string; children: React.ReactNode; colors: typeof D }) {
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

function Row({ label, value, accent, colors: C }: { label: string; value: string; accent?: boolean; colors: typeof D }) {
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
