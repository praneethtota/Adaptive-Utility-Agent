'use client'

import type { RouteDebug } from '@/types'

interface Props {
  debug: RouteDebug | null
  loading: boolean
}

export function DebuggerPanel({ debug, loading }: Props) {
  return (
    <aside style={{
      width: '268px',
      flexShrink: 0,
      borderLeft: '1px solid var(--line)',
      background: '#fafaf8',
      display: 'flex',
      flexDirection: 'column',
      overflowY: 'auto',
    }}>
      {/* Header — matches HTML page h2 section heading style */}
      <div style={{
        padding: '.85rem 1.1rem .75rem',
        borderBottom: '2px solid var(--line)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span style={{
          fontFamily: '"DM Serif Display", Georgia, serif',
          fontSize: '1rem',
          color: 'var(--ink)',
          lineHeight: 1.2,
        }}>
          Framework Debugger
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '.65rem',
          fontWeight: 600,
          letterSpacing: '.08em',
          textTransform: 'uppercase',
          background: 'var(--tag-bg)',
          color: 'var(--accent)',
          padding: '.15rem .5rem',
          borderRadius: '999px',
        }}>
          live
        </span>
      </div>

      {loading && (
        <div style={{
          padding: '.85rem 1.1rem',
          fontSize: '.8rem',
          color: 'var(--accent)',
          fontFamily: '"JetBrains Mono", monospace',
          borderBottom: '1px solid var(--line)',
          background: 'var(--tag-bg)',
        }}>
          Routing query…
        </div>
      )}

      {!debug && !loading && (
        <div style={{ padding: '2rem 1.1rem', textAlign: 'center' }}>
          <p style={{ fontSize: '.82rem', color: 'var(--muted)', fontStyle: 'italic', margin: 0 }}>
            Send a message to see routing debug info
          </p>
        </div>
      )}

      {debug && (
        <div style={{ padding: '0', fontSize: '.82rem' }}>

          {/* Route Summary — callout style */}
          <Section label="Route Summary">
            <Row label="Domain"         value={debug.domain}                    accent />
            <Row label="Mode"           value={debug.routing_mode}              />
            <Row label="U Score"        value={debug.u_score.toFixed(4)}        />
            <Row label="Confidence"     value={debug.confidence.toFixed(3)}     />
            <Row label="Latency"        value={`${debug.latency_ms.toFixed(0)}ms`} />
            <Row label="Contradictions" value={String(debug.contradictions_detected)} />
          </Section>

          {/* Classifier output */}
          {Object.keys(debug.domain_distribution).length > 0 && (
            <Section label="Classifier Output">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
                {Object.entries(debug.domain_distribution)
                  .sort(([, a], [, b]) => b - a)
                  .map(([domain, prob]) => (
                    <div key={domain}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ color: '#374151', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%' }}>
                          {domain}
                        </span>
                        <span style={{ fontFamily: '"JetBrains Mono", monospace', color: 'var(--muted)', fontSize: '.75rem' }}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      {/* Weight-track style from domain pages */}
                      <div style={{
                        height: '8px',
                        background: 'var(--line)',
                        borderRadius: '999px',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${prob * 100}%`,
                          background: 'var(--accent)',
                          borderRadius: '999px',
                          transition: 'width .3s ease',
                        }} />
                      </div>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          {/* Utility formula — code block style */}
          <Section label="Utility Breakdown">
            <div style={{
              background: 'var(--soft)',
              border: '1px solid var(--line)',
              borderRadius: '6px',
              padding: '.6rem .85rem',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '.75rem',
              lineHeight: 1.7,
              color: '#374151',
            }}>
              U = w_e·E + w_c·C + w_k·K
              <br />
              <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
                = {debug.u_score.toFixed(4)}
              </span>
            </div>
          </Section>

          {/* Specialist responses */}
          {debug.specialist_responses && Object.keys(debug.specialist_responses).length > 0 && (
            <Section label="Specialist Calls">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                {Object.entries(debug.specialist_responses).map(([name, resp]) => (
                  <div key={name}>
                    <p style={{
                      margin: '0 0 4px',
                      fontWeight: 600,
                      color: 'var(--ink)',
                      fontSize: '.78rem',
                    }}>
                      {name}
                    </p>
                    <p style={{
                      margin: 0,
                      color: 'var(--muted)',
                      background: 'var(--soft)',
                      borderRadius: '6px',
                      padding: '.5rem .7rem',
                      lineHeight: 1.6,
                      fontSize: '.78rem',
                      display: '-webkit-box',
                      WebkitLineClamp: 4,
                      WebkitBoxOrient: 'vertical' as const,
                      overflow: 'hidden',
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

/* Section block — matches the HTML page h2 section pattern */
function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ borderBottom: '1px solid var(--line)' }}>
      <div style={{
        padding: '.6rem 1.1rem .4rem',
        borderBottom: '1px solid var(--line)',
        background: 'var(--bg)',
      }}>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '.65rem',
          fontWeight: 700,
          letterSpacing: '.1em',
          textTransform: 'uppercase',
          color: 'var(--muted)',
        }}>
          {label}
        </span>
      </div>
      <div style={{ padding: '.75rem 1.1rem' }}>
        {children}
      </div>
    </div>
  )
}

/* Row — key/value pair matching the page's table-row style */
function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      padding: '3px 0',
      borderBottom: '1px solid #f0ede6',
    }}>
      <span style={{ color: 'var(--muted)', fontSize: '.78rem' }}>{label}</span>
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: '.75rem',
        fontWeight: accent ? 700 : 400,
        color: accent ? 'var(--accent)' : '#374151',
      }}>
        {value}
      </span>
    </div>
  )
}
