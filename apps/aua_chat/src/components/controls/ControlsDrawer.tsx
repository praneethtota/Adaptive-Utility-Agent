'use client'

import { useState, useEffect } from 'react'
import { X, RefreshCw } from 'lucide-react'
import { getConfig } from '@/lib/api'

interface Props { onClose: () => void }

/* Controls palette — warm amber (--soft3 / --accent3) */
const C = {
  bg:         '#fff7ed',   // --soft3
  bgHeader:   '#fed7aa',   // deeper amber for section headers
  bgCode:     '#ffedd5',   // code block bg
  border:     '#fdba74',   // amber border
  borderTop:  '#f97316',   // header top border
  accent:     '#c2410c',   // --accent3 orange
  accentSoft: '#fff7ed',   // tag bg
  ink:        '#431407',   // dark amber ink
  muted:      '#92400e',   // muted amber
  row:        '#fed7aa',   // row separator
}

export function ControlsDrawer({ onClose }: Props) {
  const [config, setConfig] = useState<any>(null)
  const [reloading, setReloading] = useState(false)
  const [singleThreshold, setSingleThreshold] = useState(0.75)
  const [fanoutThreshold, setFanoutThreshold] = useState(0.30)
  const [reloadMsg, setReloadMsg] = useState('')

  useEffect(() => {
    getConfig().then(cfg => {
      if (cfg) {
        setConfig(cfg)
        setSingleThreshold(cfg.router?.single_domain_threshold ?? 0.75)
        setFanoutThreshold(cfg.router?.fanout_threshold ?? 0.30)
      }
    })
  }, [])

  async function reloadConfig() {
    setReloading(true); setReloadMsg('')
    try {
      const r = await fetch(`${process.env.AUA_ROUTER_URL || 'http://localhost:8000'}/config/reload`, { method: 'POST' })
      setReloadMsg(r.ok ? '✓ Config reloaded' : 'Reload sent (check router logs)')
    } catch {
      setReloadMsg('Reload signal sent (run: aua config reload)')
    } finally {
      setReloading(false)
      setTimeout(() => setReloadMsg(''), 3000)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex' }}>
      {/* Backdrop */}
      <div style={{ flex: 1, background: 'rgba(24,24,27,.25)', backdropFilter: 'blur(2px)' }} onClick={onClose} />

      {/* Drawer panel */}
      <aside style={{
        width: '310px', height: '100%',
        background: C.bg, overflowY: 'auto',
        borderLeft: `1px solid ${C.border}`,
        boxShadow: '-4px 0 24px rgba(194,65,12,.12)',
        display: 'flex', flexDirection: 'column',
      }}>
        {/* Header */}
        <div style={{
          padding: '.9rem 1.25rem .8rem',
          borderBottom: `2px solid ${C.borderTop}`,
          background: C.bgHeader,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{
            fontFamily: '"DM Serif Display",Georgia,serif',
            fontSize: '1.05rem', color: C.ink, lineHeight: 1.2,
          }}>
            AUA Controls
          </span>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: C.muted, padding: '3px', borderRadius: '4px',
            display: 'flex', alignItems: 'center',
          }}
            onMouseEnter={e => (e.currentTarget.style.color = C.ink)}
            onMouseLeave={e => (e.currentTarget.style.color = C.muted)}
          >
            <X size={17} />
          </button>
        </div>

        <div style={{ padding: '0', fontSize: '.82rem', flex: 1 }}>

          {/* Routing thresholds */}
          <Section label="Routing Thresholds" colors={C}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {[
                { label: 'Single domain threshold', value: singleThreshold, set: setSingleThreshold, min: 0.5, max: 0.99, hint: 'Higher = more focused routing to single specialist' },
                { label: 'Fanout threshold', value: fanoutThreshold, set: setFanoutThreshold, min: 0.1, max: 0.6, hint: 'Lower = more cross-domain fanout queries' },
              ].map(({ label, value, set, min, max, hint }) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '.3rem' }}>
                    <label style={{ color: C.ink, fontSize: '.8rem' }}>{label}</label>
                    <span style={{ fontFamily: '"JetBrains Mono",monospace', color: C.accent, fontWeight: 700, fontSize: '.78rem' }}>
                      {value.toFixed(2)}
                    </span>
                  </div>
                  <input type="range" min={min} max={max} step={0.01} value={value}
                    onChange={e => set(Number(e.target.value))}
                    style={{ width: '100%', accentColor: C.accent }} />
                  <p style={{ margin: '.2rem 0 0', color: C.muted, fontSize: '.74rem' }}>{hint}</p>
                </div>
              ))}
            </div>
          </Section>

          {/* Config reload */}
          <Section label="Config Management" colors={C}>
            <button onClick={reloadConfig} disabled={reloading} style={{
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '.45rem',
              padding: '.55rem 1rem',
              background: reloading ? C.bgCode : C.accent,
              color: reloading ? C.muted : '#fff',
              border: `1px solid ${C.border}`,
              borderRadius: '8px', fontSize: '.8rem', fontWeight: 500,
              cursor: reloading ? 'not-allowed' : 'pointer', transition: 'background .15s',
            }}
              onMouseEnter={e => { if (!reloading) e.currentTarget.style.background = '#9a3412' }}
              onMouseLeave={e => { if (!reloading) e.currentTarget.style.background = C.accent }}
            >
              <RefreshCw size={12} style={{ animation: reloading ? 'spin 1s linear infinite' : 'none' }} />
              {reloading ? 'Reloading…' : 'Reload Config'}
            </button>
            {reloadMsg && (
              <p style={{ marginTop: '.5rem', textAlign: 'center', color: C.accent, fontSize: '.78rem', fontWeight: 600 }}>
                {reloadMsg}
              </p>
            )}
            <p style={{ marginTop: '.5rem', color: C.muted, fontSize: '.74rem', lineHeight: 1.5 }}>
              Applies hot-reloadable settings without restart. For model or port changes, restart <code style={{ fontFamily: '"JetBrains Mono",monospace', background: C.bgCode, padding: '1px 4px', borderRadius: '3px' }}>aua serve</code>.
            </p>
            <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
          </Section>

          {/* Live config */}
          {config && (
            <Section label="Live Config" colors={C}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <Row label="Backend"     value={config.backend || '—'}             colors={C} />
                <Row label="Router port" value={String(config.router?.port || 8000)} colors={C} />
                <Row label="Version"     value={config.version || '—'}             colors={C} />
                {config.specialists?.map((s: any) => (
                  <Row key={s.name} label={`${s.name} (${s.field})`} value={s.model} colors={C} />
                ))}
              </div>
            </Section>
          )}

          {/* CLI reference */}
          <Section label="CLI Quick Reference" colors={C}>
            <div style={{
              background: C.bgCode, border: `1px solid ${C.border}`,
              borderRadius: '6px', padding: '.65rem .85rem',
              fontFamily: '"JetBrains Mono",monospace', fontSize: '.72rem',
              color: C.ink, lineHeight: 1.8,
            }}>
              {[
                'aua eval run --dataset evals/coding_smoke.yaml',
                'aua token create --scope aua:query --expires 30d',
                'aua certs generate',
                'aua defaults show',
                'aua extensions list',
              ].map(cmd => <div key={cmd}>{cmd}</div>)}
            </div>
          </Section>

        </div>
      </aside>
    </div>
  )
}

function Section({ label, children, colors: C }: { label: string; children: React.ReactNode; colors: typeof C }) {
  return (
    <div style={{ borderBottom: `1px solid ${C.border}` }}>
      <div style={{ padding: '.55rem 1.25rem .4rem', borderBottom: `1px solid ${C.border}`, background: C.bgHeader }}>
        <span style={{
          fontFamily: '"JetBrains Mono",monospace', fontSize: '.62rem',
          fontWeight: 700, letterSpacing: '.1em', textTransform: 'uppercase', color: C.muted,
        }}>
          {label}
        </span>
      </div>
      <div style={{ padding: '.8rem 1.25rem' }}>{children}</div>
    </div>
  )
}

function Row({ label, value, colors: C }: { label: string; value: string; colors: typeof C }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '4px 0', borderBottom: `1px solid ${C.row}`,
    }}>
      <span style={{ color: C.muted, fontSize: '.76rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
      <span style={{ fontFamily: '"JetBrains Mono",monospace', fontSize: '.73rem', color: C.ink, marginLeft: '.5rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '130px' }}>{value}</span>
    </div>
  )
}
