'use client'

import { useState, useEffect } from 'react'
import { X, RefreshCw } from 'lucide-react'
import { getConfig } from '@/lib/api'

interface Props { onClose: () => void }

const C = {
  bg:         '#fff7ed',
  bgHeader:   '#fed7aa',
  bgCode:     '#ffedd5',
  border:     '#fdba74',
  borderTop:  '#f97316',
  accent:     '#c2410c',
  accentSoft: '#fff7ed',
  ink:        '#431407',
  muted:      '#92400e',
  row:        '#fed7aa',
}

const ROUTER_URL = process.env.AUA_ROUTER_URL || 'http://localhost:8000'

export function ControlsDrawer({ onClose }: Props) {
  const [config, setConfig] = useState<any>(null)
  const [reloading, setReloading] = useState(false)
  const [singleThreshold, setSingleThreshold] = useState(0.75)
  const [fanoutThreshold, setFanoutThreshold] = useState(0.30)
  const [reloadMsg, setReloadMsg] = useState('')
  const [arbitrationMode, setArbitrationMode] = useState<'pairwise' | 'vcg'>('pairwise')
  const [vcgToggling, setVcgToggling] = useState(false)
  const [vcgMsg, setVcgMsg] = useState('')
  const [specialistCount, setSpecialistCount] = useState(0)

  useEffect(() => {
    getConfig().then(cfg => {
      if (cfg) {
        setConfig(cfg)
        setSingleThreshold(cfg.router?.single_domain_threshold ?? 0.75)
        setFanoutThreshold(cfg.router?.fanout_threshold ?? 0.30)
        setArbitrationMode(cfg.router?.arbitration_mode ?? 'pairwise')
        setSpecialistCount(cfg.specialists?.length ?? 0)
      }
    })
  }, [])

  async function reloadConfig() {
    setReloading(true); setReloadMsg('')
    try {
      const r = await fetch(`${ROUTER_URL}/config/reload`, { method: 'POST' })
      setReloadMsg(r.ok ? '✓ Config reloaded' : 'Reload sent (check router logs)')
    } catch {
      setReloadMsg('Reload signal sent (run: aua config reload)')
    } finally {
      setReloading(false)
      setTimeout(() => setReloadMsg(''), 3000)
    }
  }

  async function toggleVCG() {
    const newMode = arbitrationMode === 'vcg' ? 'pairwise' : 'vcg'
    setVcgToggling(true); setVcgMsg('')
    try {
      const r = await fetch(`${ROUTER_URL}/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arbitration_mode: newMode, persist: true }),
      })
      if (r.ok) {
        setArbitrationMode(newMode)
        setVcgMsg(`✓ Switched to ${newMode}`)
      } else {
        setVcgMsg('Failed — check router logs')
      }
    } catch {
      setVcgMsg('Router unreachable')
    } finally {
      setVcgToggling(false)
      setTimeout(() => setVcgMsg(''), 3000)
    }
  }

  const vcgDisabled = specialistCount < 2 || vcgToggling
  const vcgOn = arbitrationMode === 'vcg'

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex' }}>
      <div style={{ flex: 1, background: 'rgba(24,24,27,.25)', backdropFilter: 'blur(2px)' }} onClick={onClose} />
      <aside style={{
        width: '310px', height: '100%',
        background: C.bg, overflowY: 'auto',
        borderLeft: `1px solid ${C.border}`,
        boxShadow: '-4px 0 24px rgba(194,65,12,.12)',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{
          padding: '.9rem 1.25rem .8rem',
          borderBottom: `2px solid ${C.borderTop}`,
          background: C.bgHeader,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontFamily: '"DM Serif Display",Georgia,serif', fontSize: '1.05rem', color: C.ink, lineHeight: 1.2 }}>
            AUA Controls
          </span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.muted, padding: '3px', borderRadius: '4px', display: 'flex', alignItems: 'center' }}
            onMouseEnter={e => (e.currentTarget.style.color = C.ink)}
            onMouseLeave={e => (e.currentTarget.style.color = C.muted)}>
            <X size={17} />
          </button>
        </div>

        <div style={{ padding: '0', fontSize: '.82rem', flex: 1 }}>

          {/* Arbitration Mode */}
          <Section label="Arbitration Mode" colors={C}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', gap: '0' }}>
                  {(['pairwise', 'vcg'] as const).map(mode => (
                    <button key={mode}
                      onClick={vcgDisabled || arbitrationMode === mode ? undefined : toggleVCG}
                      disabled={vcgDisabled && arbitrationMode !== mode}
                      title={
                        specialistCount < 2
                          ? 'Requires at least 2 specialists in aua_config.yaml'
                          : mode === 'vcg'
                          ? 'Welfare maximization: W = P(domain) × confidence × prior_U'
                          : 'Standard 4-check pairwise arbitration (default)'
                      }
                      style={{
                        padding: '.35rem .75rem',
                        background: arbitrationMode === mode
                          ? (mode === 'vcg' ? '#4338ca' : C.accent)
                          : C.bgCode,
                        color: arbitrationMode === mode ? '#fff' : C.muted,
                        border: `1px solid ${C.border}`,
                        borderRadius: mode === 'pairwise' ? '6px 0 0 6px' : '0 6px 6px 0',
                        fontSize: '.75rem', fontWeight: 600,
                        cursor: (vcgDisabled && arbitrationMode !== mode) ? 'not-allowed' : arbitrationMode === mode ? 'default' : 'pointer',
                        opacity: (vcgDisabled && arbitrationMode !== mode) ? 0.5 : 1,
                        transition: 'background .15s, color .15s',
                        fontFamily: '"JetBrains Mono",monospace',
                      }}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
                {vcgToggling && <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite', color: C.muted }} />}
              </div>

              {specialistCount < 2 && (
                <p style={{ margin: 0, color: C.accent, fontSize: '.72rem', lineHeight: 1.4, fontStyle: 'italic' }}>
                  VCG requires ≥ 2 specialists in aua_config.yaml
                </p>
              )}

              {vcgOn && specialistCount >= 2 && (
                <p style={{ margin: 0, color: C.muted, fontSize: '.72rem', lineHeight: 1.5 }}>
                  Selecting specialist with highest W = P(domain) × confidence × prior_U.
                  Fires on fanout queries. Response shows{' '}
                  <code style={{ fontFamily: '"JetBrains Mono",monospace', background: C.bgCode, padding: '1px 3px', borderRadius: '3px' }}>routing_mode: vcg</code>{' '}
                  and welfare scores per specialist in the debugger.
                </p>
              )}

              {!vcgOn && specialistCount >= 2 && (
                <p style={{ margin: 0, color: C.muted, fontSize: '.72rem', lineHeight: 1.5 }}>
                  Standard 4-check pairwise arbitration. Switch to VCG for welfare-maximizing selection across all fanout specialists (+43.3pp vs no routing in hardware validation).
                </p>
              )}

              {vcgMsg && <p style={{ margin: 0, color: C.accent, fontSize: '.75rem', fontWeight: 600 }}>{vcgMsg}</p>}
            </div>
            <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
          </Section>

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
                    <span style={{ fontFamily: '"JetBrains Mono",monospace', color: C.accent, fontWeight: 700, fontSize: '.78rem' }}>{value.toFixed(2)}</span>
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
              onMouseLeave={e => { if (!reloading) e.currentTarget.style.background = C.accent }}>
              <RefreshCw size={12} style={{ animation: reloading ? 'spin 1s linear infinite' : 'none' }} />
              {reloading ? 'Reloading…' : 'Reload Config'}
            </button>
            {reloadMsg && <p style={{ marginTop: '.5rem', textAlign: 'center', color: C.accent, fontSize: '.78rem', fontWeight: 600 }}>{reloadMsg}</p>}
            <p style={{ marginTop: '.5rem', color: C.muted, fontSize: '.74rem', lineHeight: 1.5 }}>
              Applies hot-reloadable settings without restart. For model or port changes, restart{' '}
              <code style={{ fontFamily: '"JetBrains Mono",monospace', background: C.bgCode, padding: '1px 4px', borderRadius: '3px' }}>aua serve</code>.
            </p>
          </Section>

          {/* Live config */}
          {config && (
            <Section label="Live Config" colors={C}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <Row label="Backend"     value={config.backend || '—'}               colors={C} />
                <Row label="Router port" value={String(config.router?.port || 8000)} colors={C} />
                <Row label="Arbitration" value={arbitrationMode}                     colors={C} />
                <Row label="Version"     value={config.version || '—'}               colors={C} />
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
                'aua serve --arbitration-mode vcg',
                'aua eval run --dataset evals/coding_smoke.yaml',
                'aua token create --scope aua:query --expires 30d',
                'aua certs generate',
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
        <span style={{ fontFamily: '"JetBrains Mono",monospace', fontSize: '.62rem', fontWeight: 700, letterSpacing: '.1em', textTransform: 'uppercase', color: C.muted }}>
          {label}
        </span>
      </div>
      <div style={{ padding: '.8rem 1.25rem' }}>{children}</div>
    </div>
  )
}

function Row({ label, value, colors: C }: { label: string; value: string; colors: typeof C }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', padding: '4px 0', borderBottom: `1px solid ${C.row}` }}>
      <span style={{ color: C.muted, fontSize: '.76rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
      <span style={{ fontFamily: '"JetBrains Mono",monospace', fontSize: '.73rem', color: C.ink, marginLeft: '.5rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '130px' }}>{value}</span>
    </div>
  )
}
