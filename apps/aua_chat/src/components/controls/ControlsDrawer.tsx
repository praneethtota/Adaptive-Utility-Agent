'use client'

import { useState, useEffect } from 'react'
import { X, RefreshCw } from 'lucide-react'
import { getConfig } from '@/lib/api'

interface Props {
  onClose: () => void
}

export function ControlsDrawer({ onClose }: Props) {
  const [config, setConfig] = useState<any>(null)
  const [reloading, setReloading] = useState(false)
  const [singleThreshold, setSingleThreshold] = useState(0.75)
  const [fanoutThreshold, setFanoutThreshold] = useState(0.30)
  const [reloadMsg, setReloadMsg] = useState('')

  useEffect(() => {
    getConfig().then(c => {
      if (c) {
        setConfig(c)
        setSingleThreshold(c.router?.single_domain_threshold ?? 0.75)
        setFanoutThreshold(c.router?.fanout_threshold ?? 0.30)
      }
    })
  }, [])

  async function reloadConfig() {
    setReloading(true)
    setReloadMsg('')
    try {
      const r = await fetch(`${process.env.AUA_ROUTER_URL || 'http://localhost:8000'}/config/reload`, { method: 'POST' })
      if (r.ok) {
        setReloadMsg('✓ Config reloaded')
      } else {
        setReloadMsg('Reload sent (check router logs)')
      }
    } catch {
      setReloadMsg('Reload signal sent (run: aua config reload)')
    } finally {
      setReloading(false)
      setTimeout(() => setReloadMsg(''), 3000)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="flex-1 bg-black/30 backdrop-blur-sm" onClick={onClose} />
      <aside className="w-80 bg-white border-l border-[#e4e1da] flex flex-col h-full shadow-xl overflow-y-auto">
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#e4e1da]">
          <h2 className="font-semibold text-sm text-[#18181b]">AUA Controls</h2>
          <button onClick={onClose} className="p-1 text-[#6b7280] hover:text-[#18181b]"><X size={18} /></button>
        </div>

        <div className="px-5 py-4 space-y-6 text-xs">

          {/* Routing thresholds */}
          <section>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-3">Routing Thresholds</p>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-[#374151]">Single domain threshold</label>
                  <span className="font-mono text-[#4338ca]">{singleThreshold.toFixed(2)}</span>
                </div>
                <input type="range" min={0.5} max={0.99} step={0.01} value={singleThreshold}
                  onChange={e => setSingleThreshold(Number(e.target.value))}
                  className="w-full accent-[#4338ca]" />
                <p className="text-[#9ca3af] mt-0.5">Higher = more focused routing to single specialist</p>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-[#374151]">Fanout threshold</label>
                  <span className="font-mono text-[#4338ca]">{fanoutThreshold.toFixed(2)}</span>
                </div>
                <input type="range" min={0.1} max={0.6} step={0.01} value={fanoutThreshold}
                  onChange={e => setFanoutThreshold(Number(e.target.value))}
                  className="w-full accent-[#4338ca]" />
                <p className="text-[#9ca3af] mt-0.5">Lower = more cross-domain fanout queries</p>
              </div>
            </div>
          </section>

          {/* Config reload */}
          <section>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-3">Config Management</p>
            <button onClick={reloadConfig} disabled={reloading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#4338ca] text-white rounded-lg text-xs font-medium hover:bg-[#3730a3] disabled:opacity-60 transition-colors">
              <RefreshCw size={13} className={reloading ? 'animate-spin' : ''} />
              {reloading ? 'Reloading…' : 'Reload Config'}
            </button>
            {reloadMsg && <p className="mt-2 text-center text-[#4338ca]">{reloadMsg}</p>}
            <p className="mt-2 text-[#9ca3af]">
              Applies hot-reloadable settings (thresholds, logging level) without restart.
              For model/port changes, restart aua serve.
            </p>
          </section>

          {/* Live config */}
          {config && (
            <section>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-3">Live Config</p>
              <div className="space-y-1">
                <Row label="Backend" value={config.backend || '—'} />
                <Row label="Router port" value={String(config.router?.port || 8000)} />
                <Row label="Version" value={config.version || '—'} />
                {config.specialists?.map((s: any) => (
                  <Row key={s.name} label={`${s.name} (${s.field})`} value={s.model} />
                ))}
              </div>
            </section>
          )}

          {/* Shortcuts */}
          <section>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-3">CLI Quick Reference</p>
            <div className="bg-[#f0ede6] rounded-lg p-3 font-mono text-[10px] text-[#374151] space-y-1">
              <p>aua eval run --dataset evals/coding_smoke.yaml</p>
              <p>aua token create --scope aua:query --expires 30d</p>
              <p>aua certs generate</p>
              <p>aua defaults show</p>
              <p>aua extensions list</p>
            </div>
          </section>
        </div>
      </aside>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-0.5">
      <span className="text-[#6b7280] truncate">{label}</span>
      <span className="font-mono text-[#374151] ml-2 truncate max-w-[140px]">{value}</span>
    </div>
  )
}
