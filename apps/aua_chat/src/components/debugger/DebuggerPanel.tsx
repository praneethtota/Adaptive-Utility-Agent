'use client'

import type { RouteDebug } from '@/types'

interface Props {
  debug: RouteDebug | null
  loading: boolean
}

export function DebuggerPanel({ debug, loading }: Props) {
  return (
    <aside className="w-72 border-l border-[#e4e1da] bg-[#fafaf8] flex flex-col flex-shrink-0 overflow-y-auto">
      <div className="px-4 py-3 border-b border-[#e4e1da]">
        <h3 className="text-xs font-semibold text-[#18181b] uppercase tracking-wider">Framework Debugger</h3>
      </div>

      {loading && (
        <div className="px-4 py-3 text-xs text-[#6b7280] animate-pulse">Routing query…</div>
      )}

      {!debug && !loading && (
        <div className="px-4 py-6 text-xs text-[#9ca3af] text-center">
          Send a message to see routing debug info
        </div>
      )}

      {debug && (
        <div className="px-4 py-3 space-y-4 text-xs">
          {/* Route summary */}
          <section>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-2">Route Summary</p>
            <div className="space-y-1">
              <Row label="Domain" value={debug.domain} accent />
              <Row label="Mode" value={debug.routing_mode} />
              <Row label="U Score" value={debug.u_score.toFixed(4)} />
              <Row label="Latency" value={`${debug.latency_ms.toFixed(0)}ms`} />
              <Row label="Contradictions" value={String(debug.contradictions_detected)} />
            </div>
          </section>

          {/* Domain distribution */}
          {Object.keys(debug.domain_distribution).length > 0 && (
            <section>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-2">Classifier Output</p>
              <div className="space-y-1.5">
                {Object.entries(debug.domain_distribution)
                  .sort(([, a], [, b]) => b - a)
                  .map(([domain, prob]) => (
                    <div key={domain}>
                      <div className="flex justify-between mb-0.5">
                        <span className="text-[#374151] truncate">{domain}</span>
                        <span className="text-[#6b7280] font-mono">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-[#e4e1da] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#4338ca] rounded-full transition-all"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </section>
          )}

          {/* Utility breakdown */}
          <section>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-2">Utility Breakdown</p>
            <div className="bg-[#f0ede6] rounded-lg p-2.5 font-mono text-[10px] text-[#374151]">
              U = w_e·E + w_c·C + w_k·K<br />
              <span className="text-[#4338ca]">= {debug.u_score.toFixed(4)}</span>
            </div>
          </section>

          {/* Specialist responses */}
          {debug.specialist_responses && Object.keys(debug.specialist_responses).length > 0 && (
            <section>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[#6b7280] mb-2">Specialist Calls</p>
              {Object.entries(debug.specialist_responses).map(([name, resp]) => (
                <div key={name} className="mb-2">
                  <p className="text-[#374151] font-medium mb-1">{name}</p>
                  <p className="text-[#6b7280] line-clamp-3 bg-[#f0ede6] rounded p-1.5 leading-relaxed">
                    {String(resp).slice(0, 150)}{String(resp).length > 150 ? '…' : ''}
                  </p>
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </aside>
  )
}

function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[#6b7280]">{label}</span>
      <span className={`font-mono ${accent ? 'text-[#4338ca] font-semibold' : 'text-[#374151]'}`}>{value}</span>
    </div>
  )
}
