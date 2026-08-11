import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { HEATMAP_LIGHT, mock } from '@/components/marketing/mock-ui'
import { Image, MousePointer2, Type } from 'lucide-react'
import { SITE } from '@/lib/site'

export function CrawlPhaseVisual({ className = '' }: { className?: string }) {
  return (
    <div className={`${mock.shell} ${className}`}>
      <div className={`border-b ${mock.divider} px-4 py-2 ${mock.label}`}>
        {SITE.name.toLowerCase()} · crawling same-origin pages
      </div>
      <div className="relative p-4" data-crawl-pages>
        <div className="space-y-2">
          {['/', '/features', '/pricing', '/about'].map((path, i) => (
            <div
              key={path}
              data-crawl-row
              className={`${mock.row} opacity-0`}
              style={{ transform: 'translateX(-12px)' }}
            >
              <PageRouteIcon path={path} />
              <div className="flex-1">
                <p className="font-mono text-[10px] text-signal">{path}</p>
                <div className={`mt-1 ${mock.metricBar}`}>
                  <div
                    data-crawl-bar
                    data-crawl-width={`${55 + i * 12}%`}
                    className="h-full w-0 rounded-full bg-copy"
                  />
                </div>
              </div>
              <span className="font-mono text-[9px] text-text-tertiary">{8 + i * 3} TR</span>
            </div>
          ))}
        </div>
        <div
          data-crawl-beam
          className="pointer-events-none absolute left-0 right-0 top-4 h-px bg-gradient-to-r from-transparent via-signal/50 to-transparent opacity-0"
        />
      </div>
    </div>
  )
}

export function NeuralPhaseVisual({ className = '' }: { className?: string }) {
  return (
    <div className={`${mock.shell} ${className}`}>
      <div className={`border-b ${mock.divider} px-4 py-2 ${mock.label}`}>
        TRIBE v2 · cortical inference
      </div>
      <div className={`grid grid-cols-2 gap-px bg-line`}>
        <div className={`relative ${mock.panel} p-4`}>
          <svg viewBox="0 0 160 120" className="h-full w-full" aria-hidden="true">
            <ellipse
              cx="80"
              cy="60"
              rx="55"
              ry="34"
              fill="#F5F4EF"
              stroke="#1D4ED8"
              strokeWidth="0.8"
              opacity="0.6"
            />
            <path
              data-neural-path
              d="M30 70 Q55 35 80 55 Q105 75 130 45"
              fill="none"
              stroke="#E85D24"
              strokeWidth="2.5"
              strokeDasharray="200"
              strokeDashoffset="200"
            />
            <circle data-neural-dot cx="80" cy="55" r="5" fill="#E85D24" opacity="0" />
          </svg>
          <span className="absolute bottom-2 left-2 rounded-full border border-neural/30 bg-surface px-2 py-0.5 font-mono text-[8px] text-neural shadow-card">
            VAN spike
          </span>
        </div>
        <div className={`${mock.panelInner}`}>
          <p className={`mb-2 ${mock.label}`}>Engagement track</p>
          <svg viewBox="0 0 160 80" className="w-full" aria-hidden="true">
            <path
              data-wave-path
              d="M0 50 L20 35 L40 45 L60 25 L80 55 L100 30 L120 48 L140 20 L160 40"
              fill="none"
              stroke="#1D4ED8"
              strokeWidth="2"
              strokeDasharray="220"
              strokeDashoffset="220"
            />
          </svg>
          <div className="mt-3 grid grid-cols-2 gap-2">
            {[
              { k: 'Neural', v: '84', c: 'text-signal' },
              { k: 'Peak TR', v: '14', c: 'text-neural' },
            ].map((m) => (
              <div key={m.k} className={`${mock.rowMuted} text-center`}>
                <p className="text-[8px] text-text-tertiary">{m.k}</p>
                <p className={`font-mono text-sm font-semibold ${m.c}`}>{m.v}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function HeatmapPhaseVisual({ className = '' }: { className?: string }) {
  return (
    <div className={`relative ${mock.shell} ${className}`}>
      <div className={`border-b ${mock.divider} px-4 py-2 ${mock.label}`}>
        DeepGaze · attention heatmap
      </div>
      <div className={`relative aspect-[16/10] ${mock.panel} p-6`}>
        <div className="space-y-3">
          <div className={`h-3 w-2/5 ${mock.skeletonMid}`} />
          <div className={`h-2 w-3/5 ${mock.skeleton}`} />
          <div className={`mx-auto mt-4 h-8 w-28 rounded-lg ${mock.skeletonBtn}`} />
          <div className="mt-6 grid grid-cols-3 gap-2">
            <MiniFeatureTile icon={Type} label="Headline" accent="neural" />
            <MiniFeatureTile icon={MousePointer2} label="CTA" accent="signal" />
            <MiniFeatureTile icon={Image} label="Media" accent="copy" />
          </div>
        </div>
        <div
          data-heatmap-layer
          className="absolute inset-0 opacity-0"
          style={{ background: HEATMAP_LIGHT }}
        />
        <div
          data-heatmap-sweep
          className="pointer-events-none absolute inset-y-0 left-0 w-16 opacity-0"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(29,78,216,0.15), transparent)',
          }}
        />
      </div>
    </div>
  )
}

export function InspectPhaseVisual({ className = '' }: { className?: string }) {
  return (
    <div className={`${mock.shell} ${className}`}>
      <div className={`flex items-center gap-2 border-b ${mock.divider} ${mock.chrome}`}>
        <span className={mock.label}>element inspector</span>
        <span className={`ml-auto ${mock.badge}`}>hero-h1 · 91</span>
      </div>
      <div className={`relative aspect-[16/10] ${mock.panel} p-4`}>
        <div className="mx-auto max-w-[80%] space-y-2 text-center">
          <div
            data-inspect-target
            className="relative mx-auto rounded-lg border-2 border-neural/50 bg-neural/10 px-3 py-2"
          >
            <div className={`h-3 w-full ${mock.skeletonMid}`} />
            <span
              data-inspect-label
              className="absolute -top-4 left-0 rounded-full bg-neural px-2 py-0.5 font-mono text-[8px] text-white opacity-0"
            >
              H1 · 91
            </span>
          </div>
          <div className={`mx-auto h-2 w-1/2 ${mock.skeleton}`} />
          <div className="mx-auto mt-3 h-7 w-24 rounded-lg bg-ink-950/90" />
        </div>
        <div
          data-inspect-bbox
          className="absolute rounded-sm border-2 border-signal/50 bg-signal/10 opacity-0"
          style={{ top: '18%', left: '22%', width: '56%', height: '14%' }}
        />
        <div
          data-inspect-bbox
          className="absolute rounded-sm border-2 border-copy/50 bg-copy/10 opacity-0"
          style={{ top: '48%', left: '38%', width: '24%', height: '10%' }}
        />
      </div>
      <div className={`flex gap-2 border-t ${mock.divider} bg-surface-muted p-3`}>
        {['clarity 88', 'urgency 72', 'goal fit 81'].map((chip, i) => (
          <span
            key={chip}
            data-inspect-chip
            className="rounded-full border border-copy/30 bg-copy/10 px-2 py-1 font-mono text-[8px] text-copy opacity-0"
            style={{ transitionDelay: `${i * 40}ms` }}
          >
            {chip}
          </span>
        ))}
      </div>
    </div>
  )
}

export function ReportReadyVisual() {
  return (
    <div className={`flex aspect-[16/10] flex-col justify-center gap-3 ${mock.panel} p-5`}>
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-copy" />
        <span className="font-mono text-[10px] text-copy">Scan complete · 48 TRs</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Neural', value: '84', color: 'text-signal' },
          { label: 'Attention', value: '91', color: 'text-neural' },
          { label: 'Copy', value: '76', color: 'text-copy' },
        ].map((s) => (
          <div key={s.label} className={`${mock.rowMuted} text-center shadow-card`}>
            <p className="text-[9px] text-text-tertiary">{s.label}</p>
            <p className={`font-mono text-xl font-semibold ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-signal/20 bg-signal/5 px-3 py-2 text-center font-mono text-[10px] text-signal">
        Open interactive viewer →
      </div>
    </div>
  )
}
