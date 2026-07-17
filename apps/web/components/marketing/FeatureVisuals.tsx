/** Mini product visuals for feature cards — light theme matching landing page. */

import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { HEATMAP_LIGHT_SOFT, mock } from '@/components/marketing/mock-ui'
import { Flame, LayoutGrid, ScanEye } from 'lucide-react'

export function NeuralScoringVisual() {
  return (
    <div className={`relative h-full w-full ${mock.panel} p-4`}>
      <svg viewBox="0 0 280 140" className="h-full w-full" aria-hidden="true">
        <defs>
          <linearGradient id="engGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1D4ED8" stopOpacity="0.25" />
            <stop offset="50%" stopColor="#E85D24" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#1D4ED8" stopOpacity="0.25" />
          </linearGradient>
        </defs>
        <text x="8" y="14" fill="#8A8A8A" fontSize="8" fontFamily="monospace">
          VAN−DMN Z · Track 1
        </text>
        <path
          d="M10 90 L50 55 L90 75 L130 40 L170 80 L210 50 L250 65 L270 45"
          fill="none"
          stroke="url(#engGrad)"
          strokeWidth="2.5"
        />
        <circle cx="130" cy="40" r="5" fill="#E85D24" />
        <text x="118" y="32" fill="#E85D24" fontSize="7" fontFamily="monospace">
          peak
        </text>
        <rect x="10" y="105" width="60" height="24" rx="6" fill="#FFFFFF" stroke="#E8E6DF" />
        <text x="16" y="120" fill="#1D4ED8" fontSize="9" fontFamily="monospace">
          Neural 84
        </text>
      </svg>
    </div>
  )
}

export function CrawlVisual() {
  const pages = [
    { path: '/', trs: 14, heat: 0.7 },
    { path: '/features', trs: 11, heat: 0.5 },
    { path: '/pricing', trs: 9, heat: 0.85 },
  ]
  return (
    <div className={`flex h-full flex-col justify-center gap-2 ${mock.panel} p-4`}>
      {pages.map((p) => (
        <div key={p.path} className={mock.row}>
          <PageRouteIcon path={p.path} />
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-[10px] text-signal">{p.path}</p>
            <div className={`mt-1 ${mock.metricBar}`}>
              <div
                className="h-full rounded-full bg-copy"
                style={{ width: `${p.heat * 100}%` }}
              />
            </div>
          </div>
          <span className="font-mono text-[9px] text-text-tertiary">{p.trs} TR</span>
        </div>
      ))}
    </div>
  )
}

export function CopyVisual() {
  return (
    <div className={`relative h-full ${mock.panel} p-4`}>
      <div className={`rounded-lg border border-line bg-surface p-3 shadow-card`}>
        <p className="text-[10px] leading-relaxed text-text-primary">
          <span className="rounded bg-signal/10 px-1 text-signal">AI that thinks</span>{' '}
          with you — clarity{' '}
          <span className="rounded bg-copy/10 px-1 text-copy">88</span>
        </p>
        <p className="mt-2 text-[9px] text-text-secondary">
          Subscribe today · urgency{' '}
          <span className="font-mono text-neural">72</span>
        </p>
        <div className="mt-3 flex flex-wrap gap-1">
          {['clarity 88', 'urgency 72', 'goal fit 81'].map((c) => (
            <span
              key={c}
              className="rounded-full border border-copy/30 bg-copy/10 px-2 py-0.5 font-mono text-[8px] text-copy"
            >
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="absolute bottom-3 right-3 rounded-full border border-signal/20 bg-surface px-2 py-1 font-mono text-[9px] text-signal shadow-card">
        copy fused · 20%
      </div>
    </div>
  )
}

export function HeatmapVisual() {
  return (
    <div className={`relative h-full overflow-hidden ${mock.panel}`}>
      <div className="absolute inset-3 space-y-2">
        <div className="flex items-center gap-2">
          <ScanEye size={12} className="text-signal" aria-hidden />
          <div className={`h-2 flex-1 ${mock.skeletonMid}`} />
        </div>
        <div className={`h-2 w-2/3 ${mock.skeleton}`} />
        <div className="mx-auto mt-3 flex h-6 w-28 items-center justify-center rounded-lg border border-line bg-surface shadow-card">
          <Flame size={12} className="text-neural" aria-hidden />
        </div>
        <div className="mt-4 grid grid-cols-3 gap-1">
          <MiniFeatureTile icon={LayoutGrid} label="Grid" accent="signal" className="h-10" />
          <MiniFeatureTile icon={Flame} label="Hot" accent="neural" className="h-10" />
          <MiniFeatureTile icon={ScanEye} label="Focus" accent="copy" className="h-10" />
        </div>
      </div>
      <div
        className="absolute inset-0"
        style={{ background: HEATMAP_LIGHT_SOFT }}
      />
      <div className="absolute bottom-2 left-2 flex gap-2 font-mono text-[7px] text-text-tertiary">
        <span className="text-neural">■ high</span>
        <span className="text-amber-600">■ peak</span>
        <span className="text-signal">■ low</span>
      </div>
      <span className="absolute top-2 right-2 rounded-full border border-line bg-surface px-2 py-0.5 font-mono text-[8px] text-text-tertiary shadow-card">
        DINOv2
      </span>
    </div>
  )
}

export type FeatureVisualType = 'neural' | 'crawl' | 'copy' | 'heatmap'

export function FeatureVisual({ type }: { type: FeatureVisualType }) {
  const map = {
    neural: NeuralScoringVisual,
    crawl: CrawlVisual,
    copy: CopyVisual,
    heatmap: HeatmapVisual,
  }
  const Comp = map[type]
  return (
    <div className="aspect-[16/10] w-full overflow-hidden bg-surface-muted">
      <Comp />
    </div>
  )
}
