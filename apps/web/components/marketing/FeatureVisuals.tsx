/** Mini product visuals for feature cards — each shows a real product surface. */

import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { Flame, LayoutGrid, ScanEye } from 'lucide-react'

export function NeuralScoringVisual() {
  return (
    <div className="relative h-full w-full bg-ink-950 p-4">
      <svg viewBox="0 0 280 140" className="h-full w-full" aria-hidden="true">
        <defs>
          <linearGradient id="engGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2563ff" stopOpacity="0.2" />
            <stop offset="50%" stopColor="#ff6b35" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#2563ff" stopOpacity="0.2" />
          </linearGradient>
        </defs>
        <text x="8" y="14" fill="#8b95a8" fontSize="8" fontFamily="monospace">VAN−DMN Z · Track 1</text>
        <path
          d="M10 90 L50 55 L90 75 L130 40 L170 80 L210 50 L250 65 L270 45"
          fill="none"
          stroke="url(#engGrad)"
          strokeWidth="2.5"
        />
        <circle cx="130" cy="40" r="5" fill="#ff6b35" />
        <text x="118" y="32" fill="#ff6b35" fontSize="7" fontFamily="monospace">peak</text>
        <rect x="10" y="105" width="60" height="24" rx="3" fill="#0d0f12" stroke="#1c2028" />
        <text x="16" y="120" fill="#2563ff" fontSize="9" fontFamily="monospace">Neural 84</text>
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
    <div className="flex h-full flex-col justify-center gap-2 bg-ink-950 p-4">
      {pages.map((p) => (
        <div key={p.path} className="flex items-center gap-3 rounded border border-ink-700 bg-ink-900 px-3 py-2">
          <PageRouteIcon path={p.path} />
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-[10px] text-signal">{p.path}</p>
            <div className="mt-1 h-1 rounded-full bg-ink-700">
              <div className="h-full rounded-full bg-copy" style={{ width: `${p.heat * 100}%` }} />
            </div>
          </div>
          <span className="font-mono text-[9px] text-text-secondary">{p.trs} TR</span>
        </div>
      ))}
    </div>
  )
}

export function CopyVisual() {
  return (
    <div className="relative h-full bg-ink-950 p-4">
      <div className="rounded border border-ink-700 bg-ink-900 p-3">
        <p className="text-[10px] leading-relaxed text-text-primary">
          <span className="rounded bg-signal/20 px-1 text-signal">AI that thinks</span>{' '}
          with you — clarity{' '}
          <span className="rounded bg-copy/20 px-1 text-copy">88</span>
        </p>
        <p className="mt-2 text-[9px] text-text-secondary">
          Subscribe today · urgency{' '}
          <span className="text-neural font-mono">72</span>
        </p>
        <div className="mt-3 flex flex-wrap gap-1">
          {['clarity 88', 'urgency 72', 'goal fit 81'].map((c) => (
            <span
              key={c}
              className="rounded border border-copy/30 bg-copy/10 px-1.5 py-0.5 font-mono text-[8px] text-copy"
            >
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="absolute bottom-3 right-3 rounded border border-signal/30 bg-ink-900/90 px-2 py-1 font-mono text-[9px] text-signal">
        copy fused · 20%
      </div>
    </div>
  )
}

export function HeatmapVisual() {
  return (
    <div className="relative h-full overflow-hidden bg-[#0f1117]">
      <div className="absolute inset-3 space-y-2">
        <div className="flex items-center gap-2">
          <ScanEye size={12} className="text-signal" aria-hidden />
          <div className="h-2 flex-1 rounded-sm bg-white/15" />
        </div>
        <div className="h-2 w-2/3 rounded-sm bg-white/8" />
        <div className="mx-auto mt-3 flex h-6 w-28 items-center justify-center rounded-md border border-signal/30 bg-signal/10">
          <Flame size={12} className="text-neural" aria-hidden />
        </div>
        <div className="mt-4 grid grid-cols-3 gap-1">
          <MiniFeatureTile icon={LayoutGrid} label="Grid" accent="signal" className="h-10" />
          <MiniFeatureTile icon={Flame} label="Hot" accent="neural" className="h-10" />
          <MiniFeatureTile icon={ScanEye} label="Focus" accent="copy" className="h-10" />
        </div>
      </div>
      <div
        className="absolute inset-0 mix-blend-screen"
        style={{
          background: `
            radial-gradient(ellipse 55% 40% at 45% 30%, rgba(255,180,50,0.7) 0%, transparent 55%),
            radial-gradient(ellipse 40% 35% at 70% 60%, rgba(255,80,30,0.5) 0%, transparent 50%),
            radial-gradient(ellipse 30% 25% at 20% 75%, rgba(37,99,255,0.35) 0%, transparent 55%)
          `,
        }}
      />
      <div className="absolute bottom-2 left-2 flex gap-2 font-mono text-[7px]">
        <span className="text-yellow-400">■ high</span>
        <span className="text-orange-500">■ peak</span>
        <span className="text-blue-400">■ low</span>
      </div>
      <span className="absolute top-2 right-2 rounded bg-ink-900/80 px-1.5 py-0.5 font-mono text-[8px] text-text-secondary">
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
    <div className="aspect-[16/10] w-full overflow-hidden border-b border-ink-700">
      <Comp />
    </div>
  )
}
