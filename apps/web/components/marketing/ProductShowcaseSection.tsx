import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { ProductViewerMock } from '@/components/marketing/ProductViewerMock'

const showcases = [
  {
    step: '01',
    title: 'Crawl & capture',
    body: 'Playwright scrolls every page, records video, and snapshots DOM text per frame.',
    visual: 'crawl' as const,
  },
  {
    step: '02',
    title: 'Neural inference',
    body: 'TRIBE v2 predicts cortical response from walkthrough video. DINOv2 builds attention heatmaps.',
    visual: 'neural' as const,
  },
  {
    step: '03',
    title: 'Actionable viewer',
    body: 'Click any element for combined neural + copy scores. Scrub the timeline to see engagement peaks.',
    visual: 'viewer' as const,
  },
]

export function ProductShowcaseSection() {
  return (
    <section className="border-t border-ink-700 bg-ink-900 px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <p className="mb-3 text-xs font-mono uppercase tracking-widest text-signal" data-reveal>
          The product
        </p>
        <h2
          className="mb-4 max-w-2xl text-3xl font-display font-semibold tracking-tight text-text-primary md:text-4xl"
          data-reveal
          data-reveal-delay="0.05"
        >
          See what your users&apos; brains respond to.
        </h2>
        <p className="mb-14 max-w-xl text-text-secondary" data-reveal data-reveal-delay="0.1">
          Every scan produces a full session viewer — heatmaps on real frames, element scores,
          cortical activation, and copy insights in one surface.
        </p>

        <div className="grid gap-8 lg:grid-cols-3">
          {showcases.map((item, i) => (
            <article
              key={item.step}
              className="flex flex-col"
              data-reveal
              data-reveal-delay={String(0.12 + i * 0.08)}
            >
              <div className="mb-4 overflow-hidden rounded-lg border border-ink-700 bg-ink-950 shadow-lg">
                <ShowcaseVisual type={item.visual} />
              </div>
              <span className="mb-1 font-mono text-xs text-ink-600">{item.step}</span>
              <h3 className="mb-2 text-lg font-semibold text-text-primary">{item.title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{item.body}</p>
            </article>
          ))}
        </div>

        {/* Full-width hero product shot */}
        <div className="mt-16" data-reveal data-reveal-delay="0.2">
          <p className="mb-4 text-center text-xs font-mono uppercase tracking-widest text-text-secondary">
            Full session viewer
          </p>
          <ProductViewerMock className="shadow-2xl shadow-signal/10" />
        </div>
      </div>
    </section>
  )
}

function ShowcaseVisual({ type }: { type: 'crawl' | 'neural' | 'viewer' }) {
  if (type === 'crawl') {
    return (
      <div className="aspect-[4/3] p-4">
        <div className="flex h-full flex-col gap-2">
          {['/', '/features', '/pricing', '/about'].map((path, i) => (
            <div
              key={path}
              className="flex flex-1 items-center gap-3 rounded border border-ink-700 bg-ink-900 px-3"
            >
              <PageRouteIcon path={path} />
              <div className="flex-1">
                <p className="font-mono text-[10px] text-signal">{path}</p>
                <div className="mt-1 h-1 w-full rounded-full bg-ink-700">
                  <div className="h-full rounded-full bg-copy" style={{ width: `${60 + i * 10}%` }} />
                </div>
              </div>
              <span className="font-mono text-[9px] text-text-secondary">{12 + i * 4} TRs</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (type === 'neural') {
    return (
      <div className="relative aspect-[4/3] bg-ink-950 p-4">
        <svg viewBox="0 0 280 200" className="h-full w-full" aria-hidden="true">
          <defs>
            <radialGradient id="brainGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ff6b35" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#08090b" stopOpacity="0" />
            </radialGradient>
          </defs>
          <ellipse cx="140" cy="100" rx="90" ry="55" fill="url(#brainGlow)" />
          <ellipse cx="140" cy="100" rx="75" ry="45" fill="#12151c" stroke="#2563ff" strokeWidth="1" opacity="0.5" />
          {Array.from({ length: 12 }).map((_, i) => {
            const angle = (i / 12) * Math.PI * 2
            const x = 140 + Math.cos(angle) * 55
            const y = 100 + Math.sin(angle) * 32
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r={3 + (i % 3)}
                fill={i % 3 === 0 ? '#ff6b35' : '#2563ff'}
                opacity={0.5 + (i % 4) * 0.12}
              />
            )
          })}
          <path
            d="M80 110 Q110 60 140 90 Q170 120 200 80"
            fill="none"
            stroke="#ff6b35"
            strokeWidth="2.5"
          />
        </svg>
        <div className="absolute bottom-3 left-3 rounded border border-neural/40 bg-ink-900/90 px-2 py-1 font-mono text-[9px] text-neural">
          VAN spike · Z=2.4
        </div>
      </div>
    )
  }

  return (
    <div className="scale-[0.92] origin-top p-1">
      <ProductViewerMock />
    </div>
  )
}
