'use client'

import { ProductViewerMock } from '@/components/marketing/ProductViewerMock'

export function DemoSection() {
  return (
    <section className="border-t border-ink-700 bg-ink-950 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="mb-4 text-xs font-mono uppercase tracking-widest text-text-secondary" data-reveal>
          The viewer
        </p>
        <h2
          className="mb-4 text-3xl font-display font-semibold tracking-tight text-text-primary md:text-4xl"
          style={{ letterSpacing: '-0.02em' }}
          data-reveal
          data-reveal-delay="0.05"
        >
          Every pixel annotated.
        </h2>
        <p className="mb-10 max-w-2xl text-text-secondary" data-reveal data-reveal-delay="0.08">
          Scrub the timeline, click any highlighted element, and read neural + copy scores
          alongside cortical activation — the same viewer your team gets after every scan.
        </p>

        <div data-reveal data-reveal-delay="0.12">
          <ProductViewerMock className="shadow-2xl shadow-black/30 ring-1 ring-ink-700" />
        </div>

        {/* Capability strip below viewer */}
        <div
          className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-ink-700 bg-ink-700 md:grid-cols-4"
          data-reveal
          data-reveal-delay="0.18"
        >
          {[
            { label: 'Heatmap overlay', sub: 'DINOv2 per frame' },
            { label: 'Element scores', sub: 'Click any bbox' },
            { label: 'Brain strip', sub: 'TRIBE v2 cortical' },
            { label: 'Copy chips', sub: 'Clarity · urgency · fit' },
          ].map((item) => (
            <div key={item.label} className="bg-ink-900 px-4 py-4 text-center">
              <p className="text-sm font-medium text-text-primary">{item.label}</p>
              <p className="mt-0.5 font-mono text-[10px] text-text-secondary">{item.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
