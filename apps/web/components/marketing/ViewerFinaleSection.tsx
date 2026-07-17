'use client'

import { useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { ProductViewerMock } from '@/components/marketing/ProductViewerMock'

const tabs = [
  {
    id: 'heatmap',
    label: 'Heatmap',
    sub: 'DINOv2 per frame',
    accent: 'from-amber-200/30 to-orange-300/20',
  },
  {
    id: 'elements',
    label: 'Elements',
    sub: 'Click any bbox',
    accent: 'from-blue-200/30 to-signal/10',
  },
  {
    id: 'brain',
    label: 'Brain strip',
    sub: 'TRIBE v2 cortical',
    accent: 'from-orange-200/30 to-neural/10',
  },
  {
    id: 'copy',
    label: 'Copy chips',
    sub: 'Clarity · urgency · fit',
    accent: 'from-green-200/30 to-copy/10',
  },
] as const

type TabId = (typeof tabs)[number]['id']

export function ViewerFinaleSection() {
  const [active, setActive] = useState<TabId>('heatmap')
  const overlayRef = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      if (!overlayRef.current) return
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (reduced) {
        gsap.set(overlayRef.current, { opacity: 1 })
        return
      }
      gsap.fromTo(
        overlayRef.current,
        { opacity: 0.5 },
        { opacity: 1, duration: 0.5, ease: 'power3.out' }
      )
    },
    { dependencies: [active], scope: overlayRef }
  )

  return (
    <section id="viewer" data-section className="border-t border-line bg-canvas-warm px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="section-eyebrow" data-reveal>
          Interactive viewer
        </p>
        <h2 className="section-heading mb-4" data-reveal data-reveal-delay="0.05">
          Explore every signal in one report.
        </h2>
        <p
          className="mb-8 max-w-2xl text-base leading-relaxed text-text-secondary"
          data-reveal
          data-reveal-delay="0.08"
        >
          After a scan completes, your team gets a shareable viewer with timeline scrubbing,
          heatmap overlays, element bboxes, and copy fusion chips — switch layers below to
          preview each mode.
        </p>

        <div
          className="mb-6 flex flex-wrap gap-2"
          data-reveal
          data-reveal-delay="0.1"
          role="tablist"
          aria-label="Viewer layers"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active === tab.id}
              onClick={() => setActive(tab.id)}
              className={`cursor-pointer rounded-pill border px-4 py-2 text-left transition-all duration-300 ease-premium ${
                active === tab.id
                  ? 'border-ink-950 bg-ink-950 text-white shadow-card'
                  : 'border-line bg-surface text-text-secondary hover:border-line-strong hover:text-text-primary'
              } focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink-950`}
            >
              <span className="block text-sm font-medium">{tab.label}</span>
              <span className="text-[10px] opacity-70">{tab.sub}</span>
            </button>
          ))}
        </div>

        <div className="relative" data-reveal data-reveal-delay="0.14">
          <div className="product-mock-shell">
            <ProductViewerMock />
          </div>

          <div
            ref={overlayRef}
            className={`pointer-events-none absolute inset-0 rounded-card-lg bg-gradient-to-br ${tabs.find((t) => t.id === active)?.accent} transition-all duration-500 ease-premium`}
            aria-hidden="true"
          />

          <div className="pointer-events-none absolute bottom-6 right-6 rounded-card border border-line bg-surface/95 px-4 py-3 shadow-card-lg backdrop-blur-sm">
            <TabCallout tab={active} />
          </div>
        </div>
      </div>
    </section>
  )
}

function TabCallout({ tab }: { tab: TabId }) {
  const copy = {
    heatmap: { title: 'Attention heatmap', detail: 'Warm regions = peak DINOv2 saliency' },
    elements: { title: 'Element bboxes', detail: '91 engaging · 45 low attention' },
    brain: { title: 'Cortical strip', detail: 'VAN pathway spike at TR 14' },
    copy: { title: 'Copy fusion', detail: 'clarity 88 · urgency 72 · goal fit 81' },
  }[tab]

  return (
    <>
      <p className="text-xs font-medium text-text-primary">{copy.title}</p>
      <p className="mt-0.5 text-[10px] text-text-secondary">{copy.detail}</p>
    </>
  )
}
