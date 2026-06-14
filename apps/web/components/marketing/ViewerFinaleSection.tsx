'use client'

import { useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { ProductViewerMock } from '@/components/marketing/ProductViewerMock'

const tabs = [
  { id: 'heatmap', label: 'Heatmap', sub: 'DINOv2 per frame', accent: 'from-yellow-500/20 to-orange-600/30' },
  { id: 'elements', label: 'Elements', sub: 'Click any bbox', accent: 'from-signal/20 to-signal/5' },
  { id: 'brain', label: 'Brain strip', sub: 'TRIBE v2 cortical', accent: 'from-neural/25 to-neural/5' },
  { id: 'copy', label: 'Copy chips', sub: 'Clarity · urgency · fit', accent: 'from-copy/25 to-copy/5' },
] as const

type TabId = (typeof tabs)[number]['id']

export function ViewerFinaleSection() {
  const [active, setActive] = useState<TabId>('heatmap')
  const overlayRef = useRef<HTMLDivElement>(null)
  const viewerWrapRef = useRef<HTMLDivElement>(null)

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
        { opacity: 0.4 },
        { opacity: 1, duration: 0.35, ease: 'power2.out' }
      )
    },
    { dependencies: [active], scope: overlayRef }
  )

  return (
    <section id="viewer" className="border-t border-ink-700 bg-ink-950 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="mb-4 text-xs font-mono uppercase tracking-widest text-text-secondary" data-reveal>
          Full session viewer
        </p>
        <h2
          className="mb-4 text-3xl font-display font-semibold tracking-tight text-text-primary md:text-4xl"
          style={{ letterSpacing: '-0.02em' }}
          data-reveal
          data-reveal-delay="0.05"
        >
          One surface. Every signal.
        </h2>
        <p className="mb-8 max-w-2xl text-text-secondary" data-reveal data-reveal-delay="0.08">
          The only full viewer on this page — switch layers below to see what each mode highlights
          in your team&apos;s post-scan report.
        </p>

        {/* Interactive layer tabs */}
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
              className={`rounded-md border px-4 py-2 text-left transition-colors ${
                active === tab.id
                  ? 'border-signal bg-signal/10 text-text-primary'
                  : 'border-ink-700 bg-ink-900 text-text-secondary hover:border-ink-600'
              } focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal`}
            >
              <span className="block text-sm font-medium">{tab.label}</span>
              <span className="font-mono text-[10px] opacity-70">{tab.sub}</span>
            </button>
          ))}
        </div>

        <div ref={viewerWrapRef} className="relative" data-reveal data-reveal-delay="0.14">
          <ProductViewerMock className="shadow-2xl shadow-black/30 ring-1 ring-ink-700" />

          {/* Layer emphasis overlay — changes per tab */}
          <div
            ref={overlayRef}
            className={`pointer-events-none absolute inset-0 rounded-lg bg-gradient-to-br ${tabs.find((t) => t.id === active)?.accent} transition-colors duration-300`}
            aria-hidden="true"
          />

          {/* Corner callout per tab */}
          <div className="pointer-events-none absolute bottom-4 right-4 rounded-lg border border-ink-600 bg-ink-900/90 px-4 py-3 backdrop-blur-sm">
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
      <p className="mt-0.5 font-mono text-[10px] text-text-secondary">{copy.detail}</p>
    </>
  )
}
