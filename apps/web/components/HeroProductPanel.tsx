'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { HeroLiveScanVisual } from '@/components/marketing/HeroLiveScanVisual'

/** Hero visual — live scan animation, not the full viewer. */
export function HeroProductPanel() {
  const panelRef = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (!panelRef.current) return
      if (reduced) {
        gsap.set(panelRef.current, { opacity: 1, y: 0 })
        return
      }
      gsap.fromTo(
        panelRef.current,
        { opacity: 0, y: 28 },
        { opacity: 1, y: 0, duration: 1, ease: 'power2.out', delay: 0.2 }
      )
    },
    { scope: panelRef }
  )

  return (
    <div ref={panelRef} className="relative w-full max-w-2xl">
      <div className="absolute -left-3 top-6 z-20 hidden rounded-lg border border-signal/40 bg-ink-900/95 px-3 py-2 shadow-xl sm:block">
        <p className="text-[9px] font-mono uppercase tracking-widest text-text-secondary">Pages</p>
        <p className="text-2xl font-semibold text-signal">3/8</p>
      </div>
      <div className="absolute -right-2 top-1/2 z-20 hidden rounded-lg border border-neural/40 bg-ink-900/95 px-3 py-2 shadow-xl sm:block">
        <p className="text-[9px] font-mono uppercase tracking-widest text-text-secondary">Current TR</p>
        <p className="text-2xl font-semibold text-neural">14</p>
      </div>
      <div className="absolute -bottom-2 left-1/4 z-20 hidden rounded-lg border border-copy/40 bg-ink-900/95 px-3 py-2 shadow-xl sm:block">
        <p className="text-[9px] font-mono uppercase tracking-widest text-text-secondary">Mode</p>
        <p className="text-lg font-semibold text-copy">crawling</p>
      </div>

      <HeroLiveScanVisual />

      <p className="mt-4 text-center text-xs font-mono text-text-secondary">
        Live scan in progress — scroll down to see the full pipeline
      </p>
    </div>
  )
}
