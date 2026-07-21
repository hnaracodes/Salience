'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { HeroLiveScanVisual } from '@/components/marketing/HeroLiveScanVisual'

/** Hero visual — live scan animation with floating stat cards. */
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
        { opacity: 0, y: 40, scale: 0.97 },
        { opacity: 1, y: 0, scale: 1, duration: 1.2, ease: 'power3.out', delay: 0.35 }
      )

      const floatCards = panelRef.current.querySelectorAll('[data-float-card]')
      floatCards.forEach((card, i) => {
        gsap.to(card, {
          y: i % 2 === 0 ? -6 : 6,
          duration: 4 + i * 0.5,
          ease: 'sine.inOut',
          yoyo: true,
          repeat: -1,
          delay: i * 0.3,
        })
      })
    },
    { scope: panelRef }
  )

  return (
    <div ref={panelRef} className="relative w-full max-w-2xl">
      <div
        data-float-card
        className="absolute -left-2 top-8 z-20 hidden rounded-card border border-line bg-surface px-4 py-3 shadow-card sm:block"
      >
        <p className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">Pages</p>
        <p className="text-2xl font-semibold text-signal">3/8</p>
      </div>
      <div
        data-float-card
        className="absolute -right-1 top-1/2 z-20 hidden rounded-card border border-line bg-surface px-4 py-3 shadow-card-lg sm:block"
      >
        <p className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">Peak TR</p>
        <p className="text-2xl font-semibold text-neural">14</p>
      </div>
      <div
        data-float-card
        className="absolute -bottom-1 left-1/4 z-20 hidden rounded-card border border-line bg-surface px-4 py-3 shadow-card sm:block"
      >
        <p className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">Status</p>
        <p className="text-lg font-semibold text-copy">crawling</p>
      </div>

      <div className="product-mock-shell animate-float-gentle">
        <HeroLiveScanVisual />
      </div>

      <p className="mt-5 text-center text-xs text-text-tertiary">
        Live scan in progress — scroll to walk through the full pipeline
      </p>
    </div>
  )
}
