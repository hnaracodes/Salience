'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { Globe, Layers, TrendingUp } from 'lucide-react'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { Logo } from '@/components/Logo'
import { SITE } from '@/lib/site'

/** Hero visual — live scan in progress (distinct from viewer mock). */
export function HeroLiveScanVisual({ className = '' }: { className?: string }) {
  const rootRef = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      const root = rootRef.current
      if (!root) return

      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      const beam = root.querySelector('[data-hero-beam]')
      const reticle = root.querySelector('[data-hero-reticle]')
      const pulses = gsap.utils.toArray<HTMLElement>('[data-hero-pulse]', root)

      if (reduced) {
        if (beam) gsap.set(beam, { opacity: 0.6 })
        return
      }

      if (beam) {
        gsap.to(beam, {
          y: '+=220',
          duration: 2.8,
          ease: 'none',
          repeat: -1,
        })
      }

      if (reticle) {
        gsap.to(reticle, {
          x: 24,
          y: -12,
          duration: 3.2,
          ease: 'sine.inOut',
          yoyo: true,
          repeat: -1,
        })
      }

      pulses.forEach((el, i) => {
        gsap.to(el, {
          opacity: 0.9,
          scale: 1.05,
          duration: 1.2 + i * 0.2,
          ease: 'sine.inOut',
          yoyo: true,
          repeat: -1,
          delay: i * 0.3,
        })
      })
    },
    { scope: rootRef }
  )

  return (
    <div
      ref={rootRef}
      className={`overflow-hidden rounded-lg border border-ink-700 bg-[#0a0c10] shadow-2xl shadow-black/40 ring-1 ring-signal/20 ${className}`}
    >
      <div className="flex items-center gap-2 border-b border-ink-700 bg-ink-800 px-3 py-2">
        <span className="h-2 w-2 rounded-full bg-red-500/60" />
        <span className="h-2 w-2 rounded-full bg-yellow-500/50" />
        <span className="h-2 w-2 rounded-full bg-green-500/50" />
        <span className="ml-2 flex-1 truncate font-mono text-[10px] text-text-secondary">
          {SITE.name.toLowerCase()} · scanning {SITE.exampleUrl}
        </span>
        <span
          data-hero-pulse
          className="rounded border border-signal/40 bg-signal/10 px-1.5 py-0.5 font-mono text-[9px] text-signal"
        >
          LIVE
        </span>
      </div>

      <div className="relative aspect-[4/3] bg-[#0f1117] p-5">
        {/* Page skeleton */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Logo variant="nav" size={20} />
            <div className="h-2 w-16 rounded-sm bg-white/15" />
          </div>
          <div className="flex gap-2">
            <div className="h-2 w-8 rounded-sm bg-white/8" />
            <div className="h-5 w-12 rounded-full bg-signal/30" />
          </div>
        </div>
        <div className="space-y-2 text-center">
          <div className="mx-auto h-3 w-3/4 rounded-sm bg-white/20" />
          <div className="mx-auto h-2 w-1/2 rounded-sm bg-white/10" />
          <div className="mx-auto mt-4 h-8 w-28 rounded-md bg-signal/35" />
        </div>
        <div className="mt-6 grid grid-cols-3 gap-2">
          <MiniFeatureTile icon={Globe} label="Pages" accent="signal" />
          <MiniFeatureTile icon={Layers} label="DOM" accent="copy" />
          <MiniFeatureTile icon={TrendingUp} label="TR 14" accent="neural" />
        </div>

        {/* Scan beam */}
        <div
          data-hero-beam
          className="pointer-events-none absolute left-4 right-4 top-5 h-px bg-gradient-to-r from-transparent via-signal to-transparent"
          style={{ boxShadow: '0 0 16px rgba(37,99,255,0.5)' }}
        />

        {/* Reticle */}
        <div
          data-hero-reticle
          className="pointer-events-none absolute left-1/2 top-1/3 h-10 w-10 -translate-x-1/2 border border-signal/50"
          style={{ boxShadow: 'inset 0 0 0 1px rgba(37,99,255,0.2)' }}
        >
          <span className="absolute -left-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-signal" />
          <span className="absolute -right-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-signal" />
          <span className="absolute left-1/2 -top-1 h-2 w-2 -translate-x-1/2 rounded-full bg-signal" />
          <span className="absolute left-1/2 -bottom-1 h-2 w-2 -translate-x-1/2 rounded-full bg-signal" />
        </div>

        {/* Partial heatmap — building */}
        <div
          className="pointer-events-none absolute inset-0 mix-blend-screen opacity-40"
          style={{
            background:
              'radial-gradient(ellipse 40% 30% at 50% 35%, rgba(255,140,50,0.5) 0%, transparent 60%)',
          }}
        />
      </div>

      <div className="grid grid-cols-3 gap-px border-t border-ink-700 bg-ink-700">
        {[
          { k: 'Pages', v: '3/8' },
          { k: 'TR', v: '14' },
          { k: 'Status', v: 'crawl' },
        ].map((m) => (
          <div key={m.k} className="bg-ink-900 px-3 py-2 text-center">
            <p className="text-[8px] text-text-secondary">{m.k}</p>
            <p className="font-mono text-xs text-signal">{m.v}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
