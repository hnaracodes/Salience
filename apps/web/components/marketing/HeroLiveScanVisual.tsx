'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { Globe, Layers, TrendingUp } from 'lucide-react'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { Logo } from '@/components/Logo'
import { HEATMAP_LIGHT_SOFT, mock } from '@/components/marketing/mock-ui'
import { SITE } from '@/lib/site'

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
          duration: 4.2,
          ease: 'none',
          repeat: -1,
        })
      }

      if (reticle) {
        gsap.to(reticle, {
          x: 24,
          y: -12,
          duration: 4.8,
          ease: 'sine.inOut',
          yoyo: true,
          repeat: -1,
        })
      }

      pulses.forEach((el, i) => {
        gsap.to(el, {
          opacity: 1,
          scale: 1.04,
          duration: 1.6 + i * 0.2,
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
    <div ref={rootRef} className={`${mock.shell} ${className}`}>
      <div className={mock.chrome}>
        <span className="h-2 w-2 rounded-full bg-red-400/70" />
        <span className="h-2 w-2 rounded-full bg-amber-400/70" />
        <span className="h-2 w-2 rounded-full bg-copy/70" />
        <span className={mock.chromeTitle}>
          {SITE.name.toLowerCase()} · scanning {SITE.exampleUrl}
        </span>
        <span data-hero-pulse className={mock.badgeLive}>
          LIVE
        </span>
      </div>

      <div className={`relative aspect-[4/3] ${mock.panel} p-5`}>
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Logo variant="nav" size={20} />
            <div className={`h-2 w-16 ${mock.skeleton}`} />
          </div>
          <div className="flex gap-2">
            <div className={`h-2 w-8 ${mock.skeleton}`} />
            <div className={`h-5 w-12 rounded-full ${mock.skeletonBtn}`} />
          </div>
        </div>
        <div className="space-y-2 text-center">
          <div className={`mx-auto h-3 w-3/4 ${mock.skeletonMid}`} />
          <div className={`mx-auto h-2 w-1/2 ${mock.skeleton}`} />
          <div className={`mx-auto mt-4 h-8 w-28 rounded-lg bg-ink-950`} />
        </div>
        <div className="mt-6 grid grid-cols-3 gap-2">
          <MiniFeatureTile icon={Globe} label="Pages" accent="signal" />
          <MiniFeatureTile icon={Layers} label="DOM" accent="copy" />
          <MiniFeatureTile icon={TrendingUp} label="TR 14" accent="neural" />
        </div>

        <div
          data-hero-beam
          className="pointer-events-none absolute left-4 right-4 top-5 h-px bg-gradient-to-r from-transparent via-signal/60 to-transparent"
        />

        <div
          data-hero-reticle
          className="pointer-events-none absolute left-1/2 top-1/3 h-10 w-10 -translate-x-1/2 rounded-sm border border-signal/40 bg-signal/5"
        >
          <span className="absolute -left-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-signal" />
          <span className="absolute -right-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-signal" />
          <span className="absolute left-1/2 -top-1 h-2 w-2 -translate-x-1/2 rounded-full bg-signal" />
          <span className="absolute left-1/2 -bottom-1 h-2 w-2 -translate-x-1/2 rounded-full bg-signal" />
        </div>

        <div
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{ background: HEATMAP_LIGHT_SOFT }}
        />
      </div>

      <div className={mock.footerGrid}>
        {[
          { k: 'Pages', v: '3/8' },
          { k: 'TR', v: '14' },
          { k: 'Status', v: 'crawl' },
        ].map((m) => (
          <div key={m.k} className={mock.footerCell}>
            <p className="text-[8px] text-text-tertiary">{m.k}</p>
            <p className="font-mono text-xs font-medium text-signal">{m.v}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
