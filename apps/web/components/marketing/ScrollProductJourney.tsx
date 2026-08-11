'use client'

import { useEffect, useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'
import {
  CrawlPhaseVisual,
  HeatmapPhaseVisual,
  InspectPhaseVisual,
  NeuralPhaseVisual,
} from '@/components/marketing/PipelineVisuals'

gsap.registerPlugin(ScrollTrigger)

const phases = [
  {
    id: 'crawl',
    label: '01 · Crawl',
    title: 'Every page captured on video',
    body: 'Playwright scrolls each route to the bottom, records walkthrough video, and snapshots DOM text per timeline frame.',
  },
  {
    id: 'neural',
    label: '02 · Infer',
    title: 'TRIBE models cortical response',
    body: 'Engagement tracks and VAN−DMN Z-scores are computed from walkthrough video, separately from the pixel-level heatmap.',
  },
  {
    id: 'heatmap',
    label: '03 · Heatmap',
    title: 'Attention mapped per pixel',
    body: 'DeepGaze, a saliency model trained on human eye-tracking data, sweeps across captured frames so you see exactly which regions draw the eye.',
  },
  {
    id: 'inspect',
    label: '04 · Inspect',
    title: 'Click any element for fused scores',
    body: 'Neural engagement, copy clarity, urgency, and goal-fit combine into one actionable score per element.',
  },
]

export function ScrollProductJourney() {
  const sectionRef = useRef<HTMLElement>(null)
  const pinRef = useRef<HTMLDivElement>(null)
  const textRefs = useRef<(HTMLDivElement | null)[]>([])
  const progressRef = useRef<HTMLDivElement>(null)
  const dotRefs = useRef<(HTMLSpanElement | null)[]>([])
  const [reducedMotion, setReducedMotion] = useState(false)

  useEffect(() => {
    setReducedMotion(window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  }, [])

  useGSAP(
    () => {
      const section = sectionRef.current
      const pin = pinRef.current
      if (!section || !pin) return

      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

      const panels = gsap.utils.toArray<HTMLElement>('[data-journey-panel]', section)
      const crawlRows = gsap.utils.toArray<HTMLElement>('[data-crawl-row]', section)
      const crawlBars = gsap.utils.toArray<HTMLElement>('[data-crawl-bar]', section)
      const crawlBeam = section.querySelector('[data-crawl-beam]')
      const neuralPath = section.querySelector('[data-neural-path]')
      const neuralDot = section.querySelector('[data-neural-dot]')
      const wavePath = section.querySelector('[data-wave-path]')
      const heatmapLayer = section.querySelector('[data-heatmap-layer]')
      const heatmapSweep = section.querySelector('[data-heatmap-sweep]')
      const inspectBboxes = gsap.utils.toArray<HTMLElement>('[data-inspect-bbox]', section)
      const inspectChips = gsap.utils.toArray<HTMLElement>('[data-inspect-chip]', section)
      const inspectLabel = section.querySelector('[data-inspect-label]')

      if (reduced) {
        // Static finale — show inspect phase + completed crawl state
        panels.forEach((p, i) => gsap.set(p, { opacity: i === 3 ? 1 : 0 }))
        textRefs.current.forEach((t, i) => t && gsap.set(t, { opacity: i === 3 ? 1 : 0, y: 0 }))
        crawlRows.forEach((r) => gsap.set(r, { opacity: 1, x: 0 }))
        crawlBars.forEach((b) => {
          const w = (b as HTMLElement).dataset.crawlWidth ?? '100%'
          gsap.set(b, { width: w })
        })
        gsap.set(inspectBboxes, { opacity: 1, scale: 1 })
        gsap.set(inspectChips, { opacity: 1, y: 0 })
        if (inspectLabel) gsap.set(inspectLabel, { opacity: 1 })
        if (progressRef.current) progressRef.current.style.width = '100%'
        dotRefs.current.forEach((dot, i) => {
          if (!dot) return
          dot.classList.toggle('bg-ink-950', i === 3)
          dot.classList.toggle('scale-125', i === 3)
          dot.classList.toggle('bg-line-strong', i !== 3)
        })
        return
      }

      gsap.set(panels, { opacity: 0 })
      gsap.set(panels[0], { opacity: 1 })
      textRefs.current.forEach((t, i) => t && gsap.set(t, { opacity: i === 0 ? 1 : 0, y: 0 }))
      gsap.set(crawlRows, { opacity: 0, x: -12 })
      gsap.set(crawlBars, { width: 0 })
      if (crawlBeam) gsap.set(crawlBeam, { opacity: 0, y: 0 })
      if (neuralPath) gsap.set(neuralPath, { strokeDashoffset: 200 })
      if (neuralDot) gsap.set(neuralDot, { opacity: 0 })
      if (wavePath) gsap.set(wavePath, { strokeDashoffset: 220 })
      if (heatmapLayer) gsap.set(heatmapLayer, { opacity: 0 })
      if (heatmapSweep) gsap.set(heatmapSweep, { opacity: 0, x: '-10%' })
      gsap.set(inspectBboxes, { opacity: 0, scale: 0.92 })
      gsap.set(inspectChips, { opacity: 0, y: 8 })
      if (inspectLabel) gsap.set(inspectLabel, { opacity: 0 })

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: 'top top',
          end: 'bottom bottom',
          scrub: 1.8,
          pin: pin,
          anticipatePin: 1,
          snap: {
            snapTo: 0.25,
            duration: { min: 0.35, max: 0.7 },
            ease: 'power4.inOut',
          },
          onUpdate(self) {
            if (progressRef.current) {
              progressRef.current.style.width = `${self.progress * 100}%`
            }
            const active = Math.min(3, Math.floor(self.progress * 4))
            dotRefs.current.forEach((dot, i) => {
              if (!dot) return
              dot.classList.toggle('bg-ink-950', i === active)
              dot.classList.toggle('scale-125', i === active)
              dot.classList.toggle('bg-line-strong', i !== active)
            })
            textRefs.current.forEach((t, i) => {
              if (!t) return
              gsap.to(t, {
                opacity: i === active ? 1 : 0,
                y: i === active ? 0 : i < active ? -10 : 10,
                duration: 0.65,
                ease: 'power4.out',
                overwrite: true,
              })
            })
          },
        },
      })

      // ── Phase 1: Crawl (0 → 0.25) ──
      tl.to(crawlBeam, { opacity: 1, y: '+=180', duration: 0.22, ease: 'none' }, 0)
      tl.to(
        crawlRows,
        { opacity: 1, x: 0, stagger: 0.04, duration: 0.08, ease: 'power2.out' },
        0.02
      )
      tl.to(crawlBars, {
        width: (i, el) => (el as HTMLElement).dataset.crawlWidth ?? '100%',
        stagger: 0.04,
        duration: 0.12,
        ease: 'power1.out',
      }, 0.06)

      tl.to(panels[0], { opacity: 0, duration: 0.06 }, 0.22)
      tl.to(panels[1], { opacity: 1, duration: 0.06 }, 0.24)

      // ── Phase 2: Neural (0.25 → 0.5) ──
      tl.to(neuralPath, { strokeDashoffset: 0, duration: 0.18, ease: 'power1.inOut' }, 0.26)
      tl.to(neuralDot, { opacity: 1, duration: 0.06 }, 0.38)
      tl.to(wavePath, { strokeDashoffset: 0, duration: 0.16, ease: 'none' }, 0.28)

      tl.to(panels[1], { opacity: 0, duration: 0.06 }, 0.47)
      tl.to(panels[2], { opacity: 1, duration: 0.06 }, 0.49)

      // ── Phase 3: Heatmap (0.5 → 0.75) ──
      tl.to(heatmapLayer, { opacity: 0.85, duration: 0.2, ease: 'power2.in' }, 0.5)
      tl.to(heatmapSweep, { opacity: 1, x: '110%', duration: 0.22, ease: 'none' }, 0.52)

      tl.to(panels[2], { opacity: 0, duration: 0.06 }, 0.72)
      tl.to(panels[3], { opacity: 1, duration: 0.06 }, 0.74)

      // ── Phase 4: Inspect (0.75 → 1) ──
      tl.to(inspectBboxes, { opacity: 1, scale: 1, stagger: 0.04, duration: 0.1, ease: 'power2.out' }, 0.76)
      tl.to(inspectLabel, { opacity: 1, duration: 0.06 }, 0.82)
      tl.to(inspectChips, { opacity: 1, y: 0, stagger: 0.03, duration: 0.08 }, 0.84)

      return () => {
        ScrollTrigger.getAll().forEach((st) => {
          if (st.trigger === section) st.kill()
        })
      }
    },
    { scope: sectionRef, dependencies: [] }
  )

  return (
    <section
      ref={sectionRef}
      id="product-journey"
      data-section
      className="relative border-t border-line bg-surface-muted"
      style={reducedMotion ? undefined : { height: '400vh' }}
      aria-label="Product pipeline — scroll to explore"
    >
      <div ref={pinRef} className={`relative flex flex-col justify-center px-6 ${reducedMotion ? 'py-28' : 'h-screen'}`}>
        <div className="mx-auto w-full max-w-6xl">
          <p className="section-eyebrow mb-3 text-signal">
            Scroll the pipeline
          </p>

          <div className="grid items-center gap-10 lg:grid-cols-2 lg:gap-16">
            {/* Copy — phases swap on scroll */}
            <div className="relative min-h-[200px]">
              {phases.map((phase, i) => (
                <div
                  key={phase.id}
                  ref={(el) => {
                    textRefs.current[i] = el
                  }}
                  className="absolute inset-0"
                  style={{ opacity: i === 0 ? 1 : 0 }}
                >
                  <span className="text-xs font-medium text-text-tertiary">{phase.label}</span>
                  <h2 className="section-heading mt-2 text-2xl md:text-3xl">
                    {phase.title}
                  </h2>
                  <p className="mt-4 max-w-md text-base leading-relaxed text-text-secondary">
                    {phase.body}
                  </p>
                </div>
              ))}
            </div>

            {/* Visual stack — crossfade panels */}
            <div className="relative aspect-[4/3] w-full max-w-xl justify-self-center lg:justify-self-end">
              <div data-journey-panel className="product-mock-shell absolute inset-0">
                <CrawlPhaseVisual className="h-full" />
              </div>
              <div data-journey-panel className="product-mock-shell absolute inset-0 opacity-0">
                <NeuralPhaseVisual className="h-full" />
              </div>
              <div data-journey-panel className="product-mock-shell absolute inset-0 opacity-0">
                <HeatmapPhaseVisual className="h-full" />
              </div>
              <div data-journey-panel className="product-mock-shell absolute inset-0 opacity-0">
                <InspectPhaseVisual className="h-full" />
              </div>
            </div>
          </div>

          {/* Scroll progress */}
          <div className="mt-12 flex items-center gap-4">
            <div className="h-1 flex-1 overflow-hidden rounded-pill bg-line">
              <div
                ref={progressRef}
                className="h-full w-0 rounded-pill bg-ink-950 transition-none"
              />
            </div>
            <div className="flex gap-2">
              {phases.map((p, i) => (
                <span
                  key={p.id}
                  ref={(el) => {
                    dotRefs.current[i] = el
                  }}
                  className={`h-2 w-2 rounded-full transition-all duration-500 ease-premium ${i === 0 ? 'bg-ink-950 scale-125' : 'bg-line-strong'}`}
                  aria-hidden="true"
                />
              ))}
            </div>
            <span className="hidden text-[10px] text-text-tertiary sm:block">
              scroll to explore
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
