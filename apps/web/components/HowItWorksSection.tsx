'use client'

import { ReportReadyVisual } from '@/components/marketing/PipelineVisuals'
import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { SITE } from '@/lib/site'

const steps = [
  {
    number: '01',
    title: 'Submit URL',
    description:
      `Paste any public URL. SSRF-protected intake validates the host before ${SITE.name} crawls.`,
    visual: 'submit' as const,
  },
  {
    number: '02',
    title: `${SITE.name} crawls`,
    description:
      'Playwright scrolls each page to the bottom, clicks nav links, and records video + DOM snapshots.',
    visual: 'crawl' as const,
  },
  {
    number: '03',
    title: 'Neural report',
    description:
      'TRIBE v2, heatmaps, copy signals, and engagement tracks compile into your interactive viewer.',
    visual: 'report' as const,
  },
]

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="border-t border-ink-700 bg-ink-950 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="mb-4 text-xs font-mono uppercase tracking-widest text-text-secondary" data-reveal>
          How it works
        </p>
        <h2
          className="mb-16 text-3xl font-display font-semibold tracking-tight text-text-primary md:text-4xl"
          style={{ letterSpacing: '-0.02em' }}
          data-reveal
          data-reveal-delay="0.05"
        >
          Three steps to neural clarity.
        </h2>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {steps.map((step, i) => (
            <article
              key={step.number}
              className="flex flex-col overflow-hidden rounded-lg border border-ink-700 bg-ink-900"
              data-reveal
              data-reveal-delay={String(i * 0.1)}
            >
              <div className="border-b border-ink-700 bg-ink-950">
                <StepVisual type={step.visual} />
              </div>

              <div className="flex flex-1 flex-col p-6">
                <span className="mb-3 font-mono text-4xl font-semibold text-ink-700" aria-hidden="true">
                  {step.number}
                </span>
                <h3 className="mb-2 text-lg font-semibold text-text-primary">{step.title}</h3>
                <p className="text-sm leading-relaxed text-text-secondary">{step.description}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function StepVisual({ type }: { type: 'submit' | 'crawl' | 'report' }) {
  if (type === 'submit') {
    return (
      <div className="flex aspect-[16/10] flex-col justify-center gap-3 p-6">
        <label className="text-[10px] font-mono uppercase tracking-wider text-text-secondary">
          Website URL
        </label>
        <div className="flex items-center gap-2 rounded-md border border-ink-600 bg-ink-900 px-4 py-3">
          <span className="font-mono text-sm text-signal">https://</span>
          <span className="flex-1 font-mono text-sm text-text-primary">{SITE.exampleUrl}</span>
          <span className="rounded bg-signal px-3 py-1 text-xs font-medium text-white">Scan</span>
        </div>
        <p className="text-[10px] font-mono text-copy">✓ Host validated · same-origin crawl ready</p>
      </div>
    )
  }

  if (type === 'crawl') {
    return (
      <div className="aspect-[16/10] p-4">
        <div className="flex h-full flex-col gap-1.5">
          {['/', '/features', '/pricing'].map((path, i) => (
            <div key={path} className="flex flex-1 items-center gap-2 rounded border border-ink-700 bg-ink-900 px-2">
              <PageRouteIcon path={path} size="sm" />
              <span className="font-mono text-[10px] text-signal">{path}</span>
              <div className="ml-auto h-1 w-12 rounded-full bg-ink-700">
                <div className="h-full rounded-full bg-copy" style={{ width: `${70 + i * 10}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return <ReportReadyVisual />
}
