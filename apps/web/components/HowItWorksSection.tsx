'use client'

import { ReportReadyVisual } from '@/components/marketing/PipelineVisuals'
import { PageRouteIcon } from '@/components/marketing/PageRouteIcon'
import { SITE } from '@/lib/site'

const steps = [
  {
    number: '01',
    title: 'Submit URL',
    description:
      `Paste any public URL. SSRF-protected intake validates the host before ${SITE.name} queues the crawl.`,
    visual: 'submit' as const,
  },
  {
    number: '02',
    title: `${SITE.name} crawls`,
    description:
      'Playwright scrolls each page to the bottom, follows same-origin nav links, and records walkthrough video plus DOM snapshots per timeline frame.',
    visual: 'crawl' as const,
  },
  {
    number: '03',
    title: 'Neural report',
    description:
      'TRIBE v2 cortical inference, DeepGaze eye-tracking heatmaps, copy signals, and engagement tracks compile into your interactive viewer.',
    visual: 'report' as const,
  },
]

export function HowItWorksSection() {
  return (
    <section id="how-it-works" data-section className="border-t border-line bg-canvas px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="section-eyebrow" data-reveal>
          How it works
        </p>
        <h2 className="section-heading mb-4" data-reveal data-reveal-delay="0.05">
          URL in. Neural UX report out.
        </h2>
        <p
          className="mb-16 max-w-xl text-base text-text-secondary"
          data-reveal
          data-reveal-delay="0.08"
        >
          Three steps from submission to an interactive viewer you can share with your team or
          clients.
        </p>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {steps.map((step, i) => (
            <article
              key={step.number}
              className="marketing-card flex flex-col hover:shadow-card-lg"
              data-reveal
              data-reveal-delay={String(i * 0.1)}
            >
              <div className="border-b border-line bg-surface-muted">
                <StepVisual type={step.visual} />
              </div>

              <div className="flex flex-1 flex-col p-6">
                <span
                  className="mb-3 font-mono text-4xl font-semibold text-line-strong"
                  aria-hidden="true"
                >
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
        <label className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">
          Website URL
        </label>
        <div className="flex items-center gap-2 rounded-card border border-line bg-surface px-4 py-3 shadow-card">
          <span className="font-mono text-sm text-signal">https://</span>
          <span className="flex-1 font-mono text-sm text-text-primary">{SITE.exampleUrl}</span>
          <span className="rounded-pill bg-ink-950 px-4 py-1.5 text-xs font-medium text-white">
            Scan
          </span>
        </div>
        <p className="text-[10px] text-copy">Host validated · same-origin crawl ready</p>
      </div>
    )
  }

  if (type === 'crawl') {
    return (
      <div className="aspect-[16/10] p-4">
        <div className="flex h-full flex-col gap-2">
          {['/', '/features', '/pricing'].map((path, i) => (
            <div
              key={path}
              className="flex flex-1 items-center gap-2 rounded-lg border border-line bg-surface px-3 shadow-card"
            >
              <PageRouteIcon path={path} size="sm" />
              <span className="font-mono text-[10px] text-signal">{path}</span>
              <div className="ml-auto h-1 w-12 rounded-full bg-line">
                <div
                  className="h-full rounded-full bg-copy"
                  style={{ width: `${70 + i * 10}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-t-card">
      <ReportReadyVisual />
    </div>
  )
}
