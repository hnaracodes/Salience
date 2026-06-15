'use client'

import { FeatureVisual, type FeatureVisualType } from '@/components/marketing/FeatureVisuals'

const features: {
  visual: FeatureVisualType
  title: string
  body: string
  metric: string
}[] = [
  {
    visual: 'neural',
    title: 'Neural Scoring',
    body: "TRIBE v2 predicts cortical engagement from walkthrough video — not eye-tracking, actual brain-response modeling.",
    metric: 'VAN−DMN Z tracked per TR',
  },
  {
    visual: 'crawl',
    title: 'Multi-page Crawl',
    body: 'Scroll-then-nav exploration hits every same-origin page. DOM text and video captured per timeline frame.',
    metric: 'Up to 8 pages per scan',
  },
  {
    visual: 'copy',
    title: 'Copy Analysis',
    body: 'Clarity, urgency, and goal-fit scores from visible page text — fused into the final neural compound score.',
    metric: '20% copy weight in score',
  },
  {
    visual: 'heatmap',
    title: 'Live Heatmaps',
    body: 'DINOv2 attention maps overlaid on every captured frame. See exactly which pixels drive neural response.',
    metric: 'Warm ramp · yellow→red',
  },
]

export function FeaturesSection() {
  return (
    <section className="border-t border-ink-700 bg-ink-900 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="mb-4 text-xs font-mono uppercase tracking-widest text-text-secondary" data-reveal>
          What you get
        </p>
        <h2
          className="mb-16 text-3xl font-display font-semibold tracking-tight text-text-primary md:text-4xl"
          style={{ letterSpacing: '-0.02em' }}
          data-reveal
          data-reveal-delay="0.05"
        >
          Four systems. One score.
        </h2>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {features.map((feature, i) => (
            <article
              key={feature.title}
              className="group overflow-hidden rounded-lg border border-ink-700 bg-ink-950 transition-colors hover:border-ink-600"
              data-reveal
              data-reveal-delay={String(i * 0.08)}
            >
              <FeatureVisual type={feature.visual} />

              <div className="relative p-6">
                <div className="absolute left-0 top-0 h-full w-0.5 bg-signal opacity-60 group-hover:opacity-100" aria-hidden="true" />

                <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-signal">
                  {feature.metric}
                </p>
                <h3 className="mb-2 text-lg font-semibold text-text-primary">{feature.title}</h3>
                <p className="text-sm leading-relaxed text-text-secondary">{feature.body}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
