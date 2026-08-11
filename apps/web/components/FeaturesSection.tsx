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
    body: 'TRIBE v2 models cortical engagement from walkthrough video — VAN−DMN Z-scores tracked per timeline frame, distinct from the eye-tracking-based heatmaps below.',
    metric: 'VAN−DMN Z per TR',
  },
  {
    visual: 'crawl',
    title: 'Multi-page Crawl',
    body: 'Playwright scrolls each route, follows nav links, and captures DOM text plus video for every page in the same-origin journey.',
    metric: 'Up to 8 pages per scan',
  },
  {
    visual: 'copy',
    title: 'Copy Analysis',
    body: 'Clarity, urgency, and goal-fit scores from visible page text — fused at 20% weight into the final neural compound score.',
    metric: 'Per-element copy chips',
  },
  {
    visual: 'heatmap',
    title: 'Attention Heatmaps',
    body: 'DeepGaze eye-tracking-trained saliency maps overlaid on every captured frame. Warm regions show exactly which pixels drive cortical spikes.',
    metric: 'Yellow → red saliency ramp',
  },
]

export function FeaturesSection() {
  return (
    <section data-section className="border-t border-line bg-surface px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="section-eyebrow" data-reveal>
          What you get
        </p>
        <h2 className="section-heading mb-4" data-reveal data-reveal-delay="0.05">
          Four analysis layers. One actionable score.
        </h2>
        <p
          className="mb-16 max-w-xl text-base text-text-secondary"
          data-reveal
          data-reveal-delay="0.08"
        >
          Every scan produces cortical ratings, attention heatmaps, copy signals, and
          per-element inspection — all in a single interactive viewer.
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {features.map((feature, i) => (
            <article
              key={feature.title}
              className="group marketing-card hover:-translate-y-0.5 hover:shadow-card-lg"
              data-reveal
              data-reveal-delay={String(i * 0.08)}
            >
              <div className="overflow-hidden rounded-t-card border-b border-line">
                <FeatureVisual type={feature.visual} />
              </div>

              <div className="relative p-6">
                <div
                  className="absolute left-0 top-0 h-full w-0.5 bg-signal opacity-40 transition-opacity duration-500 group-hover:opacity-100"
                  aria-hidden="true"
                />

                <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-signal">
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
