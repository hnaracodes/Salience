'use client'

const useCases = [
  {
    tag: 'Product',
    title: 'Catch hierarchy breaks before launch',
    body: 'Run a scan on every major release. Section-level VAN−DMN scores and engagement tracks show where users lose focus — on hero, pricing, or onboarding flows.',
    signal: 'Per-section neural scores',
  },
  {
    tag: 'Agency',
    title: 'Replace subjective redesign debates',
    body: 'Export heatmaps and cortical ratings for client presentations. Prove which hero layout, CTA placement, or proof block registers — not just which looks nicer in Figma.',
    signal: 'Shareable viewer URL',
  },
  {
    tag: 'Growth',
    title: 'Prioritize A/B tests with data',
    body: 'Score clarity, urgency, and goal-fit on landing variants before you spend on ads. Copy heuristics fuse with neural ratings so you know what to test first.',
    signal: '20% copy weight in score',
  },
  {
    tag: 'E‑commerce',
    title: 'Audit full purchase journeys',
    body: 'Multi-page crawls follow PDPs, cart paths, and nav depth across your site. See which product copy and imagery drive activation — not just homepage metrics.',
    signal: 'Up to 8 pages per scan',
  },
] as const

export function UseCaseSection() {
  return (
    <section id="use-cases" data-section className="border-t border-line bg-canvas-warm px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <p className="section-eyebrow" data-reveal>
          Use cases
        </p>
        <h2 className="section-heading mb-4 max-w-2xl" data-reveal data-reveal-delay="0.05">
          UX decisions backed by cortical modeling, not gut feel.
        </h2>
        <p
          className="mb-16 max-w-xl text-base leading-relaxed text-text-secondary"
          data-reveal
          data-reveal-delay="0.08"
        >
          Salience is built for teams who need to justify design changes with measurable
          attention data — from pre-launch QA to client deliverables.
        </p>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {useCases.map((item, i) => (
            <article
              key={item.title}
              className="group marketing-card cursor-default p-8 hover:-translate-y-0.5 hover:shadow-card-lg"
              data-reveal
              data-reveal-delay={String(i * 0.08)}
            >
              <div className="mb-6 flex items-center justify-between gap-4">
                <span className="inline-flex rounded-pill border border-line bg-surface-muted px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-signal">
                  {item.tag}
                </span>
                <span className="text-[10px] text-text-tertiary">{item.signal}</span>
              </div>
              <h3 className="mb-3 text-lg font-semibold text-text-primary">{item.title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{item.body}</p>
              <div
                className="mt-6 h-px w-8 bg-ink-950 opacity-20 transition-all duration-500 ease-premium group-hover:w-16 group-hover:opacity-60"
                aria-hidden="true"
              />
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
