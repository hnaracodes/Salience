import type { Metadata } from 'next'
import Link from 'next/link'
import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `About — ${SITE.name}`,
  description: `Meet the team behind ${SITE.name} — neural UX analytics from walkthrough video.`,
}

export default function AboutPage() {
  const { founder } = SITE

  return (
    <>
      <NavBar variant="app" />
      <main className="min-h-screen bg-canvas pt-14">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <Link
            href="/"
            className="mb-10 inline-block font-mono text-xs text-text-secondary transition-colors hover:text-signal"
          >
            ← Back to {SITE.name}
          </Link>

          <div className="grid gap-16 lg:grid-cols-[1fr_320px] lg:gap-20">
            <div>
              <p className="mb-3 font-mono text-xs uppercase tracking-widest text-text-secondary">
                About
              </p>
              <h1
                className="mb-6 text-3xl font-display font-semibold tracking-tight text-text-primary md:text-5xl"
                style={{ letterSpacing: '-0.03em' }}
              >
                Neural UX intelligence,<br />
                <span className="text-signal">built for product teams.</span>
              </h1>
              <p className="mb-6 max-w-2xl text-base leading-relaxed text-text-secondary">
                {SITE.description}
              </p>
              <p className="max-w-2xl text-sm leading-relaxed text-text-secondary">
                {SITE.name} connects browser walkthroughs to TRIBE v2 cortical modeling, spatial
                grounding, and copy signals — turning session video into actionable scores, heatmaps,
                and an interactive viewer. Outputs are research-grade hypotheses for iteration, not
                clinical diagnostics.
              </p>
            </div>

            <aside className="lg:pt-12">
              <div className="marketing-card p-6">
                <p className="mb-4 font-mono text-[10px] uppercase tracking-widest text-text-secondary">
                  Founder
                </p>

                <div className="mb-1 flex h-12 w-12 items-center justify-center rounded-lg border border-line bg-surface-muted font-mono text-sm text-signal">
                  {founder.name
                    .split(' ')
                    .map((part) => part[0])
                    .slice(0, 2)
                    .join('')}
                </div>

                <h2 className="mt-4 text-lg font-semibold text-text-primary">{founder.name}</h2>
                <p className="text-sm text-signal">{founder.title}</p>
                <p className="mt-1 text-[10px] text-text-tertiary">{founder.location}</p>

                <p className="mt-4 text-sm leading-relaxed text-text-secondary">{founder.bio}</p>

                <ul className="mt-6 flex flex-wrap gap-3">
                  <li>
                    <a
                      href={founder.links.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="cursor-pointer font-mono text-xs text-text-secondary transition-colors hover:text-signal"
                    >
                      LinkedIn
                    </a>
                  </li>
                  <li>
                    <a
                      href={founder.links.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="cursor-pointer font-mono text-xs text-text-secondary transition-colors hover:text-signal"
                    >
                      GitHub
                    </a>
                  </li>
                  <li>
                    <a
                      href={founder.links.devpost}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="cursor-pointer font-mono text-xs text-text-secondary transition-colors hover:text-signal"
                    >
                      Devpost
                    </a>
                  </li>
                </ul>
              </div>
            </aside>
          </div>

          <div className="mt-20 grid gap-6 border-t border-line pt-16 md:grid-cols-3">
            {[
              {
                label: 'Legal entity',
                value: SITE.legalEntity,
              },
              {
                label: 'Contact',
                value: SITE.supportEmail,
                href: `mailto:${SITE.supportEmail}`,
              },
              {
                label: 'Location',
                value: SITE.address,
              },
            ].map((item) => (
              <div key={item.label}>
                <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-text-secondary">
                  {item.label}
                </p>
                {item.href ? (
                  <a
                    href={item.href}
                    className="cursor-pointer text-sm text-text-primary transition-colors hover:text-signal"
                  >
                    {item.value}
                  </a>
                ) : (
                  <p className="text-sm text-text-primary">{item.value}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  )
}
