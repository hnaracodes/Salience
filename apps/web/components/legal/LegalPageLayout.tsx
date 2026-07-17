import Link from 'next/link'
import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'
import { SITE } from '@/lib/site'

type LegalPageLayoutProps = {
  title: string
  description: string
  category?: string
  children: React.ReactNode
}

export function LegalPageLayout({
  title,
  description,
  category = 'Legal',
  children,
}: LegalPageLayoutProps) {
  return (
    <>
      <NavBar variant="app" />
      <main className="min-h-screen bg-canvas pt-14">
        <div className="mx-auto max-w-3xl px-6 py-16">
          <Link
            href="/"
            className="mb-8 inline-block text-xs text-text-secondary transition-colors duration-300 hover:text-signal"
          >
            ← Back to {SITE.name}
          </Link>

          <span className="mb-3 inline-flex rounded-pill border border-line bg-surface px-3 py-1 text-[10px] font-medium uppercase tracking-widest text-signal">
            {category}
          </span>
          <h1 className="section-heading mb-3">{title}</h1>
          <p className="mb-2 text-sm text-text-secondary">{description}</p>
          <p className="mb-10 text-xs text-text-tertiary">Last updated: {SITE.lastUpdated}</p>

          <article className="prose-legal space-y-6 text-sm leading-relaxed text-text-secondary">
            {children}
          </article>

          <p className="mt-12 rounded-card border border-line bg-surface p-4 text-xs text-text-secondary shadow-card">
            These documents are provided for operational use. Have counsel review before
            relying on them for regulated industries or enterprise contracts. Questions:{' '}
            <a href={`mailto:${SITE.contactEmail}`} className="text-signal hover:underline">
              {SITE.contactEmail}
            </a>
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  )
}
