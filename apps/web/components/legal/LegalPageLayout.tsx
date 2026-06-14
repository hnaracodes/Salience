import Link from 'next/link'
import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'
import { SITE } from '@/lib/site'

type LegalPageLayoutProps = {
  title: string
  description: string
  children: React.ReactNode
}

export function LegalPageLayout({ title, description, children }: LegalPageLayoutProps) {
  return (
    <>
      <NavBar variant="app" />
      <main className="min-h-screen bg-ink-950 pt-14">
        <div className="mx-auto max-w-3xl px-6 py-16">
          <Link
            href="/"
            className="mb-8 inline-block font-mono text-xs text-text-secondary transition-colors hover:text-signal"
          >
            ← Back to {SITE.name}
          </Link>

          <p className="mb-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
            Legal
          </p>
          <h1 className="mb-3 text-3xl font-display font-semibold tracking-tight text-text-primary">
            {title}
          </h1>
          <p className="mb-2 text-sm text-text-secondary">{description}</p>
          <p className="mb-10 font-mono text-xs text-ink-600">
            Last updated: {SITE.lastUpdated}
          </p>

          <article className="prose-legal space-y-6 text-sm leading-relaxed text-text-secondary">
            {children}
          </article>

          <p className="mt-12 rounded-lg border border-ink-700 bg-ink-900 p-4 text-xs text-text-secondary">
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
