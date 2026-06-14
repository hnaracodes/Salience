import type { Metadata } from 'next'
import Link from 'next/link'
import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Contact — ${SITE.name}`,
  description: `Contact ${SITE.name} for support, legal, and privacy inquiries.`,
}

export default function ContactPage() {
  return (
    <>
      <NavBar variant="app" />
      <main className="min-h-screen bg-ink-950 pt-14">
        <div className="mx-auto max-w-2xl px-6 py-16">
          <Link
            href="/"
            className="mb-8 inline-block font-mono text-xs text-text-secondary transition-colors hover:text-signal"
          >
            ← Back to {SITE.name}
          </Link>

          <h1 className="mb-4 text-3xl font-display font-semibold text-text-primary">Contact</h1>
          <p className="mb-10 text-text-secondary">
            We typically respond within two business days.
          </p>

          <div className="space-y-6">
            {[
              {
                title: 'Product support',
                email: SITE.supportEmail,
                detail: 'Scan issues, billing questions, account access',
              },
              {
                title: 'Privacy requests',
                email: SITE.privacyEmail,
                detail: 'Data access, deletion, and GDPR/CCPA inquiries',
              },
              {
                title: 'Legal',
                email: SITE.contactEmail,
                detail: 'Terms, contracts, enterprise DPAs',
              },
            ].map((row) => (
              <div
                key={row.title}
                className="rounded-lg border border-ink-700 bg-ink-900 p-6"
              >
                <h2 className="text-lg font-semibold text-text-primary">{row.title}</h2>
                <p className="mt-1 text-sm text-text-secondary">{row.detail}</p>
                <a
                  href={`mailto:${row.email}`}
                  className="mt-3 inline-block font-mono text-sm text-signal hover:underline"
                >
                  {row.email}
                </a>
              </div>
            ))}
          </div>

          <p className="mt-10 text-sm text-text-secondary">
            {SITE.legalEntity} · {SITE.address}
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  )
}
