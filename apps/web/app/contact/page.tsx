import type { Metadata } from 'next'
import Link from 'next/link'
import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Contact — ${SITE.name}`,
  description: `Get in touch with ${SITE.name} for support, legal, privacy, and security inquiries.`,
}

const CONTACT_CHANNELS = [
  {
    label: 'Support',
    email: SITE.supportEmail,
    description: 'Product help, billing questions, and account issues.',
  },
  {
    label: 'Legal',
    email: SITE.contactEmail,
    description: 'Terms, contracts, DMCA notices, and general legal inquiries.',
  },
  {
    label: 'Privacy',
    email: SITE.privacyEmail,
    description: 'Data access, deletion, and privacy rights requests.',
  },
  {
    label: 'Security',
    email: SITE.securityEmail,
    description: 'Vulnerability reports, abuse, and unauthorized scan complaints.',
  },
] as const

const RESOURCE_LINKS = [
  { href: '/security', label: 'Security practices' },
  { href: '/data-processing', label: 'Data processing' },
  { href: '/privacy', label: 'Privacy Policy' },
  { href: '/acceptable-use', label: 'Acceptable Use' },
  { href: '/anti-theft', label: 'IP & Anti-Theft' },
] as const

export default function ContactPage() {
  return (
    <>
      <NavBar variant="app" />
      <main className="min-h-screen bg-canvas pt-14">
        <div className="mx-auto max-w-3xl px-6 py-16">
          <Link
            href="/"
            className="mb-8 inline-block font-mono text-xs text-text-secondary transition-colors hover:text-signal"
          >
            ← Back to {SITE.name}
          </Link>

          <p className="mb-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
            Contact
          </p>
          <h1 className="mb-3 text-3xl font-display font-semibold tracking-tight text-text-primary">
            Get in touch
          </h1>
          <p className="mb-10 max-w-xl text-sm text-text-secondary">
            Reach the right team for support, legal, privacy, or security. We typically respond
            within a few business days.
          </p>

          <div className="grid gap-4 sm:grid-cols-2">
            {CONTACT_CHANNELS.map((channel) => (
              <div
                key={channel.label}
                className="marketing-card p-5 transition-all duration-300 hover:shadow-card-lg"
              >
                <h2 className="mb-1 text-sm font-medium text-text-primary">{channel.label}</h2>
                <a
                  href={`mailto:${channel.email}`}
                  className="font-mono text-sm text-signal hover:underline"
                >
                  {channel.email}
                </a>
                <p className="mt-2 text-xs leading-relaxed text-text-secondary">
                  {channel.description}
                </p>
              </div>
            ))}
          </div>

          <section className="mt-12 marketing-card p-6">
            <h2 className="mb-2 text-sm font-medium text-text-primary">Company</h2>
            <p className="text-sm text-text-secondary">
              {SITE.legalEntity}
              <br />
              {SITE.address}
            </p>
            <p className="mt-4 text-sm text-text-secondary">
              {SITE.founder.name}, {SITE.founder.title}
              <br />
              {SITE.founder.location}
            </p>
          </section>

          <section className="mt-10">
            <h2 className="mb-4 font-mono text-xs uppercase tracking-widest text-text-secondary">
              Resources
            </h2>
            <ul className="flex flex-wrap gap-x-6 gap-y-2">
              {RESOURCE_LINKS.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary transition-colors hover:text-signal"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </section>

          <p className="mt-12 marketing-card p-4 text-xs text-text-secondary">
            For enterprise DPAs, security questionnaires, or regulated-industry deployments, email{' '}
            <a href={`mailto:${SITE.contactEmail}`} className="text-signal hover:underline">
              {SITE.contactEmail}
            </a>{' '}
            with &quot;Enterprise&quot; in the subject line. Legal documents should be reviewed by
            counsel before reliance in regulated contexts.
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  )
}
