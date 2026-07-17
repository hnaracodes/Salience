import Link from 'next/link'
import { Logo } from '@/components/Logo'
import { SITE } from '@/lib/site'

const productLinks = [
  { href: '/#use-cases', label: 'Use cases' },
  { href: '/#product-journey', label: 'Pipeline' },
  { href: '/#how-it-works', label: 'How it works' },
  { href: '/#viewer', label: 'Viewer' },
  { href: '/scans/new', label: 'New scan' },
]

const companyLinks = [
  { href: '/about', label: 'About' },
  { href: '/security', label: 'Security' },
  { href: '/data-processing', label: 'Data Processing' },
  { href: '/anti-theft', label: 'Anti-Theft' },
]

const legalLinks = [
  { href: '/privacy', label: 'Privacy Policy' },
  { href: '/terms', label: 'Terms of Service' },
  { href: '/cookies', label: 'Cookie Policy' },
  { href: '/acceptable-use', label: 'Acceptable Use' },
]

const supportLinks = [
  { href: '/contact', label: 'Contact' },
  { href: `mailto:${SITE.supportEmail}`, label: 'Support', external: true },
]

export function SiteFooter() {
  const year = new Date().getFullYear()

  return (
    <footer className="border-t border-line bg-surface px-6 py-14">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-10 md:grid-cols-2 lg:grid-cols-5">
          <div className="lg:col-span-1">
            <Link href="/" className="inline-flex items-center gap-2.5">
              <Logo variant="nav" size={24} />
              <span className="text-sm font-semibold text-text-primary">{SITE.name}</span>
            </Link>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-text-secondary">
              {SITE.description}
            </p>
            <p className="mt-4 text-[10px] text-text-tertiary">
              © {year} {SITE.legalEntity}
            </p>
          </div>

          <div>
            <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-text-tertiary">
              Product
            </h3>
            <ul className="space-y-2">
              {productLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="cursor-pointer text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-text-tertiary">
              Company
            </h3>
            <ul className="space-y-2">
              {companyLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="cursor-pointer text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-text-tertiary">
              Legal
            </h3>
            <ul className="space-y-2">
              {legalLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="cursor-pointer text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-text-tertiary">
              Support
            </h3>
            <ul className="space-y-2">
              {supportLinks.map((link) => (
                <li key={link.href}>
                  {link.external ? (
                    <a
                      href={link.href}
                      className="cursor-pointer text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="cursor-pointer text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
            <p className="mt-4 text-[10px] text-text-tertiary">{SITE.address}</p>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-line pt-8 sm:flex-row">
          <p className="text-center text-xs text-text-secondary sm:text-left">
            By using {SITE.name}, you agree to our{' '}
            <Link href="/terms" className="text-signal hover:underline">
              Terms of Service
            </Link>{' '}
            and{' '}
            <Link href="/privacy" className="text-signal hover:underline">
              Privacy Policy
            </Link>
            .
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-[10px] text-text-tertiary">
            <Link href="/about" className="transition-colors hover:text-text-secondary">
              About
            </Link>
            <Link href="/security" className="transition-colors hover:text-text-secondary">
              Security
            </Link>
            <Link href="/privacy" className="transition-colors hover:text-text-secondary">
              Privacy
            </Link>
            <Link href="/terms" className="transition-colors hover:text-text-secondary">
              Terms
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}
