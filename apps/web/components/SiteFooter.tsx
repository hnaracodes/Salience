import Link from 'next/link'
import { Logo } from '@/components/Logo'
import { SITE } from '@/lib/site'

const productLinks = [
  { href: '/#product-journey', label: 'Pipeline' },
  { href: '/#how-it-works', label: 'How it works' },
  { href: '/#viewer', label: 'Viewer' },
  { href: '/scans/new', label: 'New scan' },
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
    <footer className="border-t border-ink-700 bg-ink-950 px-6 py-14">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-10 md:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-1">
            <Link href="/" className="inline-flex items-center gap-2.5">
              <Logo variant="nav" size={24} />
              <span className="text-sm font-medium text-text-primary">{SITE.name}</span>
            </Link>
            <p className="mt-3 max-w-xs text-sm text-text-secondary">{SITE.tagline}</p>
            <p className="mt-4 font-mono text-[10px] text-ink-600">
              © {year} {SITE.legalEntity}
            </p>
          </div>

          {/* Product */}
          <div>
            <h3 className="mb-4 font-mono text-xs uppercase tracking-widest text-text-secondary">
              Product
            </h3>
            <ul className="space-y-2">
              {productLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary transition-colors hover:text-text-primary"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="mb-4 font-mono text-xs uppercase tracking-widest text-text-secondary">
              Legal
            </h3>
            <ul className="space-y-2">
              {legalLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary transition-colors hover:text-text-primary"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Support */}
          <div>
            <h3 className="mb-4 font-mono text-xs uppercase tracking-widest text-text-secondary">
              Support
            </h3>
            <ul className="space-y-2">
              {supportLinks.map((link) => (
                <li key={link.href}>
                  {link.external ? (
                    <a
                      href={link.href}
                      className="text-sm text-text-secondary transition-colors hover:text-text-primary"
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="text-sm text-text-secondary transition-colors hover:text-text-primary"
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
            <p className="mt-4 font-mono text-[10px] text-ink-600">{SITE.address}</p>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-ink-700 pt-8 sm:flex-row">
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
          <div className="flex flex-wrap justify-center gap-4 font-mono text-[10px] text-ink-600">
            <Link href="/privacy" className="hover:text-text-secondary">
              Privacy
            </Link>
            <Link href="/terms" className="hover:text-text-secondary">
              Terms
            </Link>
            <Link href="/cookies" className="hover:text-text-secondary">
              Cookies
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}
