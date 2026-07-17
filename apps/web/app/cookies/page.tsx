import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Cookie Policy — ${SITE.name}`,
  description: `How ${SITE.name} uses cookies and similar technologies.`,
}

export default function CookiesPage() {
  return (
    <LegalPageLayout
      title="Cookie Policy"
      description={`This policy explains how ${SITE.legalEntity} uses cookies and similar technologies when you visit ${SITE.name} (${SITE.domain}).`}
    >
      <section>
        <h2>1. What are cookies?</h2>
        <p>
          Cookies are small text files stored on your device when you visit a website. Similar
          technologies include local storage, session storage, and pixels. They help sites
          remember preferences, keep you signed in, and understand how the service is used.
        </p>
      </section>

      <section>
        <h2>2. How we use cookies</h2>
        <p>
          {SITE.name} uses cookies and similar technologies for the following purposes:
        </p>
        <h3>Strictly necessary (authentication and security)</h3>
        <ul>
          <li>
            <strong>Clerk session cookies</strong> — maintain your signed-in state, protect
            against cross-site request forgery, and validate your identity when you access the
            dashboard or submit scans. Clerk is our authentication provider; see{' '}
            <a href="https://clerk.com/privacy" rel="noopener noreferrer" target="_blank">
              Clerk&apos;s privacy policy
            </a>{' '}
            for provider-specific details.
          </li>
          <li>
            <strong>API authorization</strong> — when the web app calls our FastAPI backend, it
            sends a Clerk-issued JWT (Bearer token). This is not a browser cookie but is part of
            the same authentication flow.
          </li>
        </ul>
        <h3>Functional</h3>
        <ul>
          <li>Remember UI preferences within the dashboard where applicable</li>
          <li>Maintain scan list pagination or filter state during a session</li>
        </ul>
        <h3>Analytics (if enabled)</h3>
        <p>
          We may use privacy-conscious analytics on the marketing site to understand traffic and
          improve the product. If we enable third-party analytics, we will update this page and,
          where required, request consent before non-essential cookies are set.
        </p>
      </section>

      <section>
        <h2>3. Cookies we do not use for advertising</h2>
        <p>
          {SITE.name} does not sell personal information and does not use cookies for
          cross-site behavioral advertising. Scan artifacts and viewer sessions are served from
          our object storage (Cloudflare R2 or local MinIO in development) via presigned or
          public URLs — not via third-party ad trackers.
        </p>
      </section>

      <section>
        <h2>4. Third-party cookies</h2>
        <p>Third parties that may set cookies or similar identifiers when you use {SITE.name}:</p>
        <ul>
          <li>
            <strong>Clerk</strong> — sign-in, sign-up, and session management on{' '}
            {SITE.domain} and subdomains
          </li>
          <li>
            <strong>Vercel</strong> — hosting and performance for the Next.js frontend (may set
            infrastructure cookies)
          </li>
        </ul>
        <p>
          When you open a scan viewer URL, assets load from our storage provider. Those requests
          do not set marketing cookies on our behalf.
        </p>
      </section>

      <section>
        <h2>5. Managing cookies</h2>
        <p>
          You can control cookies through your browser settings — block, delete, or receive alerts
          when cookies are set. Blocking strictly necessary cookies may prevent you from signing
          in or using core features.
        </p>
        <p>
          To end your session, sign out of {SITE.name} or clear Clerk-related cookies for{' '}
          {SITE.domain}.
        </p>
      </section>

      <section>
        <h2>6. Related policies</h2>
        <p>
          For how we collect and use personal data beyond cookies, see our{' '}
          <a href="/privacy">Privacy Policy</a>,{' '}
          <a href="/data-processing">Data Processing</a> page, and{' '}
          <a href="/security">Security</a> page.
        </p>
      </section>

      <section>
        <h2>7. Contact</h2>
        <p>
          Questions about this Cookie Policy:{' '}
          <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a> or{' '}
          <a href="/contact">Contact us</a>.
        </p>
      </section>
    </LegalPageLayout>
  )
}
