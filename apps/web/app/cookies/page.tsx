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
      description={`This policy describes how ${SITE.name} uses cookies and similar technologies when you visit our website or use our application.`}
    >
      <section>
        <h2>1. What are cookies?</h2>
        <p>
          Cookies are small text files stored on your device. We also use similar technologies
          such as local storage and session tokens where needed for authentication and preferences.
        </p>
      </section>

      <section>
        <h2>2. Cookies we use</h2>
        <h3>Strictly necessary</h3>
        <p>Required for the service to function. Without these, you cannot sign in or maintain a session.</p>
        <ul>
          <li>
            <strong>Clerk session cookies</strong> — authentication, fraud prevention, session
            management (provider: Clerk)
          </li>
          <li>
            <strong>Security cookies</strong> — CSRF and request integrity where applicable
          </li>
        </ul>

        <h3>Functional</h3>
        <p>Remember choices and improve experience. Optional where marked.</p>
        <ul>
          <li>UI preferences (e.g. reduced-motion respect is handled via system settings)</li>
        </ul>

        <h3>Analytics</h3>
        <p>
          We may use privacy-respecting analytics to understand product usage. If enabled, we will
          update this page with provider names and opt-out instructions before deployment.
        </p>
      </section>

      <section>
        <h2>3. Third-party cookies</h2>
        <p>
          Clerk and our hosting providers may set cookies when you use sign-in flows or embedded
          components. Review Clerk&apos;s documentation for their cookie list. Viewer reports
          hosted on separate artifact domains may use their own storage policies.
        </p>
      </section>

      <section>
        <h2>4. Managing cookies</h2>
        <ul>
          <li>Browser settings — block or delete cookies (may break sign-in)</li>
          <li>Sign out — clears session-related cookies for our app</li>
          <li>Do Not Track — we honor applicable legal requirements; no universal DNT standard exists</li>
        </ul>
      </section>

      <section>
        <h2>5. Updates</h2>
        <p>
          We will revise this policy when our cookie practices change. Contact{' '}
          <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a> with questions.
        </p>
      </section>
    </LegalPageLayout>
  )
}
