import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Privacy Policy — ${SITE.name}`,
  description: `How ${SITE.name} collects, uses, and protects your personal data.`,
}

export default function PrivacyPage() {
  return (
    <LegalPageLayout
      title="Privacy Policy"
      description={`${SITE.legalEntity} ("we", "us") operates ${SITE.name}. This policy explains what data we collect, why we collect it, and your rights.`}
    >
      <section>
        <h2>1. Who we are</h2>
        <p>
          Data controller: {SITE.legalEntity}, {SITE.address}. Contact:{' '}
          <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a>.
        </p>
      </section>

      <section>
        <h2>2. What we collect</h2>
        <h3>Account data</h3>
        <p>
          When you sign up via Clerk, we receive your user identifier, email address, and
          profile information you choose to provide. Authentication is handled by Clerk; see
          their privacy policy for identity-provider details.
        </p>
        <h3>Scan and usage data</h3>
        <ul>
          <li>URLs you submit for analysis and optional site-goal text</li>
          <li>Scan status, timestamps, session identifiers, and viewer URLs</li>
          <li>Captured page content, DOM snapshots, walkthrough video, and derived UX scores</li>
          <li>Technical logs (IP address, browser type, API request metadata)</li>
        </ul>
        <h3>Cookies and similar technologies</h3>
        <p>
          We use essential cookies for authentication and session management. See our{' '}
          <a href="/cookies">Cookie Policy</a> for details.
        </p>
      </section>

      <section>
        <h2>3. How we use your data</h2>
        <ul>
          <li>Provide, operate, and improve the {SITE.name} service</li>
          <li>Run website crawls and generate neural UX reports you request</li>
          <li>Authenticate users and prevent abuse (including SSRF and rate limiting)</li>
          <li>Respond to support requests and legal obligations</li>
          <li>Send service-related communications (not marketing without consent)</li>
        </ul>
        <p>
          <strong>Legal bases (EEA/UK users):</strong> contract performance, legitimate interests
          (security, product improvement), and consent where required.
        </p>
      </section>

      <section>
        <h2>4. How we share data</h2>
        <p>We do not sell your personal information. We share data only with:</p>
        <ul>
          <li>
            <strong>Infrastructure providers</strong> — hosting (Vercel, Railway/Render),
            storage (Cloudflare R2), database (PostgreSQL), queue (Redis), GPU inference (Modal)
          </li>
          <li>
            <strong>Clerk</strong> — authentication and user management
          </li>
          <li>
            <strong>Legal and safety</strong> — when required by law or to protect rights and
            security
          </li>
        </ul>
        <p>
          Submitted URLs are crawled by automated browsers. Only submit URLs you are authorized
          to analyze.
        </p>
      </section>

      <section>
        <h2>5. Retention</h2>
        <p>
          Scan metadata and artifacts are retained for up to <strong>30 days</strong> from scan
          creation unless you delete them sooner or a longer period is required by law or your
          agreement. Account data is retained while your account is active and for a reasonable
          period thereafter.
        </p>
      </section>

      <section>
        <h2>6. Security</h2>
        <p>
          We use industry-standard measures including TLS in transit, access controls, SSRF
          protections on URL intake, and isolated storage for scan artifacts. No method is 100%
          secure; report concerns to {SITE.supportEmail}.
        </p>
      </section>

      <section>
        <h2>7. International transfers</h2>
        <p>
          Your data may be processed in the United States and other countries where our
          providers operate. We rely on appropriate safeguards where required by applicable law.
        </p>
      </section>

      <section>
        <h2>8. Your rights</h2>
        <p>Depending on your location, you may have the right to:</p>
        <ul>
          <li>Access, correct, or delete your personal data</li>
          <li>Export your data</li>
          <li>Object to or restrict certain processing</li>
          <li>Withdraw consent where processing is consent-based</li>
          <li>Lodge a complaint with a supervisory authority</li>
        </ul>
        <p>
          Requests: <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a>. We respond
          within applicable legal timeframes.
        </p>
      </section>

      <section>
        <h2>9. Children</h2>
        <p>
          {SITE.name} is not directed to children under 16. We do not knowingly collect data from
          children. Contact us to request deletion if you believe a child has provided data.
        </p>
      </section>

      <section>
        <h2>10. Changes</h2>
        <p>
          We may update this policy. Material changes will be posted on this page with an updated
          date. Continued use after changes constitutes acceptance where permitted by law.
        </p>
      </section>
    </LegalPageLayout>
  )
}
