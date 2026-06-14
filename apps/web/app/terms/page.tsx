import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Terms of Service — ${SITE.name}`,
  description: `Terms governing use of the ${SITE.name} platform.`,
}

export default function TermsPage() {
  return (
    <LegalPageLayout
      title="Terms of Service"
      description={`These Terms govern your access to and use of ${SITE.name}, operated by ${SITE.legalEntity}.`}
    >
      <section>
        <h2>1. Agreement</h2>
        <p>
          By creating an account, submitting a URL, or otherwise using {SITE.name}, you agree to
          these Terms, our <a href="/privacy">Privacy Policy</a>,{' '}
          <a href="/acceptable-use">Acceptable Use Policy</a>, and{' '}
          <a href="/cookies">Cookie Policy</a>. If you use the service on behalf of an
          organization, you represent that you have authority to bind that organization.
        </p>
      </section>

      <section>
        <h2>2. The service</h2>
        <p>
          {SITE.name} provides automated website crawling, neural UX analysis, heatmaps, copy
          signals, and interactive reports. Outputs are <strong>informational and
          model-based</strong> — not medical, legal, or eye-tracking measurements. You are
          responsible for how you use reports in product or business decisions.
        </p>
      </section>

      <section>
        <h2>3. Accounts</h2>
        <ul>
          <li>You must provide accurate registration information and keep credentials secure.</li>
          <li>You are responsible for activity under your account.</li>
          <li>We may suspend or terminate accounts that violate these Terms or pose risk.</li>
        </ul>
      </section>

      <section>
        <h2>4. Subscriptions and fees</h2>
        <p>
          Free and paid tiers may be offered. Paid plans, billing cycles, and refunds will be
          described at checkout or in an order form. Unless stated otherwise, fees are
          non-refundable except where required by law. We may change pricing with reasonable notice.
        </p>
      </section>

      <section>
        <h2>5. Your content and URLs</h2>
        <p>
          You retain ownership of URLs and site content you submit. You grant us a limited license
          to crawl, process, store, and display derived artifacts solely to provide the service.
          You warrant that you have the right to submit each URL and that doing so does not violate
          third-party rights or applicable law.
        </p>
      </section>

      <section>
        <h2>6. Acceptable use</h2>
        <p>
          You must comply with our <a href="/acceptable-use">Acceptable Use Policy</a>. Prohibited
          conduct includes scanning sites without authorization, attacking our infrastructure,
          circumventing rate limits, and reselling access without agreement.
        </p>
      </section>

      <section>
        <h2>7. Intellectual property</h2>
        <p>
          We own the platform, software, branding, and documentation. TRIBE v2 and related models
          are subject to their respective third-party licenses. You may not reverse engineer the
          service except where legally permitted.
        </p>
      </section>

      <section>
        <h2>8. Confidentiality and security</h2>
        <p>
          Scan reports may contain sensitive business information. Use share links carefully.
          Enterprise customers may request a separate data processing agreement (DPA).
        </p>
      </section>

      <section>
        <h2>9. Disclaimers</h2>
        <p>
          THE SERVICE IS PROVIDED &quot;AS IS&quot; AND &quot;AS AVAILABLE&quot; WITHOUT WARRANTIES
          OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR
          PURPOSE, AND NON-INFRINGEMENT. WE DO NOT GUARANTEE UNINTERRUPTED OR ERROR-FREE OPERATION
          OR ACCURACY OF NEURAL OR COPY SCORES.
        </p>
      </section>

      <section>
        <h2>10. Limitation of liability</h2>
        <p>
          TO THE MAXIMUM EXTENT PERMITTED BY LAW, {SITE.legalEntity.toUpperCase()} AND ITS
          SUPPLIERS SHALL NOT BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR
          PUNITIVE DAMAGES, OR ANY LOSS OF PROFITS, DATA, OR GOODWILL. OUR AGGREGATE LIABILITY
          FOR ANY CLAIM ARISING FROM THESE TERMS OR THE SERVICE SHALL NOT EXCEED THE GREATER OF
          (A) AMOUNTS YOU PAID US IN THE TWELVE MONTHS BEFORE THE CLAIM OR (B) ONE HUNDRED U.S.
          DOLLARS ($100).
        </p>
      </section>

      <section>
        <h2>11. Indemnification</h2>
        <p>
          You will indemnify and hold us harmless from claims arising from your use of the service,
          URLs you submit, or violation of these Terms.
        </p>
      </section>

      <section>
        <h2>12. Termination</h2>
        <p>
          You may stop using the service at any time. We may suspend or terminate access for
          breach, risk, or discontinuation. Upon termination, your right to use the service ends;
          provisions that should survive (liability limits, indemnity, governing law) remain in
          effect.
        </p>
      </section>

      <section>
        <h2>13. Governing law</h2>
        <p>
          These Terms are governed by the laws of the State of Delaware, USA, excluding conflict-of-law
          rules. Disputes shall be resolved in the state or federal courts located in Delaware,
          unless mandatory consumer protection laws in your jurisdiction provide otherwise.
        </p>
      </section>

      <section>
        <h2>14. Contact</h2>
        <p>
          Questions about these Terms:{' '}
          <a href={`mailto:${SITE.contactEmail}`}>{SITE.contactEmail}</a>.
        </p>
      </section>
    </LegalPageLayout>
  )
}
