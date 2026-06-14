import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Acceptable Use Policy — ${SITE.name}`,
  description: `Rules for acceptable use of the ${SITE.name} platform.`,
}

export default function AcceptableUsePage() {
  return (
    <LegalPageLayout
      title="Acceptable Use Policy"
      description={`This policy defines permitted and prohibited uses of ${SITE.name}. It supplements our Terms of Service.`}
    >
      <section>
        <h2>1. Permitted use</h2>
        <ul>
          <li>Analyze websites you own or have explicit written authorization to test</li>
          <li>Use reports for UX research, product improvement, and client work under contract</li>
          <li>Share viewer links with teammates and clients who are authorized to see the data</li>
          <li>Comply with robots.txt and applicable crawling norms where relevant</li>
        </ul>
      </section>

      <section>
        <h2>2. Prohibited use</h2>
        <p>You may not use {SITE.name} to:</p>
        <ul>
          <li>Scan sites without permission (competitor espionage, unauthorized audits)</li>
          <li>Target internal networks, localhost, private IPs, or metadata endpoints (SSRF)</li>
          <li>Submit malware, phishing pages, or illegal content URLs</li>
          <li>Overload or disrupt our systems (DDoS, queue flooding, credential stuffing)</li>
          <li>Bypass authentication, rate limits, or plan restrictions</li>
          <li>Resell or sublicense the platform without a written agreement</li>
          <li>Reverse engineer models or extract training data beyond normal report export</li>
          <li>Harass, discriminate, or violate privacy rights of individuals</li>
          <li>Violate export control, sanctions, or applicable law</li>
        </ul>
      </section>

      <section>
        <h2>3. URL submission rules</h2>
        <p>
          Only <strong>public HTTP/HTTPS URLs</strong> on hosts you control or are engaged to
          analyze. Our SSRF protections block private addresses and dangerous schemes — attempts
          to circumvent protections may result in immediate suspension.
        </p>
      </section>

      <section>
        <h2>4. Rate limits and fair use</h2>
        <p>
          Free tiers may limit concurrent scans and monthly volume. Automated or high-volume use
          requires an appropriate paid plan. We may throttle or suspend accounts that degrade
          service for others.
        </p>
      </section>

      <section>
        <h2>5. Enforcement</h2>
        <p>
          Violations may result in warning, scan cancellation, account suspension, or permanent
          termination without refund. We may report illegal activity to authorities.
        </p>
      </section>

      <section>
        <h2>6. Reporting abuse</h2>
        <p>
          Report misuse to{' '}
          <a href={`mailto:${SITE.supportEmail}`}>{SITE.supportEmail}</a> with URL, account
          details, and evidence where available.
        </p>
      </section>
    </LegalPageLayout>
  )
}
