import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Acceptable Use Policy — ${SITE.name}`,
  description: `Rules for submitting URLs and using the ${SITE.name} platform responsibly.`,
}

export default function AcceptableUsePage() {
  return (
    <LegalPageLayout
      title="Acceptable Use Policy"
      description={`This policy describes permitted and prohibited use of ${SITE.name}. It supplements our Terms of Service and applies to all users.`}
    >
      <section>
        <h2>1. Purpose</h2>
        <p>
          {SITE.name} crawls websites with automated browsers (Playwright), runs neural UX
          analysis (TRIBE v2 and related models on Modal GPU), and stores scan artifacts in
          object storage. Because the service interacts with third-party sites and shared
          infrastructure, you must use it responsibly and only with proper authorization.
        </p>
      </section>

      <section>
        <h2>2. URL submission rules</h2>
        <p>You may submit a URL for analysis only if:</p>
        <ul>
          <li>You own the site or have explicit permission from the site owner to crawl and analyze it</li>
          <li>
            The URL is publicly reachable over HTTP or HTTPS and passes our SSRF validation (see{' '}
            <a href="/security">Security</a>)
          </li>
          <li>
            Crawling the site does not violate its terms of use, robots.txt restrictions you are
            bound to honor, or applicable law
          </li>
          <li>
            You do not use the service to probe, scan, or attack networks, internal systems, or
            cloud metadata endpoints
          </li>
        </ul>
        <p>
          Our API rejects URLs that resolve to private IP ranges, loopback addresses, link-local
          addresses, cloud metadata hosts (e.g. 169.254.169.254), non-http(s) schemes, or URLs
          containing embedded credentials.
        </p>
      </section>

      <section>
        <h2>3. Prohibited conduct</h2>
        <p>You must not:</p>
        <ul>
          <li>Submit URLs you are not authorized to analyze, including competitors&apos; sites without permission</li>
          <li>
            Attempt to bypass authentication, SSRF protections, rate limits, or CORS restrictions
          </li>
          <li>
            Run concurrent scans beyond your tier limits (free tier: one active scan at a time)
          </li>
          <li>
            Use automated scripts to scrape, bulk-enqueue, or resell access to {SITE.name} without
            a written agreement
          </li>
          <li>
            Reverse engineer, decompile, or extract model weights, TRIBE outputs, or proprietary
            scoring logic except where legally permitted — see{' '}
            <a href="/anti-theft">Intellectual Property &amp; Anti-Theft</a>
          </li>
          <li>
            Upload malware, exploit payloads, or content designed to compromise our workers,
            Modal jobs, or storage systems
          </li>
          <li>
            Harass, impersonate, or infringe the rights of others through shared scan reports or
            viewer links
          </li>
          <li>Use the service for unlawful surveillance, stalking, or discrimination</li>
        </ul>
      </section>

      <section>
        <h2>4. Scan outputs and limitations</h2>
        <p>
          Reports, heatmaps, copy signals, and cortical activation proxies are{' '}
          <strong>model-assisted hypotheses</strong> — not eye tracking, fMRI, medical diagnoses,
          or guaranteed business outcomes. Do not represent Salience outputs as clinical
          measurements or certified user research without appropriate disclosure and validation.
        </p>
      </section>

      <section>
        <h2>5. Shared reports and viewer links</h2>
        <p>
          Scan viewer URLs may expose captured page content, DOM snapshots, and derived scores.
          Treat share links like confidential business documents. Do not publish viewer URLs where
          unauthorized parties could access sensitive material.
        </p>
      </section>

      <section>
        <h2>6. Rate limits and fair use</h2>
        <p>
          We enforce per-account limits to protect shared infrastructure (Redis job queue, worker
          containers, Modal GPU). Excessive or abusive usage may result in throttling, suspension,
          or termination. Enterprise customers may request higher limits under a separate agreement.
        </p>
      </section>

      <section>
        <h2>7. Enforcement</h2>
        <p>
          We may investigate suspected violations, remove content, cancel scans, suspend accounts,
          and report illegal activity to authorities. Repeated or egregious violations may result
          in permanent termination without refund where permitted by law.
        </p>
      </section>

      <section>
        <h2>8. Reporting abuse</h2>
        <p>
          To report misuse of {SITE.name} or unauthorized scans of your website:{' '}
          <a href={`mailto:${SITE.securityEmail}`}>{SITE.securityEmail}</a>. For copyright
          concerns, see our <a href="/anti-theft">DMCA process</a>.
        </p>
      </section>

      <section>
        <h2>9. Related documents</h2>
        <p>
          <a href="/terms">Terms of Service</a> · <a href="/privacy">Privacy Policy</a> ·{' '}
          <a href="/security">Security</a> · <a href="/data-processing">Data Processing</a>
        </p>
      </section>
    </LegalPageLayout>
  )
}
