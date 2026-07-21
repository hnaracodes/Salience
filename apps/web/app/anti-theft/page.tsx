import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Intellectual Property & Anti-Theft — ${SITE.name}`,
  description: `IP protection, TRIBE v2 licensing, prohibited scraping, and DMCA notice process for ${SITE.name}.`,
}

export default function AntiTheftPage() {
  return (
    <LegalPageLayout
      title="Intellectual Property & Anti-Theft"
      category="IP"
      description={`This page describes ${SITE.legalEntity}'s intellectual property, third-party model licenses, prohibited misuse, and how to submit copyright complaints.`}
    >
      <section>
        <h2>1. Our intellectual property</h2>
        <p>
          {SITE.name}, the Salience brand, logos, website design, documentation, API design,
          pipeline orchestration, scoring fusion logic, viewer UX, and proprietary software are
          owned by {SITE.legalEntity} or its licensors and protected by copyright, trademark, and
          other intellectual property laws.
        </p>
        <p>
          Your subscription or account grants a limited, non-exclusive, non-transferable license to
          use the service as described in our <a href="/terms">Terms of Service</a>. No other
          rights are implied.
        </p>
      </section>

      <section>
        <h2>2. TRIBE v2 and third-party models</h2>
        <p>
          Cortical activation predictions are produced using{' '}
          <a
            href="https://github.com/facebookresearch/tribev2"
            rel="noopener noreferrer"
            target="_blank"
          >
            Facebook TRIBE v2
          </a>{' '}
          and related components (e.g. DINOv2 for heatmaps), each subject to their own licenses
          and terms. {SITE.name} does not grant you ownership of TRIBE weights, architecture, or
          underlying research artifacts.
        </p>
        <ul>
          <li>You may not extract, copy, or redistribute model weights from our infrastructure</li>
          <li>
            You may not use Salience outputs to train competing foundation models without
            authorization from all applicable rights holders
          </li>
          <li>
            Academic or research use of TRIBE itself must comply with Meta&apos;s license, separate
            from your Salience account
          </li>
        </ul>
      </section>

      <section>
        <h2>3. Prohibited scraping and reselling</h2>
        <p>Without prior written agreement, you must not:</p>
        <ul>
          <li>
            Scrape, crawl, or systematically download {SITE.name} dashboards, API responses, viewer
            bundles, or scan artifacts
          </li>
          <li>
            Reverse engineer the service to reconstruct our scoring pipeline, prompts, or model
            orchestration
          </li>
          <li>
            Resell, sublicense, white-label, or offer {SITE.name} reports or API access as your own
            product
          </li>
          <li>
            Use automated tools to bypass rate limits, authentication, or per-user scan isolation
          </li>
          <li>
            Mirror or republish our marketing content, documentation, or legal pages without
            permission
          </li>
        </ul>
        <p>
          Violations may result in immediate termination and legal action. See{' '}
          <a href="/acceptable-use">Acceptable Use Policy</a>.
        </p>
      </section>

      <section>
        <h2>4. Your content and scanned sites</h2>
        <p>
          You retain ownership of URLs and site content you submit. You grant us a limited license
          to crawl, process, store, and display derived artifacts solely to provide the service.
          If a scan captures third-party copyrighted material from a site you authorized us to
          analyze, you represent that you have the rights to do so.
        </p>
        <p>
          Scan reports are for your internal product and UX decisions. Redistribution of viewer
          links containing third-party site captures should respect the site owner&apos;s rights.
        </p>
      </section>

      <section>
        <h2>5. DMCA and copyright complaints</h2>
        <p>
          If you believe content stored or displayed through {SITE.name} infringes your copyright,
          send a notice to our designated agent:
        </p>
        <p>
          <strong>DMCA Agent</strong>
          <br />
          {SITE.legalEntity}
          <br />
          Email: <a href={`mailto:${SITE.contactEmail}`}>{SITE.contactEmail}</a>
          <br />
          Subject line: &quot;DMCA Notice — {SITE.name}&quot;
        </p>
        <p>Your notice should include:</p>
        <ul>
          <li>Identification of the copyrighted work claimed to be infringed</li>
          <li>
            Identification of the material (e.g. scan ID, viewer URL, submitted URL) and sufficient
            information to locate it
          </li>
          <li>Your contact information (address, phone, email)</li>
          <li>
            A statement of good-faith belief that use is not authorized by the copyright owner
          </li>
          <li>
            A statement, under penalty of perjury, that the information is accurate and you are
            authorized to act on behalf of the owner
          </li>
          <li>Your physical or electronic signature</li>
        </ul>
        <p>
          We may remove or disable access to disputed material and notify the submitting user.
          Counter-notices may be submitted in accordance with 17 U.S.C. § 512(g).
        </p>
      </section>

      <section>
        <h2>6. Unauthorized scans of your website</h2>
        <p>
          If someone scanned your website without permission, contact{' '}
          <a href={`mailto:${SITE.securityEmail}`}>{SITE.securityEmail}</a> with the URL, scan
          details if known, and proof of ownership or authority. We investigate abuse reports under
          our <a href="/acceptable-use">Acceptable Use Policy</a> and may suspend accounts or delete
          artifacts.
        </p>
      </section>

      <section>
        <h2>7. Trademarks</h2>
        <p>
          &quot;Salience,&quot; the Salience logo, and related marks are trademarks of{' '}
          {SITE.legalEntity}. TRIBE and other third-party names are property of their respective
          owners. Use of our marks requires prior written consent except for factual reference to
          our service.
        </p>
      </section>

      <section>
        <h2>8. Related documents</h2>
        <p>
          <a href="/terms">Terms of Service</a> · <a href="/privacy">Privacy Policy</a> ·{' '}
          <a href="/security">Security</a> · <a href="/contact">Contact</a>
        </p>
      </section>
    </LegalPageLayout>
  )
}
