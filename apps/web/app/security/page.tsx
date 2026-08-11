import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Security — ${SITE.name}`,
  description: `How ${SITE.name} protects your data: authentication, encryption, SSRF controls, and storage.`,
}

export default function SecurityPage() {
  return (
    <LegalPageLayout
      title="Security"
      category="Security"
      description={`${SITE.name} is built on Clerk authentication, a FastAPI backend, PostgreSQL metadata, Redis job queues, and Cloudflare R2 object storage. This page summarizes our security practices.`}
    >
      <section>
        <h2>1. Overview</h2>
        <p>
          Security is embedded in how Salience accepts URLs, queues work, stores artifacts, and
          serves reports. No system is perfectly secure; we continuously improve controls and
          welcome responsible disclosure. Report vulnerabilities to{' '}
          <a href={`mailto:${SITE.securityEmail}`}>{SITE.securityEmail}</a>.
        </p>
      </section>

      <section>
        <h2>2. Authentication and authorization</h2>
        <ul>
          <li>
            <strong>Clerk</strong> handles user sign-up, sign-in, and session management on the
            Next.js frontend (hosted on Vercel).
          </li>
          <li>
            <strong>JWT validation (RS256)</strong> — the FastAPI backend verifies Clerk-issued
            Bearer tokens against Clerk&apos;s JWKS endpoint. Tokens are checked for signature,
            expiry, and issuer. JWKS keys are cached in-process with a one-hour TTL and refreshed
            on key rotation.
          </li>
          <li>
            <strong>Per-user isolation</strong> — scan list, status, and delete operations are
            scoped to the authenticated Clerk user ID. You cannot access another user&apos;s scans
            via the API.
          </li>
        </ul>
      </section>

      <section>
        <h2>3. Encryption and transport</h2>
        <ul>
          <li>
            <strong>TLS in transit</strong> — production traffic between your browser, Vercel,
            the API (Railway), and Cloudflare R2 uses HTTPS.
          </li>
          <li>
            <strong>Object storage</strong> — scan artifacts (walkthrough video, DOM manifest,
            preds.npz, analysis_bundle, copy_signals, ux_viewer) are stored in S3-compatible
            buckets (R2 in production, MinIO locally). Access uses scoped credentials; viewer URLs
            are presigned or served via a configured public prefix with time-limited signatures
            where applicable.
          </li>
          <li>
            <strong>Database</strong> — PostgreSQL holds scan metadata (URL, status, session ID,
            viewer URL, expiry). Connection strings and secrets are environment variables, not
            committed to source control.
          </li>
        </ul>
      </section>

      <section>
        <h2>4. SSRF protection on URL intake</h2>
        <p>
          Before any crawl is enqueued, submitted URLs pass through{' '}
          <code>services/api/ssrf.py</code> validation:
        </p>
        <ul>
          <li>Only <strong>http</strong> and <strong>https</strong> schemes are allowed</li>
          <li>URLs with embedded credentials (<code>user:pass@host</code>) are rejected</li>
          <li>
            Hostnames are DNS-resolved; all resolved IPs are checked against forbidden ranges
          </li>
          <li>
            Blocked: private networks (RFC 1918), loopback, link-local, multicast, unspecified
            addresses, and cloud metadata endpoints (e.g. 169.254.169.254, 100.100.100.200)
          </li>
          <li>
            Redirect hops during crawling should be re-validated to prevent SSRF via redirect
            chains
          </li>
        </ul>
        <p>
          Local development may set <code>SSRF_ALLOW_LOCALHOST=1</code> for fixture servers;
          this flag must never be enabled in production.
        </p>
      </section>

      <section>
        <h2>5. Rate limiting and abuse prevention</h2>
        <ul>
          <li>
            <strong>Concurrent scan limit</strong> — free tier allows one active (queued or
            running) scan per user; additional submissions receive HTTP 429.
          </li>
          <li>
            <strong>Idempotency keys</strong> — optional keys prevent duplicate scan creation on
            retries.
          </li>
          <li>
            <strong>CORS</strong> — the API restricts browser origins to configured allowlists
            (e.g. the Vercel frontend URL).
          </li>
        </ul>
        <p>
          See our <a href="/acceptable-use">Acceptable Use Policy</a> for prohibited conduct.
        </p>
      </section>

      <section>
        <h2>6. Processing infrastructure</h2>
        <ul>
          <li>
            <strong>Arq worker + Redis</strong> — scan jobs are queued and processed in isolated
            worker containers running Playwright capture and the analysis pipeline.
          </li>
          <li>
            <strong>Modal GPU</strong> — TRIBE v2 inference runs on Modal; video and manifest data
            are sent to Modal only for jobs you submit. Attention heatmaps run on an isolated
            DeepGaze worker (Modal DINOv2 remains available as a rollback backend).
          </li>
          <li>
            <strong>Artifact deletion</strong> — deleting a scan removes PostgreSQL records and
            purges objects under <code>scans/&lt;scan_id&gt;/</code> in object storage.
          </li>
        </ul>
      </section>

      <section>
        <h2>7. Viewer and content safety</h2>
        <p>
          Exported viewer HTML is a trusted template from our repository. When optional HTML
          sanitization is applied during upload, inline script tags in untrusted content are
          neutralized as defense-in-depth. Viewer assets load via relative paths from the viewer
          URL origin.
        </p>
      </section>

      <section>
        <h2>8. Retention</h2>
        <p>
          Scan metadata and artifacts are retained for up to <strong>30 days</strong> from creation
          (see <code>expires_at</code> on scan records) unless deleted sooner. Details:{' '}
          <a href="/data-processing">Data Processing</a>.
        </p>
      </section>

      <section>
        <h2>9. Enterprise and compliance</h2>
        <p>
          These practices are designed for standard SaaS operation. Regulated industries,
          healthcare, or enterprise procurement may require a Data Processing Agreement (DPA),
          SOC 2 reports, or customized retention — contact{' '}
          <a href={`mailto:${SITE.contactEmail}`}>{SITE.contactEmail}</a>. Have counsel review
          our policies before relying on them for compliance certifications we have not explicitly
          obtained.
        </p>
      </section>

      <section>
        <h2>10. Related pages</h2>
        <p>
          <a href="/privacy">Privacy Policy</a> · <a href="/data-processing">Data Processing</a>{' '}
          · <a href="/acceptable-use">Acceptable Use</a> ·{' '}
          <a href="/anti-theft">Intellectual Property</a>
        </p>
      </section>
    </LegalPageLayout>
  )
}
