import type { Metadata } from 'next'
import { LegalPageLayout } from '@/components/legal/LegalPageLayout'
import { SITE } from '@/lib/site'

export const metadata: Metadata = {
  title: `Data Processing — ${SITE.name}`,
  description: `How ${SITE.name} stores, processes, displays, and retains scan data.`,
}

export default function DataProcessingPage() {
  return (
    <LegalPageLayout
      title="Data Processing"
      category="Data"
      description={`This page describes how ${SITE.name} handles data from URL submission through artifact storage, neural analysis, dashboard display, and deletion.`}
    >
      <section>
        <h2>1. Scope</h2>
        <p>
          This document covers scan-related data flows in the Salience SaaS product. For general
          privacy rights and legal bases, see our <a href="/privacy">Privacy Policy</a>. For
          security controls, see <a href="/security">Security</a>.
        </p>
      </section>

      <section>
        <h2>2. What you submit</h2>
        <p>When you create a scan, we process:</p>
        <ul>
          <li>
            <strong>Target URL</strong> — validated for SSRF safety before enqueue (http/https only,
            no private IPs or metadata endpoints)
          </li>
          <li>
            <strong>Site goal</strong> (optional) — free-text context used for copy-signal scoring
            and optional narrative generation
          </li>
          <li>
            <strong>Account identifier</strong> — your Clerk user ID, linked to a PostgreSQL user
            record
          </li>
          <li>
            <strong>Idempotency key</strong> (optional) — prevents duplicate jobs on retry
          </li>
        </ul>
      </section>

      <section>
        <h2>3. Capture and processing pipeline</h2>
        <p>
          After submission, an Arq worker picks up the job from Redis and runs the eight-stage
          pipeline in <code>services/pipeline/runner.py</code>:
        </p>
        <ol>
          <li>
            <strong>capture</strong> — Playwright explores the site (bounded scroll, same-origin
            guard, role-based locators). Produces walkthrough video (webm/mp4) and{' '}
            <code>session_manifest.json</code> with per–repetition-time DOM snapshots (element
            geometry, visible text, scroll state).
          </li>
          <li>
            <strong>tribe</strong> — walkthrough video is sent to Modal for TRIBE v2 inference
            (or synthetic zeros when <code>FAKE_TRIBE=1</code> in dev). Produces{' '}
            <code>preds.npz</code> — cortical activation time series (T × ~20,484 vertices on
            fsaverage5).
          </li>
          <li>
            <strong>dual_track</strong> — engagement, emotion, and activation scores merged into
            the analysis bundle.
          </li>
          <li>
            <strong>heatmaps</strong> — DeepGaze, an eye-tracking-trained saliency model, on an
            isolated local worker (or uniform placeholder in dev) for spatial UI grounding.
          </li>
          <li>
            <strong>analyze</strong> — parcellation, events, sections, marketing scores →{' '}
            <code>analysis_bundle.json</code>.
          </li>
          <li>
            <strong>copy_signals</strong> — heuristic clarity, urgency, goal_fit, copy_score from
            DOM text → <code>copy_signals.json</code>.
          </li>
          <li>
            <strong>narrative</strong> — optional template or LLM-generated marketing prose.
          </li>
          <li>
            <strong>export_viewer</strong> — static <code>ux_viewer/</code> HTML bundle synced to
            video and scores.
          </li>
        </ol>
        <p>
          <strong>Important:</strong> All neural and copy outputs are{' '}
          <strong>model-assisted hypotheses</strong>, not eye tracking, fMRI ground truth, or
          clinical measurements. Artifacts include schema versioning and provenance metadata
          (e.g. placeholder vs Modal heatmaps, heuristic copy source).
        </p>
      </section>

      <section>
        <h2>4. Where data is stored</h2>
        <h3>PostgreSQL (metadata)</h3>
        <ul>
          <li>Scan ID, user ID, URL, site goal, status, current stage, session ID</li>
          <li>Viewer URL, error messages, timestamps, 30-day expiry</li>
          <li>Artifact index rows linking to object storage keys</li>
        </ul>
        <h3>Object storage — Cloudflare R2 (production) or MinIO (local)</h3>
        <p>Artifacts are stored under <code>scans/&lt;scan_id&gt;/</code>:</p>
        <ul>
          <li>Walkthrough video (<code>walkthrough.webm</code> or <code>walkthrough.mp4</code>)</li>
          <li>DOM manifest (<code>session_manifest.json</code>)</li>
          <li>TRIBE predictions (<code>preds.npz</code>)</li>
          <li>Analysis bundle (<code>analysis_bundle.json</code>)</li>
          <li>Copy signals (<code>copy_signals.json</code>)</li>
          <li>Interactive viewer (<code>ux_viewer/</code> directory)</li>
          <li>Heatmap arrays and supporting frame assets as generated</li>
        </ul>
        <h3>Redis</h3>
        <p>Transient job queue state; not a long-term data store for scan content.</p>
        <h3>Modal</h3>
        <p>
          GPU workers process video and frames during inference. Data is sent for the duration of
          the job and handled under Modal&apos;s infrastructure policies.
        </p>
      </section>

      <section>
        <h2>5. How data is displayed</h2>
        <ul>
          <li>
            <strong>Dashboard</strong> — Next.js app shows scan list, status, errors, and links to
            the viewer. Authenticated via Clerk; API calls include your JWT.
          </li>
          <li>
            <strong>Viewer URL</strong> — presigned or public-prefix URL to{' '}
            <code>ux_viewer/index.html</code> (default presign expiry: 7 days for individual
            object access). The viewer embeds video, heatmaps, scores, and captured page content.
          </li>
          <li>
            <strong>Scan detail pages</strong> — may surface metadata and iframe the viewer.
          </li>
        </ul>
        <p>
          Anyone with a valid viewer URL may access the report until the presigned link expires or
          artifacts are deleted. Share links carefully.
        </p>
      </section>

      <section>
        <h2>6. Retention and deletion</h2>
        <ul>
          <li>
            Scans receive an <code>expires_at</code> timestamp of <strong>30 days</strong> from
            creation.
          </li>
          <li>
            You may delete a scan at any time via the API (<code>DELETE /v1/scans/&#123;id&#125;</code>
            ), which removes PostgreSQL records and all objects under{' '}
            <code>scans/&lt;scan_id&gt;/</code> in object storage.
          </li>
          <li>
            Account data persists while your Clerk account is active. Closing your account may
            require a separate request to purge associated metadata — contact{' '}
            <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a>.
          </li>
        </ul>
      </section>

      <section>
        <h2>7. Subprocessors</h2>
        <p>Scan data may be processed by:</p>
        <ul>
          <li>Clerk (authentication)</li>
          <li>Vercel (frontend hosting)</li>
          <li>Railway (API and worker hosting)</li>
          <li>Cloudflare R2 (artifact storage)</li>
          <li>PostgreSQL provider (metadata)</li>
          <li>Redis provider (job queue)</li>
          <li>Modal (GPU inference)</li>
        </ul>
        <p>We do not sell scan data or use it for third-party advertising.</p>
      </section>

      <section>
        <h2>8. Your responsibilities</h2>
        <p>
          Only submit URLs you are authorized to crawl. Captured content may include personal data
          visible on the target page (names, emails in DOM text, etc.). You are responsible for
          having a lawful basis to process that third-party content through {SITE.name}.
        </p>
      </section>

      <section>
        <h2>9. Contact</h2>
        <p>
          Data processing questions:{' '}
          <a href={`mailto:${SITE.privacyEmail}`}>{SITE.privacyEmail}</a>. Security incidents:{' '}
          <a href={`mailto:${SITE.securityEmail}`}>{SITE.securityEmail}</a>.
        </p>
      </section>
    </LegalPageLayout>
  )
}
