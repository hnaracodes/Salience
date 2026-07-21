'use client'

import Link from 'next/link'
import { SITE } from '@/lib/site'

function ArrowIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 12h14M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function CTASection() {
  return (
    <section data-section className="relative overflow-hidden border-t border-line bg-canvas px-6 py-32">
      <div className="relative mx-auto max-w-3xl text-center">
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ background: 'rgba(29,78,216,0.06)', filter: 'blur(60px)' }}
          aria-hidden="true"
        />

        <p className="section-eyebrow" data-reveal>
          Get started
        </p>

        <h2 className="section-heading relative mb-6 sm:text-5xl" data-reveal data-reveal-delay="0.05">
          Your first scan takes under two minutes.
        </h2>

        <p
          className="relative mb-10 text-lg leading-relaxed text-text-secondary"
          data-reveal
          data-reveal-delay="0.1"
        >
          Paste a URL, let {SITE.name} crawl and model cortical engagement, then explore
          heatmaps and per-element scores in the interactive viewer. Free to start — no
          credit card required.
        </p>

        <div
          className="relative flex flex-col items-center gap-4 sm:flex-row sm:justify-center"
          data-reveal
          data-reveal-delay="0.15"
        >
          <Link href="/scans/new" className="btn-primary">
            Start your first scan
            <ArrowIcon />
          </Link>
          <Link href="/dashboard" className="btn-secondary">
            Go to dashboard
          </Link>
        </div>
      </div>
    </section>
  )
}
