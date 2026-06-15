'use client'

import Link from 'next/link'
import { SITE } from '@/lib/site'

export function CTASection() {
  return (
    <section className="relative overflow-hidden border-t border-ink-700 bg-ink-950 px-6 py-32">
      <div className="relative mx-auto max-w-3xl text-center">
        {/* Signal glow — restrained, single-color */}
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-64 w-64 rounded-full"
          style={{ background: 'rgba(37,99,255,0.08)', filter: 'blur(60px)' }}
          aria-hidden="true"
        />

        <p
          className="mb-4 text-xs font-mono uppercase tracking-widest text-text-secondary"
          data-reveal
        >
          Get started
        </p>

        <h2
          className="relative mb-6 text-4xl font-display font-semibold tracking-tight text-text-primary md:text-5xl"
          style={{ letterSpacing: '-0.02em' }}
          data-reveal
          data-reveal-delay="0.05"
        >
          Start measuring{' '}
          <span className="text-signal">salience.</span>
        </h2>

        <p
          className="relative mb-10 text-lg text-text-secondary"
          data-reveal
          data-reveal-delay="0.1"
        >
          Free to start with {SITE.name}. No credit card required. First scan in under two minutes.
        </p>

        <div
          className="relative flex flex-col items-center gap-4 sm:flex-row sm:justify-center"
          data-reveal
          data-reveal-delay="0.15"
        >
          <Link
            href="/scans/new"
            className="inline-flex h-12 items-center rounded-md bg-signal px-10 text-sm font-medium text-white transition-colors hover:bg-signal-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          >
            Start your first scan
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex h-12 items-center rounded-md border border-ink-700 px-8 text-sm font-medium text-text-secondary transition-colors hover:border-ink-600 hover:text-text-primary"
          >
            Go to dashboard
          </Link>
        </div>
      </div>
    </section>
  )
}
