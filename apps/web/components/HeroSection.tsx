'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import Link from 'next/link'
import { SignedIn, SignedOut, SignUpButton } from '@clerk/nextjs'
import { Logo } from '@/components/Logo'
import { HeroBackdrop } from '@/components/HeroBackdrop'
import { HeroProductPanel } from '@/components/HeroProductPanel'
import { clerkEnabled } from '@/lib/auth'
import { SITE } from '@/lib/site'

export function HeroSection() {
  const glowRef = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      const reducedMotion = window.matchMedia(
        '(prefers-reduced-motion: reduce)'
      ).matches

      if (!glowRef.current) return

      if (reducedMotion) {
        gsap.set(glowRef.current, { opacity: 1 })
        return
      }

      gsap.fromTo(
        glowRef.current,
        { opacity: 0 },
        { opacity: 1, duration: 1.5, ease: 'power2.out' }
      )
    },
    { scope: glowRef }
  )

  return (
    <section
      id="hero"
      className="relative min-h-screen overflow-hidden px-6 pb-20 pt-28 md:pt-32"
    >
      <HeroBackdrop />

      <div
        ref={glowRef}
        className="pointer-events-none absolute inset-0 opacity-0"
        style={{
          background:
            'radial-gradient(ellipse 50% 40% at 30% 45%, rgba(37,99,255,0.08) 0%, transparent 65%)',
        }}
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto grid max-w-6xl items-center gap-12 lg:grid-cols-2 lg:gap-16">
        {/* Copy column */}
        <div className="text-center lg:text-left">
          <div className="mb-8 flex justify-center lg:justify-start" data-reveal>
            <Logo variant="hero" size={120} />
          </div>

          <p
            className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-signal"
            data-reveal
            data-reveal-delay="0.05"
          >
            {SITE.tagline}
          </p>

          <h1
            className="max-w-xl text-4xl font-display font-semibold leading-[1.08] tracking-tight text-text-primary sm:text-5xl lg:text-[3.25rem]"
            style={{ letterSpacing: '-0.02em' }}
            data-reveal
            data-reveal-delay="0.1"
          >
            <span className="text-signal">{SITE.name}</span>{' '}
            maps what your users notice.
          </h1>

          <p
            className="mt-6 max-w-lg text-base text-text-secondary sm:text-lg"
            data-reveal
            data-reveal-delay="0.15"
          >
            {SITE.description}
          </p>

          <div
            className="mt-8 flex flex-col items-center gap-3 sm:flex-row lg:justify-start"
            data-reveal
            data-reveal-delay="0.2"
          >
            {clerkEnabled ? <HeroAuthCta /> : <HeroStaticCta />}
            <a
              href="#product-journey"
              className="inline-flex h-12 items-center rounded-md border border-ink-600 px-7 text-sm font-medium text-text-secondary transition-colors hover:border-signal/50 hover:text-text-primary"
            >
              Scroll the pipeline
            </a>
          </div>

          {/* Trust strip */}
          <ul
            className="mt-10 flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs font-mono text-text-secondary lg:justify-start"
            data-reveal
            data-reveal-delay="0.25"
          >
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-copy" />
              Multi-page crawl
            </li>
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-neural" />
              DINOv2 heatmaps
            </li>
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-signal" />
              Copy + neural fusion
            </li>
          </ul>
        </div>

        {/* Visual column */}
        <div className="relative flex justify-center lg:justify-end">
          <HeroProductPanel />
        </div>
      </div>

      <div
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2"
        aria-hidden="true"
      >
        <div className="h-8 w-px bg-ink-600 mx-auto" />
        <div className="mt-1 h-1.5 w-1.5 rounded-full bg-signal mx-auto animate-pulse-signal" />
      </div>
    </section>
  )
}

const primaryCtaClass =
  'inline-flex h-12 items-center rounded-md bg-signal px-8 text-sm font-medium text-white transition-colors hover:bg-signal-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal'

function HeroAuthCta() {
  return (
    <>
      <SignedIn>
        <Link href="/scans/new" className={primaryCtaClass}>
          Start free scan
        </Link>
      </SignedIn>
      <SignedOut>
        <SignUpButton mode="modal">
          <button type="button" className={primaryCtaClass}>
            Sign up to scan
          </button>
        </SignUpButton>
      </SignedOut>
    </>
  )
}

function HeroStaticCta() {
  return (
    <Link href="/?auth=required" className={primaryCtaClass}>
      Sign up to scan
    </Link>
  )
}
