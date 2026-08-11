'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import Link from 'next/link'
import { SignedIn, SignedOut, SignUpButton } from '@clerk/nextjs'
import { HeroBackdrop } from '@/components/HeroBackdrop'
import { HeroProductPanel } from '@/components/HeroProductPanel'
import { clerkEnabled } from '@/lib/auth'
import { SITE, SOCIAL_PROOF } from '@/lib/site'

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

export function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null)

  useGSAP(
    () => {
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      const section = sectionRef.current
      if (!section || reducedMotion) return

      const copyEls = section.querySelectorAll('[data-hero-copy]')
      gsap.fromTo(
        copyEls,
        { opacity: 0, y: 36 },
        {
          opacity: 1,
          y: 0,
          duration: 1.25,
          stagger: 0.12,
          ease: 'power4.out',
          delay: 0.1,
        }
      )
    },
    { scope: sectionRef }
  )

  return (
    <section
      ref={sectionRef}
      id="hero"
      data-section
      className="relative min-h-screen overflow-hidden px-6 pb-24 pt-32 md:pt-36"
    >
      <HeroBackdrop />

      <div className="relative z-10 mx-auto grid max-w-6xl items-center gap-14 lg:grid-cols-2 lg:gap-16">
        <div className="text-center lg:text-left">
          <p className="section-eyebrow" data-hero-copy>
            {SITE.tagline}
          </p>

          <h1
            className="section-heading max-w-xl sm:text-5xl lg:text-[3.25rem]"
            data-hero-copy
          >
            <span className="text-signal">{SITE.name}</span>{' '}
            {SITE.headline}
          </h1>

          <p
            className="mt-6 max-w-lg text-base leading-relaxed text-text-secondary sm:text-lg"
            data-hero-copy
          >
            {SITE.description}
          </p>

          <div
            className="mt-9 flex flex-col items-center gap-3 sm:flex-row lg:justify-start"
            data-hero-copy
          >
            {clerkEnabled ? <HeroAuthCta /> : <HeroStaticCta />}
            <a href="#product-journey" className="btn-secondary">
              See the pipeline
            </a>
          </div>

          <ul
            className="mt-10 flex flex-wrap justify-center gap-x-8 gap-y-3 text-sm text-text-secondary lg:justify-start"
            data-hero-copy
          >
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-copy" />
              Up to 8 pages per scan
            </li>
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-neural" />
              TRIBE v2 cortical scores
            </li>
            <li className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-signal" />
              DeepGaze heatmaps + copy fusion
            </li>
          </ul>
        </div>

        <div className="relative flex justify-center lg:justify-end" data-hero-copy>
          <HeroProductPanel />
        </div>
      </div>

      <div
        className="relative z-10 mx-auto mt-20 max-w-6xl border-t border-line pt-10"
        data-hero-copy
      >
        <p className="mb-6 text-center text-xs font-medium uppercase tracking-[0.2em] text-text-tertiary">
          Used by product teams at
        </p>
        <ul className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4">
          {SOCIAL_PROOF.map((name) => (
            <li
              key={name}
              className="text-sm font-semibold tracking-tight text-text-tertiary/70 transition-colors duration-500 hover:text-text-secondary"
            >
              {name}
            </li>
          ))}
        </ul>
      </div>

      <div className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2" aria-hidden="true">
        <div className="mx-auto h-8 w-px bg-line-strong" />
        <div className="mt-1.5 mx-auto h-1.5 w-1.5 rounded-full bg-ink-950 animate-pulse-signal" />
      </div>
    </section>
  )
}

function HeroAuthCta() {
  return (
    <>
      <SignedIn>
        <Link href="/scans/new" className="btn-primary">
          Start free scan
          <ArrowIcon />
        </Link>
      </SignedIn>
      <SignedOut>
        <SignUpButton mode="modal">
          <button type="button" className="btn-primary">
            Start free scan
            <ArrowIcon />
          </button>
        </SignUpButton>
      </SignedOut>
    </>
  )
}

function HeroStaticCta() {
  return (
    <Link href="/?auth=required" className="btn-primary">
      Start free scan
      <ArrowIcon />
    </Link>
  )
}
