'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { SignInButton, SignUpButton, UserButton, useUser } from '@clerk/nextjs'
import { Logo } from '@/components/Logo'
import { clerkEnabled } from '@/lib/auth'
import { NAV_LINKS, SITE } from '@/lib/site'
import clsx from 'clsx'

interface NavBarProps {
  variant?: 'landing' | 'app'
}

export function NavBar({ variant = 'landing' }: NavBarProps) {
  const [scrolled, setScrolled] = useState(variant === 'app')
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    if (variant === 'app') return
    const handler = () => setScrolled(window.scrollY > 24)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [variant])

  useEffect(() => {
    if (!mobileOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false)
    }
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = ''
      window.removeEventListener('keydown', onKey)
    }
  }, [mobileOpen])

  const shellClass = clsx(
    variant === 'landing'
      ? 'fixed top-4 left-0 right-0 z-50 flex justify-center px-4 transition-all duration-500 ease-premium relative'
      : 'fixed top-0 z-50 w-full border-b border-line bg-surface/95 backdrop-blur-md'
  )

  const barClass = clsx(
    'flex w-full items-center justify-between gap-4 transition-all duration-500 ease-premium',
    variant === 'landing'
      ? clsx(
          'h-14 max-w-5xl rounded-pill border bg-surface/90 px-5 shadow-nav backdrop-blur-md',
          scrolled ? 'border-line shadow-card' : 'border-line/80'
        )
      : 'mx-auto h-14 max-w-6xl px-6'
  )

  return (
    <nav className={shellClass}>
      <div className={barClass}>
        <Link href="/" className="group flex shrink-0 items-center gap-2.5">
          <Logo variant="nav" size={28} />
          <span className="text-sm font-semibold tracking-tight text-text-primary">
            {SITE.name}
          </span>
        </Link>

        <div className="hidden items-center gap-0.5 md:flex">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="cursor-pointer rounded-pill px-3.5 py-1.5 text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary"
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <div className="hidden items-center gap-3 sm:flex">
            {clerkEnabled ? <NavBarAuthControls /> : <NavBarStaticControls />}
          </div>

          <button
            type="button"
            className="inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-pill border border-line text-text-secondary transition-colors duration-300 hover:border-line-strong hover:text-text-primary md:hidden"
            aria-expanded={mobileOpen}
            aria-controls="mobile-nav"
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
            onClick={() => setMobileOpen((open) => !open)}
          >
            {mobileOpen ? <CloseIcon /> : <MenuIcon />}
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div
          id="mobile-nav"
          className="absolute left-4 right-4 top-[calc(100%+0.5rem)] overflow-hidden rounded-card border border-line bg-surface shadow-card-lg md:hidden"
        >
          <div className="px-4 py-4">
            <ul className="space-y-1">
              {NAV_LINKS.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="block cursor-pointer rounded-lg px-3 py-2.5 text-sm text-text-secondary transition-colors duration-300 hover:bg-surface-muted hover:text-text-primary"
                    onClick={() => setMobileOpen(false)}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>

            <div className="mt-4 flex flex-col gap-2 border-t border-line pt-4 sm:hidden">
              {clerkEnabled ? (
                <NavBarAuthControls mobile onNavigate={() => setMobileOpen(false)} />
              ) : (
                <NavBarStaticControls mobile />
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}

function MenuIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 7h16M4 12h16M4 17h16"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M6 6l12 12M18 6L6 18"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

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

function NavBarAuthControls({
  mobile = false,
  onNavigate,
}: {
  mobile?: boolean
  onNavigate?: () => void
}) {
  const { isSignedIn } = useUser()

  if (isSignedIn) {
    return (
      <>
        <Link
          href="/dashboard"
          className={clsx(
            'text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary',
            mobile && 'rounded-lg px-3 py-2.5 hover:bg-surface-muted'
          )}
          onClick={onNavigate}
        >
          Dashboard
        </Link>
        <UserButton afterSignOutUrl="/" />
      </>
    )
  }

  return (
    <>
      <SignInButton mode="modal">
        <button
          className={clsx(
            'cursor-pointer px-3 py-1.5 text-sm text-text-secondary transition-colors duration-300 hover:text-text-primary',
            mobile && 'w-full rounded-lg py-2.5 text-left hover:bg-surface-muted'
          )}
        >
          Sign in
        </button>
      </SignInButton>
      <SignUpButton mode="modal">
        <button
          className={clsx(
            'inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-pill bg-ink-950 px-4 text-sm font-medium text-white transition-all duration-300 ease-premium hover:bg-ink-900 hover:shadow-card',
            mobile && 'h-11 w-full justify-center'
          )}
        >
          Get started
          <ArrowIcon />
        </button>
      </SignUpButton>
    </>
  )
}

function NavBarStaticControls({ mobile = false }: { mobile?: boolean }) {
  return (
    <>
      <span
        className={clsx(
          'px-3 py-1.5 text-sm text-text-secondary',
          mobile && 'block rounded-lg px-3 py-2.5'
        )}
      >
        Sign in
      </span>
      <Link
        href="/?auth=required"
        className={clsx(
          'inline-flex h-9 items-center gap-1.5 rounded-pill bg-ink-950 px-4 text-sm font-medium text-white transition-all duration-300 ease-premium hover:bg-ink-900',
          mobile && 'h-11 w-full justify-center'
        )}
      >
        Get started
        <ArrowIcon />
      </Link>
    </>
  )
}
