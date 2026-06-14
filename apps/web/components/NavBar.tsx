'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { SignInButton, SignUpButton, UserButton, useUser } from '@clerk/nextjs'
import { Logo } from '@/components/Logo'
import { clerkEnabled } from '@/lib/auth'
import { SITE } from '@/lib/site'
import clsx from 'clsx'

interface NavBarProps {
  variant?: 'landing' | 'app'
}

export function NavBar({ variant = 'landing' }: NavBarProps) {
  const [scrolled, setScrolled] = useState(variant === 'app')

  useEffect(() => {
    if (variant === 'app') return
    const handler = () => setScrolled(window.scrollY > 100)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [variant])

  return (
    <nav
      className={clsx(
        'fixed top-0 z-50 w-full transition-all duration-300',
        'bg-ink-950',
        scrolled ? 'border-b border-ink-700' : 'border-b border-transparent'
      )}
    >
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        {/* Left — logo + wordmark */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <Logo variant="nav" size={28} />
          <span className="text-sm font-medium text-text-primary tracking-tight">
            {SITE.name}
          </span>
        </Link>

        {/* Right — auth controls */}
        <div className="flex items-center gap-3">
          {clerkEnabled ? <NavBarAuthControls /> : <NavBarStaticControls />}
        </div>
      </div>
    </nav>
  )
}

function NavBarAuthControls() {
  const { isSignedIn } = useUser()

  if (isSignedIn) {
    return (
      <>
        <Link
          href="/dashboard"
          className="text-sm text-text-secondary hover:text-text-primary transition-colors"
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
        <button className="text-sm text-text-secondary hover:text-text-primary transition-colors px-3 py-1.5">
          Sign in
        </button>
      </SignInButton>
      <SignUpButton mode="modal">
        <button className="inline-flex h-8 items-center rounded-md bg-signal px-4 text-sm font-medium text-white hover:bg-signal-dim transition-colors">
          Get started
        </button>
      </SignUpButton>
    </>
  )
}

function NavBarStaticControls() {
  return (
    <>
      <span className="text-sm text-text-secondary px-3 py-1.5">Sign in</span>
      <Link
        href="/?auth=required"
        className="inline-flex h-8 items-center rounded-md bg-signal px-4 text-sm font-medium text-white hover:bg-signal-dim transition-colors"
      >
        Get started
      </Link>
    </>
  )
}
