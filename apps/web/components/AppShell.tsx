'use client'

import { NavBar } from '@/components/NavBar'
import { SiteFooter } from '@/components/SiteFooter'

interface AppShellProps {
  children: React.ReactNode
}

/** Shared chrome for authenticated app routes — keeps logo + nav consistent with landing. */
export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-canvas">
      <NavBar variant="app" />
      <main className="pt-14">{children}</main>
      <SiteFooter />
    </div>
  )
}
