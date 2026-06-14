'use client'

import Link from 'next/link'
import { Plus } from 'lucide-react'

export function NewScanButton() {
  return (
    <Link
      href="/scans/new"
      className="inline-flex items-center gap-2 h-10 rounded-md bg-signal px-5 text-sm font-medium text-white transition-colors hover:bg-signal-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
    >
      <Plus size={16} strokeWidth={2} />
      New scan
    </Link>
  )
}
