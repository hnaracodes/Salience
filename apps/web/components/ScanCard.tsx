'use client'

import Link from 'next/link'
import clsx from 'clsx'

interface Scan {
  id: string
  url: string
  status: 'queued' | 'running' | 'done' | 'failed'
  created_at: string
}

interface ScanCardProps {
  scan: Scan
}

const statusConfig: Record<
  Scan['status'],
  { label: string; className: string }
> = {
  queued: {
    label: 'Queued',
    className: 'text-muted border-muted/40 bg-muted/10',
  },
  running: {
    label: 'Running',
    className:
      'text-signal border-signal/40 bg-signal/10 animate-pulse-signal',
  },
  done: {
    label: 'Done',
    className: 'text-copy border-copy/40 bg-copy/10',
  },
  failed: {
    label: 'Failed',
    className: 'text-red-400 border-red-900/40 bg-red-950/20',
  },
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export function ScanCard({ scan }: ScanCardProps) {
  const { label, className } = statusConfig[scan.status]

  // Truncate URL for display
  let displayUrl = scan.url
  try {
    const u = new URL(scan.url)
    displayUrl = u.hostname + (u.pathname !== '/' ? u.pathname : '')
  } catch {}

  return (
    <div className="flex items-center justify-between gap-4 bg-surface px-5 py-4 transition-colors duration-300 hover:bg-surface-muted">
      <div className="min-w-0 flex-1">
        <p
          className="truncate text-sm font-mono text-text-primary"
          title={scan.url}
        >
          {displayUrl}
        </p>
        <p className="mt-0.5 text-xs text-text-secondary">
          {relativeTime(scan.created_at)}
        </p>
      </div>

      <div className="flex items-center gap-4 shrink-0">
        <span
          className={clsx(
            'inline-flex items-center rounded-sm border px-2 py-0.5 text-xs font-mono',
            className
          )}
        >
          {label}
        </span>

        {scan.status === 'done' && (
          <Link
            href={`/scans/${scan.id}`}
            className="text-xs text-signal hover:underline font-medium"
          >
            View results →
          </Link>
        )}
        {scan.status !== 'done' && (
          <Link
            href={`/scans/${scan.id}`}
            className="text-xs text-text-secondary hover:text-text-primary transition-colors"
          >
            Details →
          </Link>
        )}
      </div>
    </div>
  )
}
