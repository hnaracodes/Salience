'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { useAuth } from '@clerk/nextjs'
import { apiFetch } from '@/lib/api'
import clsx from 'clsx'
import { CheckCircle, Circle, Loader } from 'lucide-react'

const STAGES = [
  { key: 'capture', label: 'Capture pages' },
  { key: 'tribe', label: 'TRIBE v2 inference' },
  { key: 'dual_track', label: 'Dual-track analysis' },
  { key: 'heatmaps', label: 'Generate heatmaps' },
  { key: 'analyze', label: 'Score attention' },
  { key: 'copy_signals', label: 'Copy signal extraction' },
  { key: 'narrative', label: 'Build narrative' },
  { key: 'export_viewer', label: 'Export viewer' },
] as const

type StageKey = (typeof STAGES)[number]['key']

interface ScanStatus {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  current_stage?: StageKey | string
  error?: string
}

interface ProgressStepperProps {
  scanId: string
  /** When the parent page already owns polling, pass the scan here to avoid
   *  a duplicate network call. ProgressStepper falls back to its own polling
   *  when this is undefined (e.g. when used standalone). */
  externalScan?: ScanStatus
}

export function ProgressStepper({ scanId, externalScan }: ProgressStepperProps) {
  const { getToken } = useAuth()
  const [ownScan, setOwnScan] = useState<ScanStatus | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const scanRef = useRef(ownScan)
  scanRef.current = ownScan

  // Only run our own poll when the parent hasn't taken over polling.
  const poll = useCallback(async () => {
    if (externalScan !== undefined) return
    try {
      const token = await getToken()
      if (!token) return
      const data = await apiFetch<ScanStatus>(`/v1/scans/${scanId}`, token)
      setOwnScan(data)
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : 'Failed to load scan.')
    }
  }, [scanId, getToken, externalScan])

  useEffect(() => {
    if (externalScan !== undefined) return
    poll()

    const interval = setInterval(() => {
      const current = scanRef.current
      if (!current || current.status === 'running' || current.status === 'queued') {
        poll()
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [poll, externalScan])

  const scan = externalScan ?? ownScan

  if (fetchError) {
    return (
      <p className="text-sm text-red-400">{fetchError}</p>
    )
  }

  if (!scan) {
    return <StepperSkeleton />
  }

  const currentStageIndex = STAGES.findIndex(
    (s) => s.key === scan.current_stage
  )

  return (
    <div className="space-y-1">
      {/* Status badge */}
      <div className="mb-6 flex items-center gap-3">
        <StatusBadge status={scan.status} />
        {scan.status === 'failed' && scan.error && (
          <span className="text-xs text-red-400">{scan.error}</span>
        )}
      </div>

      {/* Stage list */}
      <ol className="space-y-0">
        {STAGES.map((stage, i) => {
          const isDone =
            scan.status === 'done' ||
            (currentStageIndex > -1 && i < currentStageIndex)
          const isActive =
            scan.status === 'running' && stage.key === scan.current_stage
          const isPending = !isDone && !isActive

          return (
            <li
              key={stage.key}
              className="flex items-start gap-4 py-3 border-b border-ink-700 last:border-0"
            >
              {/* Stage indicator */}
              <span className="mt-0.5 shrink-0">
                {isDone ? (
                  <CheckCircle
                    size={16}
                    className="text-copy"
                    strokeWidth={2}
                  />
                ) : isActive ? (
                  <Loader
                    size={16}
                    className="text-signal animate-spin"
                    strokeWidth={2}
                  />
                ) : (
                  <Circle
                    size={16}
                    className="text-ink-600"
                    strokeWidth={1.5}
                  />
                )}
              </span>

              <span
                className={clsx(
                  'text-sm',
                  isDone && 'text-text-secondary',
                  isActive && 'text-text-primary font-medium',
                  isPending && 'text-text-secondary opacity-50'
                )}
              >
                {stage.label}
              </span>

              {/* Active pulse dot */}
              {isActive && (
                <span
                  className="ml-auto mt-1 h-1.5 w-1.5 rounded-full bg-signal animate-pulse-signal"
                  aria-hidden="true"
                />
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}

function StatusBadge({ status }: { status: ScanStatus['status'] }) {
  const map: Record<
    ScanStatus['status'],
    { label: string; className: string }
  > = {
    queued: { label: 'Queued', className: 'text-muted border-muted/40 bg-muted/10' },
    running: {
      label: 'Running',
      className: 'text-signal border-signal/40 bg-signal/10',
    },
    done: { label: 'Complete', className: 'text-copy border-copy/40 bg-copy/10' },
    failed: { label: 'Failed', className: 'text-red-400 border-red-900/40 bg-red-950/20' },
  }

  const { label, className } = map[status]

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-sm border px-2.5 py-0.5 text-xs font-mono',
        className
      )}
    >
      {label}
    </span>
  )
}

function StepperSkeleton() {
  return (
    <div className="space-y-3 animate-pulse">
      {STAGES.map((s) => (
        <div key={s.key} className="h-8 rounded bg-ink-800" />
      ))}
    </div>
  )
}
