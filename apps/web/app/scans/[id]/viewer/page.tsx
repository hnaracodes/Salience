'use client'

import Link from 'next/link'
import { useEffect, useState, useCallback, useRef } from 'react'
import { useAuth } from '@clerk/nextjs'
import { AppShell } from '@/components/AppShell'
import { ViewerFrame } from '@/components/ViewerFrame'
import { apiFetch } from '@/lib/api'

interface ViewerPageProps {
  params: { id: string }
}

interface ScanDetail {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  viewer_url?: string
}

export default function ScanViewerPage({ params }: ViewerPageProps) {
  return (
    <AppShell>
      <div className="flex flex-col h-[calc(100vh-4rem)] min-h-[600px] px-4 py-4">
        <ScanViewerFull scanId={params.id} />
      </div>
    </AppShell>
  )
}

function ScanViewerFull({ scanId }: { scanId: string }) {
  const { getToken } = useAuth()
  const [scan, setScan] = useState<ScanDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const scanRef = useRef(scan)
  scanRef.current = scan

  const poll = useCallback(async () => {
    try {
      const token = await getToken()
      if (!token) return
      const data = await apiFetch<ScanDetail>(`/v1/scans/${scanId}`, token)
      setScan(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scan.')
    }
  }, [scanId, getToken])

  useEffect(() => {
    poll()
    const interval = setInterval(() => {
      const current = scanRef.current
      if (!current || current.status === 'queued' || current.status === 'running') {
        poll()
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [poll])

  if (error) {
    return <p className="text-sm text-red-400">{error}</p>
  }

  if (scan?.status !== 'done' || !scan.viewer_url) {
    return (
      <div className="flex flex-1 items-center justify-center text-text-secondary text-sm">
        {scan?.status === 'running' || scan?.status === 'queued'
          ? 'Scan in progress — viewer will appear when analysis completes.'
          : 'Viewer not available for this scan.'}
      </div>
    )
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-3">
      <header className="flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div>
          <h1 className="text-lg font-display font-semibold text-text-primary tracking-tight">
            Neural session viewer
          </h1>
          <p className="text-text-secondary text-xs font-mono">{scanId}</p>
        </div>
        <Link
          href={`/scans/${scanId}`}
          className="text-sm text-accent hover:underline cursor-pointer"
        >
          Back to scan summary
        </Link>
      </header>
      <ViewerFrame src={scan.viewer_url} scanId={scanId} variant="fullscreen" />
    </div>
  )
}
