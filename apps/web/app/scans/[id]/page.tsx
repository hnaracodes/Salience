'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { useAuth } from '@clerk/nextjs'
import { AppShell } from '@/components/AppShell'
import { ProgressStepper } from '@/components/ProgressStepper'
import { ViewerFrame } from '@/components/ViewerFrame'
import { apiFetch } from '@/lib/api'

interface ScanPageProps {
  params: { id: string }
}

interface ScanDetail {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  current_stage?: string
  viewer_url?: string
  error?: string
}

export default function ScanPage({ params }: ScanPageProps) {
  return (
    <AppShell>
      <div className="max-w-5xl mx-auto px-6 py-12">
        <ScanDetailView scanId={params.id} />
      </div>
    </AppShell>
  )
}

function ScanDetailView({ scanId }: { scanId: string }) {
  const { getToken } = useAuth()
  const [scan, setScan] = useState<ScanDetail | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const scanRef = useRef(scan)
  scanRef.current = scan

  const poll = useCallback(async () => {
    try {
      const token = await getToken()
      if (!token) return
      const data = await apiFetch<ScanDetail>(`/v1/scans/${scanId}`, token)
      setScan(data)
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : 'Failed to load scan.')
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

  if (fetchError) {
    return <p className="text-sm text-red-400">{fetchError}</p>
  }

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-display font-semibold text-text-primary tracking-tight mb-1">
          Scan analysis
        </h1>
        <p className="text-text-secondary text-sm font-mono">ID: {scanId}</p>
      </div>

      {/* ProgressStepper consumes the already-polled scan state via props */}
      <ProgressStepper scanId={scanId} externalScan={scan ?? undefined} />

      {scan?.status === 'done' && scan.viewer_url && (
        <section>
          <h2 className="text-sm font-mono text-text-secondary uppercase tracking-widest mb-4">
            Neural viewer
          </h2>
          <ViewerFrame src={scan.viewer_url} />
        </section>
      )}
    </div>
  )
}
