'use client'

import { useEffect, useState } from 'react'
import { useAuth } from '@clerk/nextjs'
import { apiFetch } from '@/lib/api'
import { ScanCard } from '@/components/ScanCard'

interface Scan {
  id: string
  url: string
  status: 'queued' | 'running' | 'done' | 'failed'
  created_at: string
}

interface ScanListResponse {
  scans: Scan[]
  total: number
}

export function ScanList() {
  const { getToken } = useAuth()
  const [scans, setScans] = useState<Scan[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        const token = await getToken()
        if (!token) return
        const data = await apiFetch<ScanListResponse>('/v1/scans', token)
        setScans(data.scans)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load scans.')
      }
    }
    load()
  }, [getToken])

  if (error) {
    return <p className="text-sm text-red-400">{error}</p>
  }

  if (!scans) {
    return (
      <div className="space-y-3 animate-pulse">
        {[1, 2, 3].map((n) => (
          <div key={n} className="h-16 rounded bg-ink-800" />
        ))}
      </div>
    )
  }

  if (scans.length === 0) {
    return (
      <p className="text-sm text-text-secondary py-8 text-center border border-dashed border-ink-700 rounded-md">
        No scans yet. Start your first scan above.
      </p>
    )
  }

  return (
    <ul className="divide-y divide-ink-700 border border-ink-700 rounded-md overflow-hidden">
      {scans.map((scan) => (
        <li key={scan.id}>
          <ScanCard scan={scan} />
        </li>
      ))}
    </ul>
  )
}
