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
          <div key={n} className="h-16 rounded-card bg-line" />
        ))}
      </div>
    )
  }

  if (scans.length === 0) {
    return (
      <p className="rounded-card border border-dashed border-line py-8 text-center text-sm text-text-secondary">
        No scans yet. Start your first scan above.
      </p>
    )
  }

  return (
    <ul className="marketing-card divide-y divide-line overflow-hidden">
      {scans.map((scan) => (
        <li key={scan.id}>
          <ScanCard scan={scan} />
        </li>
      ))}
    </ul>
  )
}
