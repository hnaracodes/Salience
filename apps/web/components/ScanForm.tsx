'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@clerk/nextjs'
import { apiFetch } from '@/lib/api'
import { clerkEnabled } from '@/lib/auth'
import Link from 'next/link'

interface ScanResponse {
  id: string
  status: string
}

export function ScanForm() {
  if (!clerkEnabled) {
    return (
      <p className="rounded-md border border-ink-700 bg-ink-900 px-4 py-3 text-sm text-text-secondary">
        Authentication is not configured. Add Clerk keys to start scans, or{' '}
        <Link href="/" className="text-signal hover:underline">
          return home
        </Link>
        .
      </p>
    )
  }
  return <ScanFormAuthed />
}

function ScanFormAuthed() {
  const { getToken } = useAuth()
  const router = useRouter()

  const [url, setUrl] = useState('')
  const [goal, setGoal] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)

    if (!url.trim()) {
      setError('Please enter a URL to scan.')
      return
    }

    setLoading(true)
    try {
      const token = await getToken()
      if (!token) throw new Error('Not authenticated.')

      const scan = await apiFetch<ScanResponse>('/v1/scans', token, {
        method: 'POST',
        body: JSON.stringify({
          url: url.trim(),
          ...(goal.trim() ? { site_goal: goal.trim() } : {}),
        }),
      })

      router.push(`/scans/${scan.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* URL input */}
      <div>
        <label
          htmlFor="url"
          className="mb-2 block text-sm font-medium text-text-primary"
        >
          Website URL
        </label>
        <input
          id="url"
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com"
          required
          className="w-full rounded-md border border-ink-700 bg-ink-900 px-4 py-3 text-sm text-text-primary placeholder-text-secondary outline-none transition-colors focus:border-signal focus:ring-1 focus:ring-signal"
          disabled={loading}
          autoFocus
        />
      </div>

      {/* Goal textarea */}
      <div>
        <label
          htmlFor="goal"
          className="mb-2 block text-sm font-medium text-text-primary"
        >
          Site goal{' '}
          <span className="text-text-secondary font-normal">(optional)</span>
        </label>
        <textarea
          id="goal"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="e.g. Drive sign-ups for our SaaS trial. Reduce bounce on the pricing page."
          rows={3}
          className="w-full resize-none rounded-md border border-ink-700 bg-ink-900 px-4 py-3 text-sm text-text-primary placeholder-text-secondary outline-none transition-colors focus:border-signal focus:ring-1 focus:ring-signal"
          disabled={loading}
        />
        <p className="mt-1 text-xs text-text-secondary">
          Describe what you want visitors to do. Used to score copy goal-fit.
        </p>
      </div>

      {/* Error message */}
      {error && (
        <p className="rounded-md border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-400">
          {error}
        </p>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="inline-flex h-12 w-full items-center justify-center rounded-md bg-signal text-sm font-medium text-white transition-colors hover:bg-signal-dim disabled:cursor-not-allowed disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <Spinner />
            Submitting…
          </span>
        ) : (
          'Start scan'
        )}
      </button>
    </form>
  )
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  )
}
