'use client'

import Link from 'next/link'

interface ViewerFrameProps {
  src: string
  scanId?: string
  variant?: 'embedded' | 'fullscreen'
}

export function ViewerFrame({ src, scanId, variant = 'embedded' }: ViewerFrameProps) {
  const fullScreenPath = scanId ? `/scans/${scanId}/viewer` : null
  const heightClass = variant === 'fullscreen' ? 'flex-1 min-h-0 h-full' : 'h-[800px]'

  return (
    <div className={`relative flex flex-col ${variant === 'fullscreen' ? 'flex-1 min-h-0' : ''}`}>
      {variant === 'embedded' && (
        <div className="flex items-center justify-end gap-3 mb-2">
          {fullScreenPath && (
            <Link
              href={fullScreenPath}
              className="text-sm font-medium text-accent hover:underline cursor-pointer"
            >
              Open full screen
            </Link>
          )}
          <button
            type="button"
            className="text-sm text-text-secondary hover:text-text-primary cursor-pointer transition-colors"
            onClick={() => window.open(src, '_blank', 'noopener,noreferrer')}
          >
            Open in new tab
          </button>
        </div>
      )}
      <iframe
        src={src}
        className={`w-full rounded-lg border border-ink-700 ${heightClass}`}
        title="UX Analysis Viewer"
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  )
}
