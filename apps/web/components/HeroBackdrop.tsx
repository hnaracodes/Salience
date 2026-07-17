'use client'

/**
 * Warm editorial backdrop — subtle grain, soft wash, no dark mesh gradients.
 */
export function HeroBackdrop() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <div
        className="absolute inset-0 opacity-[0.45]"
        style={{
          backgroundImage:
            'radial-gradient(circle, rgba(15,15,15,0.04) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />

      <div
        className="absolute -right-32 top-0 h-[600px] w-[600px] rounded-full opacity-60"
        style={{
          background:
            'radial-gradient(circle, rgba(29,78,216,0.06) 0%, transparent 68%)',
        }}
      />

      <div
        className="absolute -left-24 bottom-0 h-[480px] w-[480px] rounded-full opacity-50"
        style={{
          background:
            'radial-gradient(circle, rgba(232,93,36,0.05) 0%, transparent 70%)',
        }}
      />

      <div className="hero-scan-line absolute left-0 right-0 top-[42%] h-px bg-gradient-to-r from-transparent via-line-strong to-transparent" />
    </div>
  )
}
