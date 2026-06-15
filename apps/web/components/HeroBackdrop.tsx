'use client'

/**
 * Precision-instrument backdrop: dot grid, corner reticles, horizontal scan line.
 * Solid ink base + signal accents — no mesh gradients or glassmorphism.
 */
export function HeroBackdrop() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      {/* Dot grid */}
      <div
        className="absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            'radial-gradient(circle, rgba(37,99,255,0.18) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />

      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 80% 70% at 50% 40%, transparent 0%, #08090b 72%)',
        }}
      />

      {/* Signal wash behind product panel (right) */}
      <div
        className="absolute right-0 top-1/4 h-[480px] w-[55%] max-w-2xl opacity-60"
        style={{
          background:
            'radial-gradient(ellipse 70% 60% at 70% 50%, rgba(37,99,255,0.14) 0%, transparent 70%)',
        }}
      />

      {/* Corner reticles */}
      <svg
        className="absolute left-6 top-24 h-16 w-16 text-signal/40 md:left-12"
        viewBox="0 0 64 64"
        fill="none"
      >
        <path d="M4 20V4H20" stroke="currentColor" strokeWidth="1" />
        <path d="M44 4H60V20" stroke="currentColor" strokeWidth="1" />
      </svg>
      <svg
        className="absolute bottom-24 right-6 h-16 w-16 text-signal/30 md:right-12"
        viewBox="0 0 64 64"
        fill="none"
      >
        <path d="M4 44V60H20" stroke="currentColor" strokeWidth="1" />
        <path d="M44 60H60V44" stroke="currentColor" strokeWidth="1" />
      </svg>

      {/* Scan line */}
      <div className="hero-scan-line absolute left-0 right-0 top-[38%] h-px bg-gradient-to-r from-transparent via-signal/50 to-transparent" />

      {/* Vertical axis ticks */}
      <div className="absolute left-4 top-1/3 hidden h-40 w-px bg-ink-700 md:block" />
      <div className="absolute right-4 top-1/4 hidden h-56 w-px bg-ink-700 md:block" />
    </div>
  )
}
