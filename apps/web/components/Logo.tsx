'use client'

import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import clsx from 'clsx'
import { SITE } from '@/lib/site'

interface LogoProps {
  size?: number
  variant?: 'nav' | 'hero' | 'static'
  className?: string
}

export function Logo({ size = 80, variant = 'static', className }: LogoProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useGSAP(
    () => {
      if (!svgRef.current) return

      // Respect prefers-reduced-motion
      const reducedMotion = window.matchMedia(
        '(prefers-reduced-motion: reduce)'
      ).matches
      if (reducedMotion) return

      const neuralPath = svgRef.current.querySelector<SVGPathElement>('#neural-path')
      const glowDot = svgRef.current.querySelector<SVGCircleElement>('#glow-dot')

      if (!neuralPath || !glowDot) return

      if (variant === 'hero') {
        // Measure path length and set up stroke-dash draw animation
        const pathLength = neuralPath.getTotalLength()
        gsap.set(neuralPath, {
          strokeDasharray: pathLength,
          strokeDashoffset: pathLength,
        })

        // Single signature motion: neural path draws once, then holds.
        gsap.to(neuralPath, {
          strokeDashoffset: 0,
          duration: 2.5,
          ease: 'power2.out',
        })

        gsap.to(glowDot, {
          scale: 1.3,
          opacity: 0.85,
          duration: 1.2,
          ease: 'sine.inOut',
          transformOrigin: '50% 50%',
        })
      } else if (variant === 'nav') {
        // Hover-only: handled via CSS-level transform for performance
        // Subtle glow dot on hover (triggered by parent hover class)
        gsap.set(glowDot, { scale: 1, opacity: 1 })

        svgRef.current.addEventListener('mouseenter', () => {
          gsap.to(svgRef.current, { scale: 1.05, duration: 0.2, ease: 'power2.out' })
          gsap.to(glowDot, { scale: 1.4, opacity: 0.9, duration: 0.25, ease: 'power2.out' })
        })
        svgRef.current.addEventListener('mouseleave', () => {
          gsap.to(svgRef.current, { scale: 1, duration: 0.2, ease: 'power2.inOut' })
          gsap.to(glowDot, { scale: 1, opacity: 1, duration: 0.25, ease: 'power2.inOut' })
        })
      }
    },
    { scope: svgRef, dependencies: [variant] }
  )

  return (
    <svg
      ref={svgRef}
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={clsx('shrink-0', className)}
      aria-label={`${SITE.name} logo`}
    >
      {/* Left neural wave path — brain signal waveform */}
      <path
        id="neural-path"
        d="M 8 60 C 16 60 16 20 24 20 C 32 20 32 60 40 40 C 48 20 48 60 56 40 C 64 20 64 40 72 40"
        stroke="#2563ff"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
      />
      {/* Right N-stroke paths */}
      <path
        id="cursor-path"
        d="M 8 20 L 8 60 M 8 20 L 40 60 M 40 20 L 40 60"
        stroke="#f0f2f5"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
      />
      {/* Crossing point glow dot */}
      <circle
        id="glow-dot"
        cx="40"
        cy="40"
        r="4"
        fill="#2563ff"
      />
      {/* Outer ring */}
      <circle
        id="outer-ring"
        cx="40"
        cy="40"
        r="36"
        stroke="#2563ff"
        strokeWidth="0.5"
        opacity="0.3"
      />
    </svg>
  )
}
