'use client'

import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * ScrollScene — mounts once on the landing page and drives all
 * scroll-triggered reveal animations via [data-reveal] attributes.
 * Runs no animations if the user prefers reduced motion.
 */
export function ScrollScene() {
  useGSAP(() => {
    document.documentElement.classList.add('has-scroll-scene')

    const reducedMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches

    const revealEls = document.querySelectorAll<HTMLElement>('[data-reveal]')

    if (reducedMotion) {
      // Immediately surface all elements — no animation
      revealEls.forEach((el) => {
        gsap.set(el, { opacity: 1, y: 0 })
      })
      return () => {
        document.documentElement.classList.remove('has-scroll-scene')
      }
    }

    revealEls.forEach((el) => {
      const delay = parseFloat(el.dataset.revealDelay ?? '0')

      gsap.fromTo(
        el,
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
          delay,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 85%',
            once: true,
          },
        }
      )
    })

    // Hero / above-fold content: reveal immediately if already in view
    ScrollTrigger.refresh()

    // Counter animations
    const counterEls = document.querySelectorAll<HTMLElement>('[data-counter]')
    counterEls.forEach((el) => {
      const target = parseFloat(el.dataset.counter ?? '0')
      const obj = { val: 0 }

      gsap.to(obj, {
        val: target,
        duration: 1.4,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 85%',
          once: true,
        },
        onUpdate() {
          el.textContent = obj.val.toFixed(
            el.dataset.counterDecimals ? parseInt(el.dataset.counterDecimals) : 0
          )
        },
      })
    })

    return () => {
      document.documentElement.classList.remove('has-scroll-scene')
      ScrollTrigger.getAll().forEach((st) => st.kill())
    }
  }, [])

  return null
}
