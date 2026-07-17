'use client'

import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * ScrollScene — scroll-triggered reveals for [data-reveal] elements
 * and smooth section entrance for [data-section] blocks.
 */
export function ScrollScene() {
  useGSAP(() => {
    document.documentElement.classList.add('has-scroll-scene')

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const revealEls = document.querySelectorAll<HTMLElement>('[data-reveal]')
    const sections = document.querySelectorAll<HTMLElement>('[data-section]')

    if (reducedMotion) {
      revealEls.forEach((el) => gsap.set(el, { opacity: 1, y: 0 }))
      sections.forEach((el) => gsap.set(el, { opacity: 1 }))
      return () => {
        document.documentElement.classList.remove('has-scroll-scene')
      }
    }

    sections.forEach((section) => {
      if (section.id === 'hero') return
      gsap.fromTo(
        section,
        { opacity: 0.72 },
        {
          opacity: 1,
          duration: 1.4,
          ease: 'power2.inOut',
          scrollTrigger: {
            trigger: section,
            start: 'top 92%',
            end: 'top 55%',
            scrub: 1.2,
          },
        }
      )
    })

    revealEls.forEach((el) => {
      const delay = parseFloat(el.dataset.revealDelay ?? '0')

      gsap.fromTo(
        el,
        { opacity: 0, y: 32 },
        {
          opacity: 1,
          y: 0,
          duration: 1.15,
          delay,
          ease: 'power4.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 90%',
            once: true,
          },
        }
      )
    })

    ScrollTrigger.refresh()

    const counterEls = document.querySelectorAll<HTMLElement>('[data-counter]')
    counterEls.forEach((el) => {
      const target = parseFloat(el.dataset.counter ?? '0')
      const obj = { val: 0 }

      gsap.to(obj, {
        val: target,
        duration: 2,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 90%',
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
