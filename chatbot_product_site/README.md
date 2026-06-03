# Threadmind — Playwright E2E fixture site

Multi-page marketing site for an AI chatbot product (ChatGPT-style). Built for **TRIBEv2 / Playwright walkthrough testing** with rich scroll motion, semantic DOM landmarks, and subscription-focused copy.

## Design direction

**Editorial warmth + mechanical precision** — ink backgrounds, copper/signal accents, Fraunces + IBM Plex (no Inter, no purple mesh gradients). The homepage features a **scroll-driven metallic ball** on a fixed rail that rotates and morphs color as you pass each section (GSAP ScrollTrigger).

## Preview

```bash
python -m http.server 8780 --directory chatbot_product_site
```

Open [http://127.0.0.1:8780/index.html](http://127.0.0.1:8780/index.html)

Requires network on first load for Google Fonts + GSAP CDN; works offline afterward.

## Pages

| Page | Path | Purpose |
|------|------|---------|
| Home (scroll journey) | `index.html` | Hero, why subscribe, flow, models, pricing teaser, FAQ, CTA |
| Features | `features.html` | Capability grid |
| Models | `models.html` | Lite / Reason / Atlas detail |
| Pricing | `pricing.html` | Comparison table + cost breakdown |
| Enterprise | `enterprise.html` | SSO, audit, VPC |
| About | `about.html` | Mission, careers, legal |

## DOM landmarks (analytics / Playwright)

| Section | ID | ~scroll region |
|---------|-----|----------------|
| Hero | `#hero` | top |
| Why subscribe | `#why-subscribe` | early scroll |
| How it works | `#how-it-works` | mid |
| Models preview | `#models-preview` | mid |
| Pricing teaser | `#pricing-teaser` | mid-late |
| Testimonials | `#testimonials` | late |
| FAQ | `#faq` | late |
| CTA | `#cta` | bottom |

Primary CTAs: `#hero-cta`, `#cta-primary`, `#nav-subscribe`, `#pricing-cta`

## Suggested walkthrough script

Create `configs/walkthrough_scripts/threadmind_showcase.yaml` pointing at port 8780 with `scroll_mode: linear` — same pattern as `aurora_showcase.yaml`.

## Stack

- Static HTML/CSS/JS (no build step)
- GSAP 3 + ScrollTrigger for scroll ball and section reveals
- `prefers-reduced-motion` respected
