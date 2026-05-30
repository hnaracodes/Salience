# Aurora walkthrough fixture

Visual showcase site for Playwright + TRIBEv2 pipeline testing. Designed to produce **motion-rich, high-contrast** screen recordings (gradients, particles, marquee, glass cards) so zero-shot emotion/engagement tracks have more stimulus than a flat HTML page.

## Preview locally

```bash
python -m http.server 8765 --directory tests/fixtures/walkthrough_site
```

Open [http://127.0.0.1:8765/index.html](http://127.0.0.1:8765/index.html)

## Record

**Quick (5 TRs, legacy scroll checks):**

```bash
python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml
```

**Cinematic (14 TRs, recommended for Layers 2–3):**

```bash
python scripts/record_website_session.py --script configs/walkthrough_scripts/aurora_showcase.yaml
```

## Sections (DOM / analytics)

| Section | Element | ~scrollY @ 1280×720 |
|---------|---------|---------------------|
| Hero | `#hero` | 0 |
| Experience | `#experience` | ~750 |
| Gallery | `#gallery` | ~1550–2400 |
| Pricing | `#pricing` | ~3200 |
| CTA | `#cta` | ~3900 |

Requires network on first load for Google Fonts; animations run offline after that.
