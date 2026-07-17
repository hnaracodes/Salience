/**
 * Smoke-test all static marketing/app routes return non-5xx responses.
 * Usage: node scripts/smoke-routes.mjs [baseUrl]
 */
const base = process.argv[2] ?? 'http://localhost:3002'

const ROUTES = [
  '/',
  '/about',
  '/contact',
  '/security',
  '/privacy',
  '/terms',
  '/cookies',
  '/acceptable-use',
  '/anti-theft',
  '/data-processing',
  '/dashboard',
  '/scans/new',
  '/sign-in',
  '/sign-up',
  '/#use-cases',
  '/#product-journey',
  '/#how-it-works',
  '/#viewer',
]

const results = []

for (const route of ROUTES) {
  const url = `${base}${route}`
  try {
    const res = await fetch(url, { redirect: 'manual' })
    const ok = res.status < 500
    results.push({ route, status: res.status, ok })
    const mark = ok ? 'OK' : 'FAIL'
    console.log(`[${mark}] ${res.status} ${route}`)
  } catch (err) {
    results.push({ route, status: 0, ok: false, error: String(err) })
    console.log(`[FAIL] --- ${route} — ${err.message}`)
  }
}

const failed = results.filter((r) => !r.ok)
if (failed.length > 0) {
  console.error(`\n${failed.length} route(s) failed smoke test against ${base}`)
  process.exit(1)
}

console.log(`\nAll ${results.length} routes passed smoke test against ${base}`)
