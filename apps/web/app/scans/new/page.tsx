import { AppShell } from '@/components/AppShell'
import { ScanForm } from '@/components/ScanForm'
import { SITE } from '@/lib/site'

export const dynamic = 'force-dynamic'

export default function NewScanPage() {
  return (
    <AppShell>
      <div className="max-w-2xl mx-auto px-6 py-16">
        <div className="mb-10">
          <h1 className="text-3xl font-display font-semibold text-text-primary tracking-tight mb-2">
            Start a new scan
          </h1>
          <p className="text-text-secondary text-base">
            Enter any public URL. {SITE.name} will crawl every page, model cortical
            engagement, and generate a full UX report.
          </p>
        </div>

        <ScanForm />
      </div>
    </AppShell>
  )
}
