import { currentUser } from '@clerk/nextjs/server'
import { AppShell } from '@/components/AppShell'
import { NewScanButton } from '@/components/NewScanButton'
import { ScanList } from '@/components/ScanList'

export default async function DashboardPage() {
  const user = await currentUser()

  return (
    <AppShell>
      <div className="max-w-5xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-3xl font-display font-semibold text-text-primary tracking-tight">
            Welcome back
            {user?.firstName ? (
              <span className="text-signal">, {user.firstName}</span>
            ) : null}
            .
          </h1>
          <p className="text-text-secondary mt-2 text-base">
            Run a new scan or review previous analyses below.
          </p>
        </div>

        <div className="mb-12">
          <NewScanButton />
        </div>

        <section>
          <h2 className="text-sm font-mono text-text-secondary uppercase tracking-widest mb-4">
            Recent scans
          </h2>
          <ScanList />
        </section>
      </div>
    </AppShell>
  )
}
