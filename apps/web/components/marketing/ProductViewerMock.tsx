/**
 * Static high-fidelity mock of the Salience session viewer — light theme.
 */
import { Logo } from '@/components/Logo'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { HEATMAP_LIGHT, mock } from '@/components/marketing/mock-ui'
import { BarChart3, Shield, Sparkles } from 'lucide-react'
import { SITE } from '@/lib/site'

export function ProductViewerMock({ className = '' }: { className?: string }) {
  return (
    <div className={`${mock.shell} ${className}`}>
      <div className={mock.chrome}>
        <span className="h-2 w-2 rounded-full bg-red-400/70" />
        <span className="h-2 w-2 rounded-full bg-amber-400/70" />
        <span className="h-2 w-2 rounded-full bg-copy/70" />
        <span className={`${mock.chromeTitle} rounded-md bg-surface px-2 py-0.5`}>
          viewer · {SITE.exampleUrl} · session f848e992
        </span>
        <span className="rounded-full border border-copy/30 bg-copy/10 px-2 py-0.5 text-[9px] font-mono text-copy">
          TR 14 / 48
        </span>
      </div>

      <div className={`border-b ${mock.divider} bg-surface-muted px-3 py-2`}>
        <div className="flex items-center gap-1">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="h-3 flex-1 rounded-sm"
              style={{
                backgroundColor:
                  i === 13
                    ? 'rgba(29,78,216,0.85)'
                    : i < 13
                      ? `rgba(29,78,216,${0.12 + (i % 4) * 0.06})`
                      : 'rgba(232,230,223,0.9)',
                opacity: i === 13 ? 1 : 0.85,
              }}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[8px] font-mono text-text-tertiary">
          <span>0:00</span>
          <span className="text-signal">peak engagement · TR 14</span>
          <span>1:24</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px]">
        <div
          className={`relative min-h-[240px] border-b ${mock.divider} lg:min-h-[320px] lg:border-b-0 lg:border-r`}
        >
          <div className={`absolute inset-0 ${mock.panel} p-4`}>
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Logo variant="nav" size={20} />
                <div className={`h-2 w-20 ${mock.skeletonMid}`} />
              </div>
              <div className="flex gap-2">
                <div className={`h-2 w-10 ${mock.skeleton}`} />
                <div className={`h-2 w-10 ${mock.skeleton}`} />
                <div className={`h-5 w-14 rounded-full ${mock.skeletonBtn}`} />
              </div>
            </div>
            <div className="mx-auto max-w-[85%] space-y-3 text-center">
              <div className={`mx-auto h-3 w-3/4 ${mock.skeletonMid}`} />
              <div className={`mx-auto h-2 w-1/2 ${mock.skeleton}`} />
              <div className="mx-auto mt-4 h-8 w-32 rounded-lg bg-ink-950" />
            </div>
            <div className="mt-8 grid grid-cols-3 gap-2 px-2">
              <MiniFeatureTile icon={Sparkles} label="Feature" accent="signal" />
              <MiniFeatureTile icon={Shield} label="Trust" accent="copy" />
              <MiniFeatureTile icon={BarChart3} label="Metrics" accent="neural" />
            </div>
          </div>

          <div
            className="pointer-events-none absolute inset-0"
            style={{ background: HEATMAP_LIGHT, opacity: 0.75 }}
          />

          <BBox top="14%" left="18%" w="64%" h="12%" score={91} tone="engaging" label="H1 hero" />
          <BBox top="32%" left="38%" w="24%" h="8%" score={84} tone="engaging" label="CTA" />
          <BBox top="52%" left="12%" w="26%" h="18%" score={72} tone="neutral" label="Card" />
          <BBox top="52%" left="42%" w="26%" h="18%" score={88} tone="engaging" label="Card" />
          <BBox top="52%" left="72%" w="26%" h="18%" score={45} tone="boring" label="Card" />

          <div className="absolute bottom-2 left-2 flex gap-2 rounded-full border border-line bg-surface/90 px-2 py-1 text-[8px] font-mono shadow-card">
            <span className="text-neural">● Engaging</span>
            <span className="text-signal">● Neutral</span>
            <span className="text-text-tertiary">● Low</span>
          </div>
        </div>

        <div className={`space-y-2 ${mock.panel} p-2 text-[9px]`}>
          <SidebarPanel title="Session metrics">
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { k: 'Neural', v: '84', c: 'text-signal' },
                { k: 'Attention', v: '91', c: 'text-neural' },
                { k: 'Copy', v: '76', c: 'text-copy' },
                { k: 'Pages', v: '8', c: 'text-text-primary' },
              ].map((m) => (
                <div key={m.k} className={`${mock.rowMuted} text-center`}>
                  <p className="text-text-tertiary">{m.k}</p>
                  <p className={`font-mono text-sm font-semibold ${m.c}`}>{m.v}</p>
                </div>
              ))}
            </div>
            <svg viewBox="0 0 180 32" className="mt-2 w-full text-signal" aria-hidden="true">
              <path
                d="M0 20 L20 12 L40 18 L60 8 L80 22 L100 14 L120 26 L140 10 L160 18 L180 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </SidebarPanel>

          <SidebarPanel title="Cortical activation">
            <div className={`relative h-16 overflow-hidden rounded-lg border border-line bg-surface`}>
              <svg viewBox="0 0 120 60" className="h-full w-full" aria-hidden="true">
                <ellipse
                  cx="60"
                  cy="32"
                  rx="38"
                  ry="22"
                  fill="#F5F4EF"
                  stroke="#1D4ED8"
                  strokeWidth="0.5"
                  opacity="0.6"
                />
                <path
                  d="M35 38 Q50 18 60 28 Q72 42 85 30"
                  fill="none"
                  stroke="#E85D24"
                  strokeWidth="2"
                  opacity="0.85"
                />
                <circle cx="58" cy="28" r="4" fill="#E85D24" opacity="0.9" />
              </svg>
              <span className="absolute bottom-1 right-1 text-[7px] font-mono text-text-tertiary">
                TRIBE v2
              </span>
            </div>
          </SidebarPanel>

          <SidebarPanel title="Element · hero-h1">
            <p className="font-mono text-[8px] text-signal">91/100 combined</p>
            <div className={`mt-1 ${mock.metricBar}`}>
              <div className="h-full w-[91%] rounded-full bg-neural" />
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              <Chip text="clarity 88" />
              <Chip text="urgency 72" />
              <Chip text="goal fit 81" />
            </div>
            <p className="mt-2 leading-snug text-text-secondary">
              Hero headline drives peak VAN engagement at TR 14 — strong goal alignment.
            </p>
          </SidebarPanel>
        </div>
      </div>
    </div>
  )
}

function SidebarPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className={`rounded-lg border border-line bg-surface p-2 shadow-card`}>
      <p className={`mb-1.5 ${mock.label} uppercase tracking-wider`}>{title}</p>
      {children}
    </div>
  )
}

function Chip({ text }: { text: string }) {
  return (
    <span className="rounded-full border border-copy/30 bg-copy/10 px-1.5 py-0.5 font-mono text-[7px] text-copy">
      {text}
    </span>
  )
}

function BBox({
  top,
  left,
  w,
  h,
  score,
  tone,
  label,
}: {
  top: string
  left: string
  w: string
  h: string
  score: number
  tone: 'engaging' | 'neutral' | 'boring'
  label: string
}) {
  const colors = {
    engaging: { bg: 'rgba(232,93,36,0.12)', border: '#E85D24' },
    neutral: { bg: 'rgba(29,78,216,0.1)', border: '#1D4ED8' },
    boring: { bg: 'rgba(138,138,138,0.12)', border: '#8A8A8A' },
  }[tone]

  return (
    <div
      className="pointer-events-none absolute rounded-sm border-2"
      style={{
        top,
        left,
        width: w,
        height: h,
        backgroundColor: colors.bg,
        borderColor: colors.border,
      }}
    >
      <span
        className="absolute -top-4 left-0 whitespace-nowrap rounded-full px-2 py-px font-mono text-[8px] text-white"
        style={{ backgroundColor: colors.border }}
      >
        {label} · {score}
      </span>
    </div>
  )
}
