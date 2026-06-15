/**
 * Static high-fidelity mock of the Salience session viewer.
 * Mirrors ux_session_viewer.html layout: frame + heatmap + bboxes + sidebar.
 */
import { Logo } from '@/components/Logo'
import { MiniFeatureTile } from '@/components/marketing/PageRouteIcon'
import { BarChart3, Shield, Sparkles } from 'lucide-react'
import { SITE } from '@/lib/site'
export function ProductViewerMock({ className = '' }: { className?: string }) {
  return (
    <div className={`overflow-hidden rounded-lg border border-ink-700 bg-[#0a0c10] ${className}`}>
      {/* Browser chrome */}
      <div className="flex items-center gap-2 border-b border-ink-700 bg-ink-800 px-3 py-2">
        <span className="h-2 w-2 rounded-full bg-red-500/60" />
        <span className="h-2 w-2 rounded-full bg-yellow-500/50" />
        <span className="h-2 w-2 rounded-full bg-green-500/50" />
        <span className="ml-2 flex-1 truncate rounded bg-ink-700 px-2 py-0.5 text-[10px] font-mono text-text-secondary">
          viewer · {SITE.exampleUrl} · session f848e992
        </span>
        <span className="rounded border border-copy/30 bg-copy/10 px-1.5 py-0.5 text-[9px] font-mono text-copy">
          TR 14 / 48
        </span>
      </div>

      {/* Timeline scrubber */}
      <div className="border-b border-ink-700 bg-ink-900 px-3 py-2">
        <div className="flex items-center gap-1">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="h-3 flex-1 rounded-sm"
              style={{
                backgroundColor:
                  i === 13
                    ? 'rgba(37,99,255,0.9)'
                    : i < 13
                      ? `rgba(37,99,255,${0.15 + (i % 4) * 0.08})`
                      : 'rgba(28,32,40,0.9)',
                opacity: i === 13 ? 1 : 0.7,
              }}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[8px] font-mono text-text-secondary">
          <span>0:00</span>
          <span className="text-signal">peak engagement · TR 14</span>
          <span>1:24</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px]">
        {/* Main canvas — captured page + heatmap + bboxes */}
        <div className="relative min-h-[240px] border-b border-ink-700 lg:min-h-[320px] lg:border-b-0 lg:border-r">
          {/* Threadmind-style page mock */}
          <div className="absolute inset-0 bg-[#0f1117] p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Logo variant="nav" size={20} />
                <div className="h-2 w-20 rounded-sm bg-white/20" />
              </div>
              <div className="flex gap-2">
                <div className="h-2 w-10 rounded-sm bg-white/10" />
                <div className="h-2 w-10 rounded-sm bg-white/10" />
                <div className="h-5 w-14 rounded-full bg-signal/40" />
              </div>
            </div>
            <div className="mx-auto max-w-[85%] space-y-3 text-center">
              <div className="mx-auto h-3 w-3/4 rounded-sm bg-white/25" />
              <div className="mx-auto h-2 w-1/2 rounded-sm bg-white/12" />
              <div className="mx-auto mt-4 h-8 w-32 rounded-md bg-signal/50" />
            </div>
            <div className="mt-8 grid grid-cols-3 gap-2 px-2">
              <MiniFeatureTile icon={Sparkles} label="Feature" accent="signal" />
              <MiniFeatureTile icon={Shield} label="Trust" accent="copy" />
              <MiniFeatureTile icon={BarChart3} label="Metrics" accent="neural" />
            </div>
          </div>

          {/* Heatmap overlay */}
          <div
            className="absolute inset-0 mix-blend-screen pointer-events-none"
            style={{
              background: `
                radial-gradient(ellipse 50% 35% at 50% 22%, rgba(255,120,40,0.55) 0%, transparent 55%),
                radial-gradient(ellipse 35% 30% at 72% 48%, rgba(255,200,60,0.35) 0%, transparent 50%),
                radial-gradient(ellipse 40% 25% at 25% 70%, rgba(37,99,255,0.25) 0%, transparent 55%)
              `,
              opacity: 0.75,
            }}
          />

          {/* Element bboxes */}
          <BBox top="14%" left="18%" w="64%" h="12%" score={91} tone="engaging" label="H1 hero" />
          <BBox top="32%" left="38%" w="24%" h="8%" score={84} tone="engaging" label="CTA" />
          <BBox top="52%" left="12%" w="26%" h="18%" score={72} tone="neutral" label="Card" />
          <BBox top="52%" left="42%" w="26%" h="18%" score={88} tone="engaging" label="Card" />
          <BBox top="52%" left="72%" w="26%" h="18%" score={45} tone="boring" label="Card" />

          {/* Legend */}
          <div className="absolute bottom-2 left-2 flex gap-2 rounded bg-black/60 px-2 py-1 text-[8px] font-mono">
            <span className="text-orange-400">● Engaging</span>
            <span className="text-blue-400">● Neutral</span>
            <span className="text-gray-500">● Low</span>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-2 bg-[#0d0f14] p-2 text-[9px]">
          <SidebarPanel title="Session metrics">
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { k: 'Neural', v: '84', c: 'text-signal' },
                { k: 'Attention', v: '91', c: 'text-neural' },
                { k: 'Copy', v: '76', c: 'text-copy' },
                { k: 'Pages', v: '8', c: 'text-text-primary' },
              ].map((m) => (
                <div key={m.k} className="rounded border border-ink-700 bg-ink-900 px-2 py-1.5">
                  <p className="text-text-secondary">{m.k}</p>
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
            <div className="relative h-16 overflow-hidden rounded border border-ink-700 bg-ink-950">
              <svg viewBox="0 0 120 60" className="h-full w-full" aria-hidden="true">
                <ellipse cx="60" cy="32" rx="38" ry="22" fill="#1a2030" stroke="#2563ff" strokeWidth="0.5" opacity="0.6" />
                <path
                  d="M35 38 Q50 18 60 28 Q72 42 85 30"
                  fill="none"
                  stroke="#ff6b35"
                  strokeWidth="2"
                  opacity="0.85"
                />
                <circle cx="58" cy="28" r="4" fill="#ff6b35" opacity="0.9" />
              </svg>
              <span className="absolute bottom-1 right-1 text-[7px] font-mono text-text-secondary">TRIBE v2</span>
            </div>
          </SidebarPanel>

          <SidebarPanel title="Element · hero-h1">
            <p className="font-mono text-[8px] text-signal">91/100 combined</p>
            <div className="mt-1 h-1 w-full rounded-full bg-ink-700">
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
    <div className="rounded border border-ink-700/80 bg-ink-900/50 p-2">
      <p className="mb-1.5 font-mono text-[8px] uppercase tracking-wider text-text-secondary">{title}</p>
      {children}
    </div>
  )
}

function Chip({ text }: { text: string }) {
  return (
    <span className="rounded border border-copy/30 bg-copy/10 px-1 py-0.5 font-mono text-[7px] text-copy">
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
    engaging: { bg: 'rgba(255,107,53,0.2)', border: '#ff6b35' },
    neutral: { bg: 'rgba(37,99,255,0.15)', border: '#2563ff' },
    boring: { bg: 'rgba(100,116,139,0.15)', border: '#64748b' },
  }[tone]

  return (
    <div
      className="absolute rounded-sm border-2 pointer-events-none"
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
        className="absolute -top-4 left-0 whitespace-nowrap rounded px-1 py-px font-mono text-[8px] text-white"
        style={{ backgroundColor: colors.border }}
      >
        {label} · {score}
      </span>
    </div>
  )
}
