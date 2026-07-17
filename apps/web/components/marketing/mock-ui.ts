/** Shared light-theme classes for product mock / widget visuals. */
export const mock = {
  shell: 'overflow-hidden bg-surface text-text-primary',
  chrome:
    'flex items-center gap-2 border-b border-line bg-surface-muted px-3 py-2',
  chromeTitle: 'ml-2 flex-1 truncate text-[10px] text-text-tertiary',
  panel: 'bg-canvas-warm',
  panelInner: 'bg-surface p-4',
  row: 'flex items-center gap-3 rounded-lg border border-line bg-surface px-3 py-2 shadow-card',
  rowMuted: 'rounded-lg border border-line bg-surface-muted px-3 py-2',
  metricBar: 'h-1 rounded-full bg-line',
  skeleton: 'rounded-sm bg-line',
  skeletonMid: 'rounded-sm bg-line-strong/60',
  skeletonBtn: 'rounded-md bg-ink-950/10',
  label: 'text-[10px] font-medium text-text-tertiary',
  mono: 'font-mono text-[10px] text-text-secondary',
  divider: 'border-line',
  footerGrid: 'grid grid-cols-3 gap-px border-t border-line bg-line',
  footerCell: 'bg-surface-muted px-3 py-2 text-center',
  badge:
    'rounded-full border border-line bg-surface px-2 py-0.5 text-[9px] font-medium text-signal',
  badgeLive:
    'rounded-full border border-signal/25 bg-signal/10 px-2 py-0.5 text-[9px] font-medium text-signal',
} as const

/** Heatmap overlay tuned for light mock backgrounds */
export const HEATMAP_LIGHT = `
  radial-gradient(ellipse 50% 38% at 48% 28%, rgba(232,93,36,0.22) 0%, transparent 55%),
  radial-gradient(ellipse 38% 32% at 72% 55%, rgba(245,158,11,0.18) 0%, transparent 50%),
  radial-gradient(ellipse 35% 28% at 22% 72%, rgba(29,78,216,0.12) 0%, transparent 55%)
`

export const HEATMAP_LIGHT_SOFT = `
  radial-gradient(ellipse 55% 40% at 45% 30%, rgba(232,93,36,0.2) 0%, transparent 55%),
  radial-gradient(ellipse 40% 35% at 70% 60%, rgba(245,158,11,0.15) 0%, transparent 50%),
  radial-gradient(ellipse 30% 25% at 20% 75%, rgba(29,78,216,0.1) 0%, transparent 55%)
`
