import {
  CreditCard,
  FileText,
  Home,
  Info,
  LayoutGrid,
  Sparkles,
  type LucideIcon,
} from 'lucide-react'
import clsx from 'clsx'

const ROUTE_ICONS: Record<string, LucideIcon> = {
  '/': Home,
  '/features': Sparkles,
  '/pricing': CreditCard,
  '/about': Info,
}

type PageRouteIconProps = {
  path: string
  className?: string
  iconClassName?: string
  size?: 'sm' | 'md'
}

const sizeMap = {
  sm: { box: 'h-8 w-10', icon: 14 },
  md: { box: 'h-9 w-12', icon: 16 },
}

/** Route thumbnail with a meaningful icon — no empty gradient boxes. */
export function PageRouteIcon({
  path,
  className,
  iconClassName,
  size = 'md',
}: PageRouteIconProps) {
  const Icon = ROUTE_ICONS[path] ?? FileText
  const s = sizeMap[size]

  return (
    <div
      className={clsx(
        'flex shrink-0 items-center justify-center rounded-lg border border-line bg-surface-muted',
        s.box,
        className
      )}
      title={path}
    >
      <Icon size={s.icon} className={clsx('text-signal', iconClassName)} aria-hidden />
    </div>
  )
}

type MiniFeatureTileProps = {
  icon: LucideIcon
  label: string
  accent?: 'signal' | 'neural' | 'copy'
  className?: string
}

const accentMap = {
  signal: 'text-signal bg-signal/10 border-signal/30',
  neural: 'text-neural bg-neural/10 border-neural/30',
  copy: 'text-copy bg-copy/10 border-copy/30',
}

/** Small labeled tile with icon for mock UIs (hero, heatmap previews). */
export function MiniFeatureTile({
  icon: Icon,
  label,
  accent = 'signal',
  className,
}: MiniFeatureTileProps) {
  return (
    <div
      className={clsx(
        'flex h-14 flex-col items-center justify-center gap-1 rounded-lg border bg-surface',
        accentMap[accent],
        className
      )}
    >
      <Icon size={16} aria-hidden />
      <span className="font-mono text-[7px] uppercase tracking-wide opacity-80">{label}</span>
    </div>
  )
}

export { LayoutGrid }
