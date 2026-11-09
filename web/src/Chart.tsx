// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { useState } from 'react'

export type Series = { name: string; color: string; dashed?: boolean; points: { x: number; y: number; note?: string }[] }

type Props = {
  series: Series[]
  xLabel: string
  yLabel: string
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  yMin?: number
  yMax?: number
  height?: number
  target?: number
}

/** Small dependency-free line chart: accuracy against dollars. */
export function LineChart({ series, xLabel, yLabel, xFormat = String, yFormat = (y) => `${Math.round(y * 100)}%`, yMin, yMax, height = 260, target }: Props) {
  const [hover, setHover] = useState<{ x: number; y: number; s: string; px: number; py: number; note?: string } | null>(null)
  const W = 480
  const H = height
  const pad = { l: 46, r: 14, t: 12, b: 34 }
  const all = series.flatMap((s) => s.points)
  if (all.length === 0) {
    return <div className="flex h-40 items-center justify-center text-sm text-mute">No measurements yet</div>
  }
  const xs = all.map((p) => p.x)
  const ys = all.map((p) => p.y).concat(target != null ? [target] : [])
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs, x0 + 1)
  const lo = yMin ?? Math.max(0, Math.floor((Math.min(...ys) - 0.03) * 20) / 20)
  const hi = yMax ?? Math.min(1, Math.ceil((Math.max(...ys) + 0.02) * 20) / 20)
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r)
  const sy = (y: number) => pad.t + (1 - (y - lo) / (hi - lo || 1)) * (H - pad.t - pad.b)
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4)
  const xTicks = Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4)

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke="#1f2b45" />
            <text x={pad.l - 8} y={sy(t) + 4} textAnchor="end" className="fill-[#8a97b2] text-[12px]">
              {yFormat(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - pad.b + 16} textAnchor="middle" className="fill-[#8a97b2] text-[12px]">
            {xFormat(t)}
          </text>
        ))}
        <text x={W / 2} y={H - 4} textAnchor="middle" className="fill-[#8a97b2] text-[12px]">
          {xLabel}
        </text>
        {target != null && target <= hi && target >= lo && (
          <g>
            <line x1={pad.l} x2={W - pad.r} y1={sy(target)} y2={sy(target)} stroke="#eab308" strokeDasharray="4 4" opacity={0.7} />
            <text x={W - pad.r} y={sy(target) - 5} textAnchor="end" className="fill-[#eab308] text-[12px]">
              target {yFormat(target)}
            </text>
          </g>
        )}
        {series.map((s) => (
          <g key={s.name}>
            <polyline
              fill="none"
              stroke={s.color}
              strokeWidth={2.2}
              strokeDasharray={s.dashed ? '5 5' : undefined}
              points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
            />
            {s.points.map((p, i) => (
              <circle
                key={i}
                cx={sx(p.x)}
                cy={sy(p.y)}
                r={hover && hover.s === s.name && hover.x === p.x ? 5 : 3}
                fill={s.color}
                onMouseEnter={() => setHover({ x: p.x, y: p.y, s: s.name, px: sx(p.x) / W, py: sy(p.y) / H, note: p.note })}
                onMouseLeave={() => setHover(null)}
              />
            ))}
          </g>
        ))}
      </svg>
      {hover && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border border-line bg-ink px-2 py-1 text-xs shadow-lg"
          style={{ left: `${hover.px * 100}%`, top: `${hover.py * 100}%` }}
        >
          <div className="font-medium">{hover.s}</div>
          <div className="text-mute">
            {xFormat(hover.x)} · {yFormat(hover.y)}
          </div>
          {hover.note && <div className="text-mute">{hover.note}</div>}
        </div>
      )}
      <div className="mt-1 flex flex-wrap gap-4 px-2 text-xs text-mute">
        {series.map((s) => (
          <span key={s.name} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}
