// A marker shape drawn around (cx, cy). The shape is the second identity channel, next to color.
export const FAMILY_COLOR = {
  floor: 'var(--c-floor)',
  copula: 'var(--c-blue)',
  ctgan: 'var(--c-orange)',
  tvae: 'var(--c-aqua)',
  diffusion: 'var(--c-plum)',
  other: 'var(--muted)',
}

export function shapePath(shape, r) {
  switch (shape) {
    case 'square':
      return `M${-r} ${-r}H${r}V${r}H${-r}Z`
    case 'diamond':
      return `M0 ${-r * 1.25}L${r * 1.25} 0L0 ${r * 1.25}L${-r * 1.25} 0Z`
    case 'triangle':
      return `M0 ${-r * 1.2}L${r * 1.15} ${r * 0.9}H${-r * 1.15}Z`
    case 'triangleDown':
      return `M0 ${r * 1.2}L${r * 1.15} ${-r * 0.9}H${-r * 1.15}Z`
    case 'hexagon': {
      const k = r * 1.12
      const pts = [0, 1, 2, 3, 4, 5].map((i) => [k * Math.cos((Math.PI / 3) * i), k * Math.sin((Math.PI / 3) * i)])
      return 'M' + pts.map((pt) => pt.map((v) => v.toFixed(2)).join(' ')).join('L') + 'Z'
    }
    case 'plus': {
      const w = r * 0.42
      const k = r * 1.15
      return `M${-w} ${-k}H${w}V${-w}H${k}V${w}H${w}V${k}H${-w}V${w}H${-k}V${-w}H${-w}Z`
    }
    default:
      return null
  }
}

// In-chart marker with a 2px surface ring so overlapping marks stay legible.
export function Marker({ shape, x, y, r = 6, color, ring = true, hollow = false }) {
  const common = {
    fill: hollow ? 'var(--surface)' : color,
    stroke: hollow ? color : ring ? 'var(--surface)' : 'none',
    strokeWidth: hollow ? 2 : 2,
    paintOrder: 'stroke',
  }
  if (shape === 'circle' || !shapePath(shape, r)) return <circle cx={x} cy={y} r={r} {...common} />
  return <path transform={`translate(${x} ${y})`} d={shapePath(shape, r)} {...common} />
}

// Inline swatch for legends, tables and tooltips.
export function Swatch({ shape, color, size = 14 }) {
  const c = size / 2
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true" style={{ flex: 'none' }}>
      <Marker shape={shape} x={c} y={c} r={size * 0.32} color={color} ring={false} />
    </svg>
  )
}
