// How each generator arm is drawn: color by family, shape by the arm within it, so the pair is unique.
// The five arms in data/dashboard.json carry the same family and marker; this adds the four from Phase 3
// (ADR-029) and a fallback for any generator name the workbench is given.
import { FAMILY_COLOR } from './marks'

export const ARM_STYLE = {
  independent_marginals: { label: 'Independent marginals', short: 'Floor', family: 'floor', marker: 'square' },
  gaussian_copula: { label: 'Gaussian copula', short: 'Copula', family: 'copula', marker: 'circle' },
  gaussian_copula_shrink025: { label: 'Gaussian copula, shrink 0.25', short: 'Copula 0.25', family: 'copula', marker: 'diamond' },
  gaussian_copula_shrink010: { label: 'Gaussian copula, shrink 0.10', short: 'Copula 0.10', family: 'copula', marker: 'hexagon' },
  ctgan: { label: 'CTGAN', short: 'CTGAN', family: 'ctgan', marker: 'triangle' },
  ctgan_small: { label: 'CTGAN, small networks', short: 'CTGAN small', family: 'ctgan', marker: 'plus' },
  tvae: { label: 'TVAE', short: 'TVAE', family: 'tvae', marker: 'triangleDown' },
  tabular_diffusion: { label: 'Tabular diffusion', short: 'Diffusion', family: 'diffusion', marker: 'circle' },
  tabular_diffusion_reg: { label: 'Tabular diffusion, regularized', short: 'Diffusion reg.', family: 'diffusion', marker: 'diamond' },
}

const FAMILY_BY_PREFIX = [
  ['independent', 'floor'],
  ['gaussian_copula', 'copula'],
  ['copula', 'copula'],
  ['ctgan', 'ctgan'],
  ['tvae', 'tvae'],
  ['tabular_diffusion', 'diffusion'],
  ['diffusion', 'diffusion'],
]

export function armStyle(name) {
  const known = ARM_STYLE[name]
  if (known) return { id: name, ...known, color: FAMILY_COLOR[known.family] }
  const family = FAMILY_BY_PREFIX.find(([p]) => String(name).toLowerCase().startsWith(p))?.[1] ?? 'other'
  return { id: name, label: name, short: name, family, marker: family === 'other' ? 'plus' : 'circle', color: FAMILY_COLOR[family], unknown: true }
}
