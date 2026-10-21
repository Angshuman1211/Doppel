// The groups the page's sentences are about. Every number quoted for a group is computed from the data.
// The three arms that leak are TVAE and both diffusion settings: two copy their training rows by Track 1's own
// check, and the regularized diffusion setting passed that check yet leaks second most (final_evaluation_result.md).
export const LEAKY = ['tabular_diffusion', 'tabular_diffusion_reg', 'tvae']
export const NOT_LEAKY = ['independent_marginals', 'gaussian_copula', 'gaussian_copula_shrink025', 'gaussian_copula_shrink010', 'ctgan', 'ctgan_small']
export const COPULAS = ['gaussian_copula', 'gaussian_copula_shrink025', 'gaussian_copula_shrink010']
export const NEURAL = ['ctgan', 'ctgan_small', 'tvae', 'tabular_diffusion', 'tabular_diffusion_reg']

export const range = (ids, get) => {
  const v = ids.map(get)
  return [Math.min(...v), Math.max(...v)]
}
export const argmax = (ids, get) => ids.reduce((best, id) => (get(id) > get(best) ? id : best), ids[0])
export const argmin = (ids, get) => ids.reduce((best, id) => (get(id) < get(best) ? id : best), ids[0])
export const pct0 = (x) => `${Math.round(x * 100)}%`

// A domain that holds every mean ± sd and every reference value, rounded out to tick values.
export function domainFor(values, refs = [], { floorZero = false, count = 5, niceTicks }) {
  const lo = Math.min(...values.map((v) => v.mean - v.sd), ...refs)
  const hi = Math.max(...values.map((v) => v.mean + v.sd), ...refs)
  const pad = (hi - lo) * 0.06
  const ticks = niceTicks(floorZero ? 0 : lo - pad, hi + pad, count)
  return { domain: [Math.min(floorZero ? 0 : lo - pad, ticks[0]), Math.max(hi + pad, ticks[ticks.length - 1])], ticks }
}
