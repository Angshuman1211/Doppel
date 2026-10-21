// The Streamlit app's evaluation tab: load_report, the per-generator means, and the Pareto frontier.

// Python's str(KeyError('x')) is "'x'"; the other errors print their message.
class ReportError extends Error {}
const need = (obj, key) => {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj) || !(key in obj)) throw new ReportError(`'${key}'`)
  return obj[key]
}
const toFloat = (v) => {
  if (typeof v === 'number') return v
  if (typeof v === 'boolean') return Number(v)
  if (typeof v === 'string' && v.trim() !== '' && !Number.isNaN(Number(v))) return Number(v)
  throw new ReportError(`could not convert ${typeof v === 'string' ? `string to float: '${v}'` : 'value to float'}`)
}

export function loadReport(name, text) {
  try {
    let result
    try {
      result = JSON.parse(text)
    } catch (e) {
      throw new ReportError(e.message)
    }
    const metrics = need(result, 'metrics')
    const tstr = need(need(metrics, 'utility'), 'tstr_auroc')
    const values = tstr && typeof tstr === 'object' ? Object.values(tstr) : []
    if (!values.length) throw new ReportError('no TSTR AUROC values in the report')
    // The evaluation contract has no top-level seed field, so it comes out of run_id (gaussian_copula_seed42).
    const label = String(result.run_id ?? name)
    return {
      file: name,
      generator: result.generator_name ?? name,
      seed: String(result.seed ?? (label.includes('_seed') ? label.slice(label.lastIndexOf('_seed') + 5) : 'unknown')),
      fidelityJs: toFloat(need(need(metrics, 'fidelity'), 'mean_js_divergence')),
      utilityTstr: values.reduce((a, v) => a + toFloat(v), 0) / values.length,
      privacyWorstMia: toFloat(need(need(need(metrics, 'privacy'), 'membership_worst_case'), 'mean_attack_auroc')),
      tstrDetails: tstr,
      metrics,
      error: null,
    }
  } catch (e) {
    if (e instanceof ReportError) return { file: name, generator: name, error: e.message }
    throw e
  }
}

// True when a is at least as good as b on fidelity, utility and privacy, and better on one.
export function dominates(a, b) {
  const av = [a.fidelityJs, -a.utilityTstr, a.privacyWorstMia]
  const bv = [b.fidelityJs, -b.utilityTstr, b.privacyWorstMia]
  return av.every((x, i) => x <= bv[i]) && av.some((x, i) => x < bv[i])
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length
const sd = (xs) => {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1))
}

// One row per generator, in first-seen order, with the means the Streamlit frontier uses and the spread over seeds.
export function generatorSummary(reports) {
  const groups = new Map()
  for (const r of reports) {
    if (!groups.has(r.generator)) groups.set(r.generator, [])
    groups.get(r.generator).push(r)
  }
  const rows = [...groups].map(([generator, rs]) => ({
    generator,
    seeds: rs.length,
    fidelityJs: mean(rs.map((r) => r.fidelityJs)),
    utilityTstr: mean(rs.map((r) => r.utilityTstr)),
    privacyWorstMia: mean(rs.map((r) => r.privacyWorstMia)),
    sd: {
      fidelityJs: sd(rs.map((r) => r.fidelityJs)),
      utilityTstr: sd(rs.map((r) => r.utilityTstr)),
      privacyWorstMia: sd(rs.map((r) => r.privacyWorstMia)),
    },
  }))
  for (const row of rows) row.onFrontier = !rows.some((o) => o.generator !== row.generator && dominates(o, row))
  return rows
}
