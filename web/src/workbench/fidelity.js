// Fidelity of a synthetic frame against a real one. Two definitions, both from the Python side:
//   officialFidelity  — evaluation/fidelity_metrics.run_fidelity_report (Track 2), the headline numbers
//   fidelityTable     — dashboard/app.py's per-column table, its column rules and its profile-match check
// The numeric steps copy numpy, scipy and pandas exactly so results agree with the Python to rounding.
import { isNumericDtype, pyStr } from './csv.js'

export const ID_COLUMNS = ['subject_id', 'hadm_id', 'icd9_codes', 'split']
export const CATEGORICAL_COLS = ['admission_type', 'ethnicity', 'first_careunit', 'gender', 'icd9_primary']
export const CONTINUOUS_COLS = ['age', 'los_hospital_days', 'los_icu_days', 'n_diagnoses']
export const BINARY_COLS = ['hospital_expire_flag', 'readmit_30d', 'age_89_plus']
export const NUMERIC_COLS = [...CONTINUOUS_COLS, 'hospital_expire_flag', 'readmit_30d']

const present = (values) => values.filter((v) => v !== null && !(typeof v === 'number' && Number.isNaN(v)))
const toNum = (v) => (typeof v === 'boolean' ? Number(v) : v)

// numpy's linear quantile: virtual index q*(n-1), interpolated with numpy's _lerp.
function lerp(a, b, t) {
  const d = b - a
  return t >= 0.5 ? b - d * (1 - t) : a + d * t
}
export function quantile(sorted, q) {
  const n = sorted.length
  const v = q * (n - 1)
  const lo = Math.floor(v)
  const hi = Math.min(lo + 1, n - 1)
  return lerp(sorted[lo], sorted[hi], v - lo)
}
const linspace11 = Array.from({ length: 11 }, (_, i) => (i === 10 ? 1 : i * 0.1))

// np.histogram on fixed edges: half-open bins, the last one closed.
function histogram(values, edges) {
  const counts = new Array(edges.length - 1).fill(0)
  const last = edges.length - 1
  for (const v of values) {
    if (v < edges[0] || v > edges[last]) continue
    if (v === edges[last]) {
      counts[last - 1]++
      continue
    }
    let lo = 0
    let hi = last
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (edges[mid] <= v) lo = mid
      else hi = mid
    }
    counts[lo]++
  }
  return counts
}

// scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2
export function jsBits(p, q) {
  const sp = p.reduce((a, b) => a + b, 0)
  const sq = q.reduce((a, b) => a + b, 0)
  let left = 0
  let right = 0
  for (let i = 0; i < p.length; i++) {
    const pi = p[i] / sp
    const qi = q[i] / sq
    const m = (pi + qi) / 2
    if (pi > 0) left += pi * Math.log(pi / m)
    if (qi > 0) right += qi * Math.log(qi / m)
  }
  const js = (left + right) / Math.log(2)
  const d = Math.sqrt(js / 2)
  return d * d
}

// Shares of each category, ordered the way the Python reindexes them (sorted union of both sides).
function categoricalShares(real, synth, realType, synthType) {
  const r = present(real).map((v) => pyStr(v, realType))
  const s = present(synth).map((v) => pyStr(v, synthType))
  const levels = [...new Set([...r, ...s])].sort()
  const count = (arr) => {
    const m = new Map()
    for (const v of arr) m.set(v, (m.get(v) || 0) + 1)
    return levels.map((l) => (m.get(l) || 0) / Math.max(arr.length, 1))
  }
  return [count(r), count(s)]
}

export function jsCategorical(real, synth, realType, synthType) {
  const [p, q] = categoricalShares(real, synth, realType, synthType)
  return jsBits(p, q)
}

export function jsContinuous(real, synth, bins = 10) {
  const r = present(real).map(toNum)
  const s = present(synth).map(toNum)
  const combined = [...r, ...s].sort((a, b) => a - b)
  if (!combined.length) return NaN
  const edges = [...new Set(linspace11.slice(0, bins + 1).map((q) => quantile(combined, q)))].sort((a, b) => a - b)
  if (edges.length < 2) return 0
  const rc = histogram(r, edges)
  const sc = histogram(s, edges)
  const rs = Math.max(rc.reduce((a, b) => a + b, 0), 1)
  const ss = Math.max(sc.reduce((a, b) => a + b, 0), 1)
  return jsBits(rc.map((c) => c / rs), sc.map((c) => c / ss))
}

// pandas DataFrame.corr(): pairwise-complete Pearson by Welford's method, clipped to [-1, 1].
export function corrMatrix(frame, cols) {
  for (const c of cols) {
    if (!isNumericDtype(frame.dtypes[c])) throw new Error(`could not convert column '${c}' to numbers for the correlation`)
  }
  const mat = cols.map((c) => frame.data[c].map((v) => (v === null ? NaN : toNum(v))))
  const k = cols.length
  const out = Array.from({ length: k }, () => new Array(k).fill(NaN))
  for (let xi = 0; xi < k; xi++) {
    for (let yi = 0; yi <= xi; yi++) {
      let nobs = 0
      let meanx = 0
      let meany = 0
      let ssqx = 0
      let ssqy = 0
      let cov = 0
      const X = mat[xi]
      const Y = mat[yi]
      for (let i = 0; i < X.length; i++) {
        const vx = X[i]
        const vy = Y[i]
        if (!Number.isFinite(vx) || !Number.isFinite(vy)) continue
        nobs++
        const dx = vx - meanx
        const dy = vy - meany
        meanx += dx / nobs
        meany += dy / nobs
        ssqx += (vx - meanx) * dx
        ssqy += (vy - meany) * dy
        cov += (vx - meanx) * dy
      }
      let r = NaN
      if (nobs >= 1) {
        const div = Math.sqrt(ssqx * ssqy)
        if (div !== 0) r = Math.min(1, Math.max(-1, cov / div))
      }
      out[xi][yi] = r
      out[yi][xi] = r
    }
  }
  return out
}

// Mean absolute difference of the upper triangles, NaN counted as 0 (np.nan_to_num).
export function correlationGap(real, synth, cols) {
  const a = corrMatrix(real, cols)
  const b = corrMatrix(synth, cols)
  let sum = 0
  let n = 0
  for (let i = 0; i < cols.length; i++) {
    for (let j = i + 1; j < cols.length; j++) {
      const x = Number.isNaN(a[i][j]) ? 0 : a[i][j]
      const y = Number.isNaN(b[i][j]) ? 0 : b[i][j]
      sum += Math.abs(x - y)
      n++
    }
  }
  return n ? sum / n : NaN
}

const labCols = (frame) => frame.columns.filter((c) => c.startsWith('lab_'))
const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN)

// evaluation/fidelity_metrics.run_fidelity_report, without the KS test (the Streamlit page does not show it).
export function officialFidelity(real, synth) {
  const has = (c) => real.columns.includes(c) && synth.columns.includes(c)
  const labs = labCols(real)
  const perColumn = {}
  for (const c of [...CATEGORICAL_COLS, ...BINARY_COLS].filter(has)) {
    perColumn[c] = jsCategorical(real.data[c], synth.data[c], real.dtypes[c], synth.dtypes[c])
  }
  for (const c of [...CONTINUOUS_COLS, ...labs].filter(has)) {
    if (!isNumericDtype(real.dtypes[c]) || !isNumericDtype(synth.dtypes[c])) throw new Error(`column '${c}' is not numeric`)
    perColumn[c] = jsContinuous(real.data[c], synth.data[c])
  }
  const corrCols = [...NUMERIC_COLS, ...labs].filter(has)
  return { meanJs: mean(Object.values(perColumn)), corrGap: correlationGap(real, synth, corrCols), perColumn }
}

// dashboard/app.py comparable_columns: categorical = text, or two values or fewer in the real data.
export function comparableColumns(real, synth) {
  const shared = real.columns.filter((c) => synth.columns.includes(c) && !ID_COLUMNS.includes(c))
  const categorical = shared.filter(
    (c) => real.dtypes[c] === 'object' || new Set(present(real.data[c]).map((v) => pyStr(v, real.dtypes[c]))).size <= 2,
  )
  const numeric = shared.filter((c) => !categorical.includes(c) && isNumericDtype(real.dtypes[c]))
  return { categorical, numeric }
}

// dashboard/app.py fidelity_table: per-column JS divergence, worst first.
export function fidelityTable(real, synth) {
  const { categorical, numeric } = comparableColumns(real, synth)
  const rows = [
    ...categorical.map((c) => ({ column: c, kind: 'categorical', js: jsCategorical(real.data[c], synth.data[c], real.dtypes[c], synth.dtypes[c]) })),
    ...numeric.map((c) => ({
      column: c,
      kind: 'numeric',
      js: isNumericDtype(synth.dtypes[c]) ? jsContinuous(real.data[c], synth.data[c]) : NaN,
    })),
  ]
  return rows.sort((a, b) => (Number.isNaN(a.js) ? 1 : Number.isNaN(b.js) ? -1 : b.js - a.js))
}

// dashboard/app.py profile_match_rate: share of candidate rows whose categorical profile matches some real row.
export function profileMatchRate(real, candidate, cols) {
  if (!cols.length || !candidate.n) return NaN
  const key = (frame, i) => JSON.stringify(cols.map((c) => pyStr(frame.data[c][i], frame.dtypes[c])))
  const realKeys = new Set()
  for (let i = 0; i < real.n; i++) realKeys.add(key(real, i))
  let hits = 0
  for (let i = 0; i < candidate.n; i++) if (realKeys.has(key(candidate, i))) hits++
  return hits / candidate.n
}

// Series.describe() for a numeric column.
export function describe(values) {
  const v = present(values).map(toNum)
  const n = v.length
  const s = [...v].sort((a, b) => a - b)
  const m = mean(v)
  const variance = n > 1 ? v.reduce((a, x) => a + (x - m) * (x - m), 0) / (n - 1) : NaN
  return {
    count: n,
    mean: m,
    std: Math.sqrt(variance),
    min: n ? s[0] : NaN,
    '25%': n ? quantile(s, 0.25) : NaN,
    '50%': n ? quantile(s, 0.5) : NaN,
    '75%': n ? quantile(s, 0.75) : NaN,
    max: n ? s[n - 1] : NaN,
  }
}

// Series.astype(str).value_counts(normalize=True): missing counts as the category 'nan', as in the Python.
export function valueShares(values, dtype) {
  const m = new Map()
  for (const v of values) {
    const k = pyStr(v, dtype)
    m.set(k, (m.get(k) || 0) + 1)
  }
  const out = {}
  for (const [k, c] of m) out[k] = c / values.length
  return out
}
